"""
env.py — 异构多边缘 云-边-端 DAG 任务卸载环境（Heterogeneous Multi-Edge, Method A）
对应数学文档 Section 1-2，实现 MDP (S, A, P, R, γ)

【本版改动：v3 异构多边缘 + 观测补全 + 逐位置 CCSM】

  v2 的问题（本版修复的根因）：
    动作空间扩展到 L=E+2，回程状态扩展为 Q_ec 向量，但观测空间没有同步扩展——
    观测里没有 node_avail（各计算节点空闲时刻），也没有 locations（放置历史）。
    E=1 时"在多个边缘间选择"这一决策不存在，缺失不可见；E≥2 时各边缘参数相同、
    状态不可见、历史不可见，策略在边缘之间做选择时没有任何可依据的信息，
    必然坍缩到少数索引（实测 E=3 时某边缘利用率仅 1%）。

  三处同步扩展：
    1) 异构：每个元任务为 E 个边缘独立采样算力 f_edge[e] 与回程带宽 rate_ec_max[e]
       （取值集合均值 = 原同构值；同构是本设定的退化特例）。
       采样由 DAG 结构确定性派生 → 同一任务的多次 reset 得到同一套基础设施配置，
       符合"任务 = DAG + 基础设施配置"的元学习语义。
    2) 观测补全：
       task_features   : 7 → 7 + L（新增放置位置 one-hot）
       global_features : E+1 → 4E+3（新增各边缘空闲时刻、算力、回程带宽、
                                      本地/云空闲时刻）
    3) 逐位置 CCSM：软掩码从"仅云端一维"扩展为逐位置权重（见 get_ccsm_weights）

兼容两种 DAG 类型（接口一致）：DaggenDAG/DaggenTask、DAG/Task
"""
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EnvState:
    finish_times: np.ndarray
    scheduled:    set
    done:         set
    locations:    np.ndarray
    node_avail:   np.ndarray      # 长度 E+2：[本地, 边1..边E, 云]
    Q_ec:         np.ndarray      # 长度 E：每条 边→云 回程的队列(MB)
    rate_ue:      float
    step:         int
    cloud_relay:  np.ndarray      # 长度 N：云端任务实际所走的中继边索引(0..E-1)，非云=-1
    f_edge:       np.ndarray      # 长度 E：各边缘算力 (cycles/s)  ← 异构
    rate_ec_max:  np.ndarray      # 长度 E：各回程最大带宽 (MB/s)  ← 异构


class ThreeTierEnv:
    """
    异构多边缘 云-边-端协同 DAG 任务卸载环境

    动作 a = i * L + x，L = E + 2
        x = 0            → 本地
        x = 1 .. E       → 边缘服务器 (x-1)
        x = E + 1        → 云端（自动经最空回程中继）
    奖励 r_t = -(ΔT^span)，终止条件：所有子任务调度完毕
    """

    LOC_LOCAL = 0

    def __init__(self, cfg, rng: np.random.Generator = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.dag = None
        self.state: Optional[EnvState] = None

        self.E = int(getattr(cfg.env, "num_edges", 1))
        self.L = self.E + 2
        self.LOC_CLOUD = self.E + 1

    # ── 位置语义辅助 ──────────────────────────────────────
    def _is_edge(self, x: int) -> bool:
        return 1 <= x <= self.E

    def _edge_idx(self, x: int) -> int:
        """位置码 x(1..E) → 边缘数组下标(0..E-1)"""
        return x - 1

    # ── 异构基础设施配置 ──────────────────────────────────
    @staticmethod
    def _task_signature(dag) -> int:
        """
        由 DAG 结构确定性派生一个整数指纹。

        用途：同一个 DAG（= 同一个元任务）在多次 reset 中必须得到同一套
        基础设施配置，否则内循环的"任务内适配"语义就不成立。
        """
        acc = (dag.N * 1000003) & 0xFFFFFFFF
        for t in dag.tasks:
            acc = (acc * 31 + int(t.cpu)) & 0xFFFFFFFF
            acc = (acc * 31 + int(t.data_in * 1e6)) & 0xFFFFFFFF
        return int(acc)

    def _sample_infra(self, dag) -> Tuple[np.ndarray, np.ndarray]:
        """为当前元任务采样 E 个边缘的算力与回程带宽"""
        ec = self.cfg.env
        E = self.E
        if not getattr(ec, "heterogeneous", False):
            return (np.full(E, float(ec.f_edge)),
                    np.full(E, float(ec.rate_ec_max)))
        r = np.random.default_rng(self._task_signature(dag))
        f_edge = r.choice(np.asarray(ec.f_edge_options, dtype=float), size=E)
        rate_ec = r.choice(np.asarray(ec.rate_ec_max_options, dtype=float), size=E)
        return f_edge.astype(float), rate_ec.astype(float)

    # ── 公共接口 ──────────────────────────────────────────
    def reset(self, dag, q_ec_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        q_ec_range: (lo, hi) 指定各回程初始队列的采样区间（MB），用于拥塞实验；
                    None 时用默认区间 [0, 0.3*Q_max]。每条回程独立采样。
        """
        self.dag = dag
        ec = self.cfg.env

        if q_ec_range is None:
            q_init = self.rng.uniform(0.0, ec.Q_ec_max * 0.3, size=self.E)
        else:
            lo, hi = q_ec_range
            q_init = self.rng.uniform(lo, hi, size=self.E)

        f_edge, rate_ec_max = self._sample_infra(dag)

        self.state = EnvState(
            finish_times = np.zeros(dag.N),
            scheduled    = set(),
            done         = set(),
            locations    = np.full(dag.N, -1, dtype=int),
            node_avail   = np.zeros(self.L),
            Q_ec         = q_init.astype(float),
            rate_ue      = float(self.rng.choice(ec.rate_ue_options)),
            step         = 0,
            cloud_relay  = np.full(dag.N, -1, dtype=int),
            f_edge       = f_edge,
            rate_ec_max  = rate_ec_max,
        )
        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        task_idx, location = divmod(action, self.L)
        assert self._is_valid(task_idx, location), \
            f"非法动作: task={task_idx} loc={location} (L={self.L})"

        prev_span = self._makespan()
        self._schedule(task_idx, location)
        new_span  = self._makespan()
        reward    = -(new_span - prev_span)

        self.state.step += 1
        done = (len(self.state.scheduled) == self.dag.N)
        info = {"makespan": new_span, "step": self.state.step,
                "Q_ec": self.state.Q_ec.copy()}
        if done:
            info["final_makespan"] = new_span
            info["locations"] = self.state.locations.copy()   # 供动作分布诊断

        return self._get_obs(), reward, done, info

    def get_valid_action_mask(self) -> np.ndarray:
        """硬掩码：shape = [N*L]，True 表示合法（就绪任务的全部 L 个位置）"""
        mask = np.zeros(self.dag.N * self.L, dtype=bool)
        for i in self.dag.get_ready_tasks(self.state.done, self.state.scheduled):
            mask[i * self.L: i * self.L + self.L] = True
        return mask

    def get_ccsm_weights(self) -> np.ndarray:
        """
        逐位置拥塞感知软掩码 (CCSM)：shape = [N*L]

            w_local  = 1
            w_edge_e = exp( -alpha_edge · (avail_e − min_j avail_j) / T_ref )
            w_cloud  = exp( -alpha      · min_e Q_ec[e] / Q_ec_max )

        其中 T_ref = max_j avail_j（仅在边缘之间取参考，量纲自洽，取值∈[0,1]）。

        设计边界：只使用与具体子任务无关的拥塞状态（排队相对差额、回程占用），
        不引入子任务大小做逐任务完成时刻估计——保持"先验/软掩码"定位，
        避免退化为启发式调度器。

        退化关系：
            alpha_edge = 0            → 仅云端一维先验（v2 行为）
            alpha = alpha_edge = 0    → 无 CCSM
            E = 1                     → 边缘项恒为 1（无边缘间差额），与 v2 一致
        """
        st = self.state
        N, L, E = self.dag.N, self.L, self.E
        w = np.ones(N * L, dtype=np.float32)

        # ── 云端：回程拥塞（Method A 实际会走最空回程）────────
        alpha_c = float(self.cfg.ccsm.alpha)
        if alpha_c > 0.0:
            q_relay = float(st.Q_ec.min())
            w_cloud = float(np.exp(-alpha_c * q_relay / self.cfg.env.Q_ec_max))
            w[self.LOC_CLOUD::L] = w_cloud

        # ── 边缘：各边缘计算排队相对最空边缘的差额 ─────────────
        alpha_e = float(getattr(self.cfg.ccsm, "alpha_edge", 0.0))
        if alpha_e > 0.0:
            avail_e = st.node_avail[1:E + 1]
            T_ref   = max(float(avail_e.max()), 1e-8)
            d       = (avail_e - avail_e.min()) / T_ref        # ∈ [0, 1]
            w_edge  = np.exp(-alpha_e * d)
            for e in range(E):
                w[1 + e::L] = float(w_edge[e])

        return w

    # ── 时延模型 ──────────────────────────────────────────
    def _cloud_relay(self) -> int:
        """Method A：云端流量走当前队列最短的回程，返回边索引(0..E-1)"""
        return int(np.argmin(self.state.Q_ec))

    def _compute_task_exec_time(self, task, location: int, relay: int = -1) -> float:
        """
        返回 exec_time（float）
          relay：云端任务所走的中继边索引(0..E-1)；<0 时内部按最空回程自动选
        """
        ec = self.cfg.env
        st = self.state

        if location == self.LOC_LOCAL:
            return task.cpu / ec.f_local

        if self._is_edge(location):
            e = self._edge_idx(location)
            return (task.data_in  / st.rate_ue +
                    task.cpu      / st.f_edge[e] +      # ← 异构算力
                    task.data_out / ec.rate_eu)

        # CLOUD：经中继边（未指定则选最空回程）
        e_star = relay if relay >= 0 else self._cloud_relay()
        r_ec   = self._eff_ec_rate(st.Q_ec[e_star], st.rate_ec_max[e_star])
        return (task.data_in  / st.rate_ue  +
                task.data_in  / r_ec         +
                task.cpu      / ec.f_cloud   +
                task.data_out / ec.rate_ce   +
                task.data_out / ec.rate_eu)

    def _eff_ec_rate(self, q: float, rate_max: Optional[float] = None) -> float:
        """回程有效速率随队列占用衰减；rate_max 缺省时用配置参考值（向后兼容）"""
        ec = self.cfg.env
        rmax = float(rate_max) if rate_max is not None else float(ec.rate_ec_max)
        util = min(q / ec.Q_ec_max, 1.0)
        return max(rmax * (1.0 - 0.9 * util), rmax * 0.1)

    def _schedule(self, task_idx: int, location: int):
        task  = self.dag.tasks[task_idx]
        state = self.state

        pred_ready = max((state.finish_times[p] for p in task.predecessors), default=0.0)
        relay = self._cloud_relay() if location == self.LOC_CLOUD else -1
        exec_time = self._compute_task_exec_time(task, location, relay)
        ft = max(pred_ready, state.node_avail[location]) + exec_time

        state.finish_times[task_idx] = ft
        state.node_avail[location]   = ft
        state.locations[task_idx]    = location
        state.scheduled.add(task_idx)
        state.done.add(task_idx)

        # 回程队列演化（与 v2 一致，保证 E=1 严格复现）
        if location == self.LOC_CLOUD:
            state.cloud_relay[task_idx] = relay
            state.Q_ec[relay] = min(state.Q_ec[relay] + task.data_in,
                                    self.cfg.env.Q_ec_max)
        else:
            state.Q_ec = np.maximum(0.0, state.Q_ec - 0.05 * exec_time)

    def _makespan(self) -> float:
        done  = self.state.done
        exits = [t.idx for t in self.dag.exit_tasks if t.idx in done]
        return max((self.state.finish_times[i] for i in exits), default=0.0)

    def _is_valid(self, task_idx: int, location: int) -> bool:
        if not (0 <= task_idx < self.dag.N and 0 <= location < self.L):
            return False
        return (task_idx not in self.state.scheduled and
                self.dag.tasks[task_idx].is_ready(self.state.done))

    # ── 观测构建 ──────────────────────────────────────────
    def _get_obs(self) -> Dict:
        N     = self.dag.N
        ec    = self.cfg.env
        st    = self.state
        E, L  = self.E, self.L

        ft_max = max(float(st.finish_times.max()), 1e-8)
        T_ref  = max(float(st.node_avail.max()), 1e-8)

        # ── 子任务特征：7 基础 + L 放置位置 one-hot ──────────
        task_feats = np.zeros((N, 7 + L), dtype=np.float32)
        for t in self.dag.tasks:
            i = t.idx
            task_feats[i, 0] = t.cpu      / ec.cpu_high
            task_feats[i, 1] = t.data_in  / ec.data_in_high
            task_feats[i, 2] = t.data_out / ec.data_in_high
            task_feats[i, 3] = len(t.predecessors) / N
            task_feats[i, 4] = len(t.successors)   / N
            task_feats[i, 5] = st.finish_times[i] / ft_max
            task_feats[i, 6] = float(i in st.scheduled)
            loc = int(st.locations[i])
            if loc >= 0:                       # 已调度 → 放置位置 one-hot
                task_feats[i, 7 + loc] = 1.0

        # ── 全局特征：4E + 3 ─────────────────────────────────
        rate_opts = ec.rate_ue_options
        rate_norm = ((st.rate_ue - min(rate_opts)) /
                     max(max(rate_opts) - min(rate_opts), 1e-8))

        global_feats = np.concatenate([
            st.Q_ec / ec.Q_ec_max,                          # E：各回程排队
            st.node_avail[1:E + 1] / T_ref,                 # E：各边缘空闲时刻
            st.f_edge / ec.f_edge_ref,                      # E：各边缘算力（异构参数）
            st.rate_ec_max / ec.rate_ec_ref,                # E：各回程带宽（异构参数）
            np.array([
                st.node_avail[self.LOC_LOCAL] / T_ref,      # 1：本地空闲时刻
                st.node_avail[self.LOC_CLOUD] / T_ref,      # 1：云端空闲时刻
                rate_norm,                                  # 1：信道
            ], dtype=np.float32),
        ]).astype(np.float32)

        return {
            "task_features"  : task_feats,                                   # [N, 7+L]
            "status"         : np.array([float(i in st.scheduled) for i in range(N)],
                                         dtype=np.float32),
            "valid_mask"     : self.get_valid_action_mask(),                 # [N*L]
            "ccsm_weights"   : self.get_ccsm_weights(),                      # [N*L]
            "global_features": global_feats,                                 # [4E+3]
            "N"              : N,
            "L"              : self.L,
        }
