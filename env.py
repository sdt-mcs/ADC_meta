"""
env.py — 三层云-边-端 DAG 任务卸载环境
对应数学文档 Section 1-2，实现 MDP (S, A, P, R, γ)

兼容两种 DAG 类型（接口一致，无需区分）：
  DaggenDAG / DaggenTask  ← daggen_py.py（推荐，对齐论文参数）
  DAG / Task              ← dag_generator.py（旧版，保留兼容）
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
    node_avail:   np.ndarray
    Q_ec:         float
    rate_ue:      float
    step:         int


class ThreeTierEnv:
    """
    三层云-边-端协同 DAG 任务卸载环境

    动作 a = i * 3 + x，x ∈ {0=本地, 1=边缘, 2=云端}
    奖励 r_t = -(ΔT^span)，终止条件：所有子任务调度完毕
    """

    LOC_LOCAL = 0
    LOC_EDGE  = 1
    LOC_CLOUD = 2

    def __init__(self, cfg, rng: np.random.Generator = None):
        self.cfg   = cfg
        self.rng   = rng or np.random.default_rng()
        self.dag   = None
        self.state: Optional[EnvState] = None

    # ── 公共接口 ──────────────────────────────────────────

    def reset(self, dag, q_ec_range=None) -> Dict:
        self.dag = dag
        ec = self.cfg.env
        self.state = EnvState(
            finish_times = np.zeros(dag.N),
            scheduled    = set(),
            done         = set(),
            locations    = np.full(dag.N, -1, dtype=int),
            node_avail   = np.zeros(3),
            Q_ec         = float(self.rng.uniform(
                               q_ec_range[0] if q_ec_range is not None else 0.0,
                               q_ec_range[1] if q_ec_range is not None else ec.Q_ec_max * 0.3
                           )),
            rate_ue      = float(self.rng.choice(ec.rate_ue_options)),
            step         = 0,
        )
        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        task_idx, location = divmod(action, 3)
        assert self._is_valid(task_idx, location), \
            f"非法动作: task={task_idx} loc={location}"

        prev_span = self._makespan()
        self._schedule(task_idx, location)
        new_span  = self._makespan()
        reward    = -(new_span - prev_span)

        self.state.step += 1
        done = (len(self.state.scheduled) == self.dag.N)
        info = {"makespan": new_span, "step": self.state.step, "Q_ec": self.state.Q_ec}
        if done:
            info["final_makespan"] = new_span

        return self._get_obs(), reward, done, info

    def get_valid_action_mask(self) -> np.ndarray:
        """硬掩码：shape = [3N]，True 表示合法"""
        mask = np.zeros(self.dag.N * 3, dtype=bool)
        for i in self.dag.get_ready_tasks(self.state.done, self.state.scheduled):
            mask[i * 3: i * 3 + 3] = True
        return mask

    def get_ccsm_weights(self) -> np.ndarray:
        """CCSM 软掩码权重：shape = [3N]"""
        w = np.ones(self.dag.N * 3, dtype=np.float32)
        w_cloud = float(np.exp(
            -self.cfg.ccsm.alpha * self.state.Q_ec / self.cfg.env.Q_ec_max
        ))
        w[2::3] = w_cloud
        return w

    # ── 时延模型 ──────────────────────────────────────────

    def _compute_task_exec_time(self, task, location: int) -> float:
        ec = self.cfg.env
        st = self.state

        if location == self.LOC_LOCAL:
            return task.cpu / ec.f_local

        elif location == self.LOC_EDGE:
            return (task.data_in  / st.rate_ue +
                    task.cpu      / ec.f_edge   +
                    task.data_out / ec.rate_eu)

        else:  # CLOUD
            r_ec = self._eff_ec_rate(st.Q_ec)
            return (task.data_in  / st.rate_ue  +
                    task.data_in  / r_ec         +
                    task.cpu      / ec.f_cloud   +
                    task.data_out / ec.rate_ce   +
                    task.data_out / ec.rate_eu)

    def _eff_ec_rate(self, Q_ec: float) -> float:
        ec   = self.cfg.env
        util = min(Q_ec / ec.Q_ec_max, 1.0)
        return max(ec.rate_ec_max * (1.0 - 0.9 * util), ec.rate_ec_max * 0.1)

    def _schedule(self, task_idx: int, location: int):
        task  = self.dag.tasks[task_idx]
        state = self.state

        pred_ready = max((state.finish_times[p] for p in task.predecessors), default=0.0)
        exec_time  = self._compute_task_exec_time(task, location)
        ft         = max(pred_ready, state.node_avail[location]) + exec_time

        state.finish_times[task_idx] = ft
        state.node_avail[location]   = ft
        state.locations[task_idx]    = location
        state.scheduled.add(task_idx)
        state.done.add(task_idx)

        if location == self.LOC_CLOUD:
            state.Q_ec = min(state.Q_ec + task.data_in, self.cfg.env.Q_ec_max)
        else:
            state.Q_ec = max(0.0, state.Q_ec - 0.05 * exec_time)

    def _makespan(self) -> float:
        done  = self.state.done
        exits = [t.idx for t in self.dag.exit_tasks if t.idx in done]
        return max((self.state.finish_times[i] for i in exits), default=0.0)

    def _is_valid(self, task_idx: int, location: int) -> bool:
        if not (0 <= task_idx < self.dag.N and 0 <= location <= 2):
            return False
        return (task_idx not in self.state.scheduled and
                self.dag.tasks[task_idx].is_ready(self.state.done))

    # ── 观测构建 ──────────────────────────────────────────

    def _get_obs(self) -> Dict:
        N      = self.dag.N
        ec     = self.cfg.env
        state  = self.state
        ft_max = max(state.finish_times.max(), 1e-8)

        task_feats = np.zeros((N, 7), dtype=np.float32)
        for t in self.dag.tasks:
            i = t.idx
            task_feats[i, 0] = t.cpu      / ec.cpu_high
            task_feats[i, 1] = t.data_in  / ec.data_in_high
            task_feats[i, 2] = t.data_out / ec.data_in_high
            task_feats[i, 3] = len(t.predecessors) / N
            task_feats[i, 4] = len(t.successors)   / N
            task_feats[i, 5] = state.finish_times[i] / ft_max
            task_feats[i, 6] = float(i in state.scheduled)

        rate_opts = ec.rate_ue_options
        global_feats = np.array([
            state.Q_ec    / ec.Q_ec_max,
            (state.rate_ue - min(rate_opts)) / max(max(rate_opts) - min(rate_opts), 1e-8),
        ], dtype=np.float32)

        return {
            "task_features"  : task_feats,
            "status"         : np.array([float(i in state.scheduled) for i in range(N)],
                                         dtype=np.float32),
            "valid_mask"     : self.get_valid_action_mask(),
            "ccsm_weights"   : self.get_ccsm_weights(),
            "global_features": global_feats,
            "N"              : N,
        }
