"""
baseline.py — 对比基线算法（异构多边缘版 / Heterogeneous Multi-Edge）

实现了以下基线，用于论文中的性能对比实验：
  1. LocalOnly      — 全部本地执行
  2. EdgeOnly       — 全部卸载到边缘（在 E 个边中选 EFT 最小者）
  3. Greedy         — 贪心：每步选 EFT（最早完成时间）最小的 (task, location)
  4. HEFT           — Heterogeneous Earliest Finish Time 经典启发式
  5. RandomPolicy   — 随机合法动作（用于下界参考）

【v3 异构改动】
  * HEFT._heft_exec 使用各边缘真实算力 f_edge[e] 与各回程真实带宽 rate_ec_max[e]。
    口径划分（回复信 AE-2 依据）：
      - 静态且已知的基础设施参数 → 用真值。HEFT 全称即 Heterogeneous EFT，
        用平均值抹平异构性等于把基线打残，不是"公平的离线启发式"。
      - 随机且离线不可观测的量（信道 rate_ue、回程队列 Q_ec）→ 用均值/零队列估计。
  * EdgeOnly 从"选 node_avail 最小的边"改为"选 EFT 最小的边"：异构下算力差异
    可达 3 倍，只看空闲时刻会系统性低估边缘层能力，构成稻草人基线。
  * Greedy 无需改动：它调用 env._compute_task_exec_time，异构参数自动生效。
"""
import numpy as np
from typing import List, Dict, Optional
from daggen_py import DaggenPy, DaggenDAG as DAG
from env import ThreeTierEnv
from config import Config


# ─────────────────────────────────────────────────────────
#  基类
# ─────────────────────────────────────────────────────────

class BaselinePolicy:
    name: str = "Base"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        raise NotImplementedError

    def run_episode(self, dag: DAG, env: ThreeTierEnv) -> float:
        obs = env.reset(dag)
        done = False
        while not done:
            action = self.select_action(obs, env)
            obs, _, done, info = env.step(action)
        return info.get("final_makespan", 0.0)

    def evaluate(self, dags: List[DAG], env: ThreeTierEnv) -> Dict:
        makespans = [self.run_episode(dag, env) for dag in dags]
        return {
            "name": self.name,
            "mean": float(np.mean(makespans)),
            "std": float(np.std(makespans)),
            "min": float(np.min(makespans)),
            "max": float(np.max(makespans)),
            "makespans": makespans,
        }


# ─────────────────────────────────────────────────────────
#  1. 全本地执行
# ─────────────────────────────────────────────────────────

class LocalOnlyPolicy(BaselinePolicy):
    name = "LocalOnly"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        L = env.L
        for i in range(obs["N"]):
            if mask[i * L + env.LOC_LOCAL]:
                return i * L + env.LOC_LOCAL
        raise ValueError("No valid local action found")


# ─────────────────────────────────────────────────────────
#  2. 全边缘执行（异构：在 E 个边中选 EFT 最小者）
# ─────────────────────────────────────────────────────────

class EdgeOnlyPolicy(BaselinePolicy):
    name = "EdgeOnly"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        L, E = env.L, env.E
        st = env.state

        for i in range(obs["N"]):
            if not any(mask[i * L + (1 + e)] for e in range(E)):
                continue
            task = env.dag.tasks[i]
            pred_ready = max((st.finish_times[p] for p in task.predecessors),
                             default=0.0)
            best_e, best_eft = 0, float("inf")
            for e in range(E):
                x = 1 + e
                if not mask[i * L + x]:
                    continue
                exec_t = env._compute_task_exec_time(task, x)
                eft = max(pred_ready, st.node_avail[x]) + exec_t
                if eft < best_eft:
                    best_eft, best_e = eft, e
            return i * L + (1 + best_e)
        raise ValueError("No valid edge action found")


# ─────────────────────────────────────────────────────────
#  3. 随机策略
# ─────────────────────────────────────────────────────────

class RandomPolicy(BaselinePolicy):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        valid = np.where(obs["valid_mask"])[0]
        return int(self.rng.choice(valid))


# ─────────────────────────────────────────────────────────
#  4. 贪心策略（EFT，论文定义）
# ─────────────────────────────────────────────────────────

class GreedyPolicy(BaselinePolicy):
    """
    贪心：每步选择能最小化该子任务 EFT（最早完成时间）的 (task, location)。
    EFT = max(前驱就绪, 目标节点空闲) + 执行时延（用 env 实时状态，异构参数自动生效）。
    """
    name = "Greedy"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        N, L = obs["N"], env.L
        st = env.state
        best_action, best_eft = -1, float("inf")

        for i in range(N):
            task = env.dag.tasks[i]
            pred_ready = max((st.finish_times[p] for p in task.predecessors),
                             default=0.0)
            for x in range(L):
                a = i * L + x
                if not mask[a]:
                    continue
                exec_t = env._compute_task_exec_time(task, x)
                eft = max(pred_ready, st.node_avail[x]) + exec_t
                if eft < best_eft:
                    best_eft, best_action = eft, a

        return best_action


# ─────────────────────────────────────────────────────────
#  5. HEFT（Heterogeneous Earliest Finish Time）
# ─────────────────────────────────────────────────────────

class HEFTPolicy(BaselinePolicy):
    """
    HEFT 启发式（经典离线基线）
      1. 为每个子任务计算 upward rank（L 个位置执行时延的平均）
      2. 按 rank 降序确定调度顺序
      3. 每个子任务选择 EFT 最小的位置（在全部 L 个位置上比较）

    参数口径（AE-2）：
      * 静态已知的异构基础设施参数 f_edge[e] / rate_ec_max[e] → 用真值
      * 随机且离线不可观测的 rate_ue / Q_ec → 用均值、零队列估计
    """
    name = "HEFT"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._priority_order: Optional[List[int]] = None
        self._priority_ptr: int = 0

    def run_episode(self, dag: DAG, env: ThreeTierEnv) -> float:
        obs = env.reset(dag)
        self._priority_order = self._compute_priority(dag, env)
        self._priority_ptr = 0
        done = False
        while not done:
            action = self.select_action(obs, env)
            obs, _, done, info = env.step(action)
        return info.get("final_makespan", 0.0)

    def _heft_exec(self, task, x: int, env: ThreeTierEnv) -> float:
        """某位置的离线执行时延估计（见类文档的参数口径说明）"""
        ec = self.cfg.env
        st = env.state
        avg_rate_ue = float(np.mean(list(ec.rate_ue_options)))

        if x == env.LOC_LOCAL:
            return task.cpu / ec.f_local

        if 1 <= x <= env.E:
            e = x - 1
            return (task.data_in / avg_rate_ue
                    + task.cpu / st.f_edge[e]              # ← 该边缘真实算力
                    + task.data_out / ec.rate_eu)

        # 云端：回程按"零队列 + 各回程带宽均值"估计
        mean_rate_ec = float(np.mean(st.rate_ec_max))
        r_ec = env._eff_ec_rate(0.0, mean_rate_ec)
        return (task.data_in / avg_rate_ue
                + task.data_in / r_ec
                + task.cpu / ec.f_cloud
                + task.data_out / ec.rate_ce
                + task.data_out / ec.rate_eu)

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        L = env.L
        st = env.state

        task_idx = None
        for ti in self._priority_order[self._priority_ptr:]:
            if any(mask[ti * L + x] for x in range(L)):
                task_idx = ti
                self._priority_ptr = self._priority_order.index(ti) + 1
                break
        if task_idx is None:
            return int(np.where(mask)[0][0])

        task = env.dag.tasks[task_idx]
        pred_ready = max((st.finish_times[p] for p in task.predecessors), default=0.0)

        best_action, best_eft = task_idx * L, float("inf")
        for x in range(L):
            a = task_idx * L + x
            if not mask[a]:
                continue
            exec_t = self._heft_exec(task, x, env)
            eft = max(pred_ready, st.node_avail[x]) + exec_t
            if eft < best_eft:
                best_eft, best_action = eft, a

        return best_action

    def _compute_priority(self, dag: DAG, env: ThreeTierEnv) -> List[int]:
        """upward rank：L 个位置执行时延的平均（异构参数已在 _heft_exec 中生效）"""
        ec = self.cfg.env
        N = dag.N
        rank = np.zeros(N)
        avg_exec = np.zeros(N)
        avg_rate_ue = float(np.mean(list(ec.rate_ue_options)))

        for t in dag.tasks:
            exec_list = [self._heft_exec(t, x, env) for x in range(env.L)]
            avg_exec[t.idx] = float(np.mean(exec_list))

        topo = dag.topological_order()
        for v in reversed(topo):
            task = dag.tasks[v]
            if not task.successors:
                rank[v] = avg_exec[v]
            else:
                comm = task.data_out / avg_rate_ue
                rank[v] = avg_exec[v] + max(comm + rank[s] for s in task.successors)
        return list(np.argsort(-rank))


# ─────────────────────────────────────────────────────────
#  评估入口
# ─────────────────────────────────────────────────────────

def run_all_baselines(dags: List[DAG], cfg: Config, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    env = ThreeTierEnv(cfg, rng)

    policies = [
        LocalOnlyPolicy(),
        EdgeOnlyPolicy(),
        RandomPolicy(seed=seed),
        GreedyPolicy(),
        HEFTPolicy(cfg),
    ]

    results = []
    for policy in policies:
        result = policy.evaluate(dags, env)
        results.append(result)
        print(f"  {policy.name:12s}: mean={result['mean']:.3f}s  std={result['std']:.3f}s")
    return results
