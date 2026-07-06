"""
baseline.py — 对比基线算法

实现了以下基线，用于论文中的性能对比实验：
  1. LocalOnly      — 全部本地执行
  2. EdgeOnly       — 全部卸载到边缘
  3. Greedy         — 贪心：每步选执行时间最短的 (task, location) 组合
  4. HEFT           — Heterogeneous Earliest Finish Time 经典启发式
  5. RandomPolicy   — 随机合法动作（用于下界参考）
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
    """所有基线策略的基类"""
    name: str = "Base"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        raise NotImplementedError

    def run_episode(self, dag: DAG, env: ThreeTierEnv) -> float:
        """运行一个完整 episode，返回最终 makespan"""
        obs = env.reset(dag)
        done = False
        while not done:
            action = self.select_action(obs, env)
            obs, _, done, info = env.step(action)
        return info.get("final_makespan", 0.0)

    def evaluate(self, dags: List[DAG], env: ThreeTierEnv) -> Dict:
        """在多个 DAG 上评估，返回统计结果"""
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
    """所有子任务全部本地执行（x_i=0）"""
    name = "LocalOnly"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        # 找到合法任务中第一个，选本地执行（action = i*3+0）
        for i in range(obs["N"]):
            if mask[i * 3]:  # 本地执行动作
                return i * 3
        raise ValueError("No valid local action found")


# ─────────────────────────────────────────────────────────
#  2. 全边缘执行
# ─────────────────────────────────────────────────────────

class EdgeOnlyPolicy(BaselinePolicy):
    """所有子任务全部卸载到边缘服务器（x_i=1）"""
    name = "EdgeOnly"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        for i in range(obs["N"]):
            if mask[i * 3 + 1]:  # 边缘执行动作
                return i * 3 + 1
        raise ValueError("No valid edge action found")


# ─────────────────────────────────────────────────────────
#  3. 随机策略
# ─────────────────────────────────────────────────────────

class RandomPolicy(BaselinePolicy):
    """均匀随机从合法动作中选择"""
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        valid = np.where(obs["valid_mask"])[0]
        return int(self.rng.choice(valid))


# ─────────────────────────────────────────────────────────
#  4. 贪心策略
# ─────────────────────────────────────────────────────────

class GreedyPolicy(BaselinePolicy):
    """
    贪心：每步选择能最小化当前增量时延的 (task, location) 组合
    即：argmin_{(i,x) 合法} T_i^x（子任务单独执行时延，不考虑全局最优）
    """
    name = "Greedy"

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        N = obs["N"]
        best_action = -1
        best_time = float("inf")

        for i in range(N):
            task = env.dag.tasks[i]
            for x in range(3):
                a = i * 3 + x
                if not mask[a]:
                    continue
                t = env._compute_task_exec_time(task, x)
                if t < best_time:
                    best_time = t
                    best_action = a

        return best_action


# ─────────────────────────────────────────────────────────
#  5. HEFT（Heterogeneous Earliest Finish Time）
# ─────────────────────────────────────────────────────────

class HEFTPolicy(BaselinePolicy):
    """
    HEFT 启发式算法（经典基线）
    步骤：
      1. 为每个子任务计算 upward rank（考虑三个位置的平均执行时延）
      2. 按 rank 降序确定调度顺序
      3. 每个子任务选择 EFT（Earliest Finish Time）最小的位置
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

    def select_action(self, obs: Dict, env: ThreeTierEnv) -> int:
        mask = obs["valid_mask"]
        ec   = self.cfg.env

        # 找下一个合法任务（按 rank 顺序）
        task_idx = None
        for ti in self._priority_order[self._priority_ptr:]:
            if any(mask[ti * 3 + x] for x in range(3)):
                task_idx = ti
                self._priority_ptr = self._priority_order.index(ti) + 1
                break
        if task_idx is None:
            return int(np.where(mask)[0][0])

        # EFT 计算：用配置均值，不用 env.state 里的实时 Q_ec 和 rate_ue
        avg_rate_ue = float(np.mean(list(ec.rate_ue_options)))
        avg_r_ec    = env._eff_ec_rate(0.0)   # 假设无拥塞

        best_action, best_eft = task_idx * 3, float("inf")
        task = env.dag.tasks[task_idx]

        for x in range(3):
            a = task_idx * 3 + x
            if not mask[a]:
                continue

            if x == 0:   # local
                exec_t = task.cpu / ec.f_local
            elif x == 1: # edge
                exec_t = (task.data_in  / avg_rate_ue
                        + task.cpu      / ec.f_edge
                        + task.data_out / ec.rate_eu)
            else:        # cloud
                exec_t = (task.data_in  / avg_rate_ue
                        + task.data_in  / avg_r_ec
                        + task.cpu      / ec.f_cloud
                        + task.data_out / ec.rate_ce
                        + task.data_out / ec.rate_eu)

        pred_ready = 0.0
        if task.predecessors:
            pred_ready = max((env.state.finish_times[p] for p in task.predecessors))
        node_ready = env.state.node_avail[x]
        eft = max(pred_ready, node_ready) + exec_t
        if eft < best_eft:
            best_eft, best_action = eft, a

        return best_action

    def _compute_priority(self, dag: DAG, env: ThreeTierEnv) -> List[int]:
        """
        弱版 HEFT rank 计算：使用配置均值而非 episode 精确参数，
        与 DRLTO/MRLCO 的 HEFT baseline 实现保持一致。
        """
        ec = self.cfg.env
        N = dag.N
        rank = np.zeros(N)
        avg_exec = np.zeros(N)

        # 使用配置均值，不用 env.state 里的实时值
        avg_rate_ue = float(np.mean(list(ec.rate_ue_options)))   # 11.0 MB/s
        avg_q_ec = 0.0   # 假设无拥塞，与两层 HEFT 等价
        avg_r_ec = env._eff_ec_rate(avg_q_ec)                    # 最大回程速率

        for t in dag.tasks:
            t_local = t.cpu / ec.f_local
            t_edge  = (t.data_in  / avg_rate_ue
                     + t.cpu      / ec.f_edge
                     + t.data_out / ec.rate_eu)
            t_cloud = (t.data_in  / avg_rate_ue
                     + t.data_in  / avg_r_ec
                     + t.cpu      / ec.f_cloud
                     + t.data_out / ec.rate_ce
                     + t.data_out / ec.rate_eu)
            avg_exec[t.idx] = (t_local + t_edge + t_cloud) / 3.0

        topo = dag.topological_order()
        for v in reversed(topo):
            task = dag.tasks[v]
            if not task.successors:
                rank[v] = avg_exec[v]
            else:
                comm = task.data_out / avg_rate_ue
                rank[v] = avg_exec[v] + max(comm + rank[s]
                                            for s in task.successors)
        return list(np.argsort(-rank))


# ─────────────────────────────────────────────────────────
#  评估入口
# ─────────────────────────────────────────────────────────

def run_all_baselines(dags: List[DAG], cfg: Config, seed: int = 42) -> List[Dict]:
    """
    在给定 DAG 列表上运行所有基线，返回统计结果列表
    用于论文中的对比表格
    """
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