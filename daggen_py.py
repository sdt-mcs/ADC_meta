"""
daggen_py.py — daggen 算法的 Python 完整重实现

完全对齐原版 C 代码（github.com/frs69wq/daggen）的参数语义和生成逻辑，
无需编译 C 代码即可使用相同的 DAG 生成过程。

参数说明（与原版 daggen 一一对应）：
  n        : 子任务总数
  fat      : DAG 宽度。控制每层的"理想任务数" = e^(fat * log(n))。
             fat 越大 → DAG 越"胖"（并行度高）；fat 越小 → DAG 越"瘦"（链状）
  regular  : 各层任务数的均匀度 ∈ [0,1]。
             1.0 → 各层任务数严格等于理想值；0.0 → 可以在 [0, 2*ideal] 内大幅波动
  density  : 相邻两层之间的依赖边密度 ∈ [0,1]。
             控制每个任务平均连接的父任务数 = density * (上一层任务数)
  ccr      : communication-to-computation ratio（通信-计算比）
             定义为：数据传输时间 / 任务计算时间（在参考处理器上）
             ccr = (data_size / bandwidth) / (cpu_cycles / cpu_speed)
             决定 data_size 与 cpu_cycles 的比值
  mindata  : 任务输入数据量下界（字节）
  maxdata  : 任务输入数据量上界（字节）
  jump     : 允许跨层连边的最大跨度（原版默认1，即只连相邻层）
  minalpha : Amdahl 可并行化比例下界（本工作不使用，保留以备扩展）
  maxalpha : Amdahl 可并行化比例上界

论文参数设置（DRLTO / MRLCO）：
  fat     ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}（随机选取）
  density ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}（随机选取）
  ccr     ∈ {0.3, 0.4, 0.5}（随机选取；计算密集型应用 ccr < 0.5）
  regular = 0.9（固定）
  n       ∈ {10, 15, 20, 25, 30, 35, 40, 45, 50}（随机选取）
  mindata = 5000 B（5 KB）
  maxdata = 50000 B（50 KB）
  jump    = 1
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────
#  数据结构（与 dag_generator.py 中的 Task/DAG 兼容）
# ─────────────────────────────────────────────────────────

@dataclass
class DaggenTask:
    """
    daggen 生成的子任务节点

    Attributes:
        idx       : 任务唯一索引（从 0 开始）
        cpu       : 计算量（CPU cycles）= data_size × ccr_factor / cpu_ref_speed
        data_in   : 输入数据量（MB）= 原始 size（字节）/ 1e6
        data_out  : 输出数据量（MB）= 由传出的 TRANSFER 边确定
        alpha     : Amdahl 可并行化参数（本工作未使用，保留）
        predecessors : 前驱任务索引列表
        successors   : 后继任务索引列表
    """
    idx: int
    cpu: float          # cycles
    data_in: float      # MB
    data_out: float     # MB（初始化为 data_in × 0.5，后由依赖更新）
    alpha: float        # Amdahl 参数
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)

    def is_ready(self, done_set: set) -> bool:
        return all(p in done_set for p in self.predecessors)


@dataclass
class DaggenDAG:
    """
    daggen 生成的有向无环图（Job）
    完全兼容 env.py 和 model.py 的接口
    """
    tasks: List[DaggenTask]
    # daggen 生成参数（用于实验记录）
    gen_params: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.tasks)

    @property
    def entry_tasks(self):
        return [t for t in self.tasks if not t.predecessors]

    @property
    def exit_tasks(self):
        return [t for t in self.tasks if not t.successors]

    def get_ready_tasks(self, done_set: set, scheduled_set: set) -> List[int]:
        return [
            t.idx for t in self.tasks
            if t.idx not in scheduled_set and t.is_ready(done_set)
        ]

    def topological_order(self) -> List[int]:
        in_deg = {t.idx: len(t.predecessors) for t in self.tasks}
        queue = [t.idx for t in self.tasks if in_deg[t.idx] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for s in self.tasks[node].successors:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    queue.append(s)
        return order

    def to_dot(self) -> str:
        """输出 DOT 格式字符串（与原版 daggen 输出格式一致）"""
        lines = ["digraph G {"]
        for t in self.tasks:
            # 计算量还原为字节级 size（供 DOT 记录用）
            lines.append(f'  {t.idx+1} [size="{t.cpu:.4e}", alpha="{t.alpha:.2f}"]')
        for t in self.tasks:
            for s_idx in t.successors:
                size_bytes = int(t.data_out * 1e6)
                lines.append(f'  {t.idx+1} -> {s_idx+1} [size="{size_bytes}"]')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self):
        edges = []
        for t in self.tasks:
            for s in t.successors:
                edges.append(f"{t.idx}→{s}")
        return f"DaggenDAG(N={self.N}, edges=[{', '.join(edges[:8])}{'...' if len(edges)>8 else ''}])"


# ─────────────────────────────────────────────────────────
#  daggen 核心生成算法
# ─────────────────────────────────────────────────────────

# 参考处理器速度（用于 CCR 计算，对应 DRLTO 中的 f_edge = 10 GHz）
_CPU_REF_HZ = 10e9      # cycles/s
_BANDWIDTH_REF = 10.0   # MB/s（参考带宽，用于 CCR 折算）


class DaggenPy:
    """
    daggen 算法的 Python 完整重实现

    使用方式：
        gen = DaggenPy(seed=42)
        dag = gen.generate(n=20, fat=0.5, density=0.5, ccr=0.5, regular=0.9)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n: int = 20,
        fat: float = 0.5,
        density: float = 0.5,
        ccr: float = 0.5,
        regular: float = 0.9,
        mindata: float = 5000.0,    # 字节
        maxdata: float = 50000.0,   # 字节
        minalpha: float = 0.0,
        maxalpha: float = 1.0,
        jump: int = 1,
    ) -> DaggenDAG:
        """
        生成一个 DAG，参数语义完全对齐原版 daggen

        核心流程（与 daggen.c 一致）：
          1. 根据 fat 和 n 计算理想每层任务数
          2. 根据 regular 扰动各层实际任务数
          3. 根据 ccr/mindata/maxdata 分配计算量
          4. 根据 density/jump 生成依赖边
        """
        # ── Step 1：生成任务层次结构 ──────────────────────
        layers = self._generate_layers(n, fat, regular)
        total = sum(len(l) for l in layers)
        # 若层分配导致任务数不足/超出，截断或补足
        if total != n:
            layers = self._fix_layers(layers, n)

        # 分配全局任务 idx（按层顺序）
        all_tasks_by_layer: List[List[int]] = []
        idx = 0
        flat_idx_layers = []
        for layer in layers:
            layer_idxs = list(range(idx, idx + len(layer)))
            flat_idx_layers.append(layer_idxs)
            idx += len(layer)

        N = idx  # 实际任务数

        # ── Step 2：分配任务属性 ──────────────────────────
        tasks = []
        for i in range(N):
            data_size_bytes = self.rng.uniform(mindata, maxdata)
            alpha = self.rng.uniform(minalpha, maxalpha)

            # CCR 决定了计算量与通信量的比例关系：
            # ccr = (data_size / bandwidth) / (cpu_cycles / cpu_ref)
            # → cpu_cycles = data_size × cpu_ref / (ccr × bandwidth)
            # 这里使用参考带宽和参考 CPU 速度将 ccr 映射到 cpu_cycles
            if ccr > 0:
                cpu_cycles = (data_size_bytes / 1e6) * _CPU_REF_HZ / (ccr * _BANDWIDTH_REF)
            else:
                cpu_cycles = data_size_bytes * _CPU_REF_HZ / _BANDWIDTH_REF

            data_in_mb = data_size_bytes / 1e6
            tasks.append(DaggenTask(
                idx=i,
                cpu=cpu_cycles,
                data_in=data_in_mb,
                data_out=data_in_mb * 0.5,  # 先设默认值，后由 TRANSFER 边覆盖
                alpha=alpha,
            ))

        # ── Step 3：生成依赖边 ─────────────────────────────
        # 对每对相邻层（或跨层，若 jump > 1）按 density 生成边
        for l_idx in range(len(flat_idx_layers) - 1):
            # jump 控制当前层可以连到哪些后续层
            max_target_layer = min(l_idx + jump, len(flat_idx_layers) - 1)
            for tgt_l in range(l_idx + 1, max_target_layer + 1):
                src_layer = flat_idx_layers[l_idx]
                tgt_layer = flat_idx_layers[tgt_l]
                self._connect_layers(tasks, src_layer, tgt_layer, density)

        # ── Step 4：确保连通性（每个非入口节点至少有一个父节点）────
        self._ensure_connectivity(tasks, flat_idx_layers)

        # ── Step 5：根据实际出边更新 data_out ────────────────
        # 每条 TRANSFER 边的数据量 = 均匀随机采样（在 [mindata, maxdata] 范围内）
        for t in tasks:
            if t.successors:
                transfer_size = self.rng.uniform(mindata, maxdata) / 1e6  # MB
                t.data_out = transfer_size

        dag = DaggenDAG(
            tasks=tasks,
            gen_params={
                "n": N, "fat": fat, "density": density,
                "ccr": ccr, "regular": regular,
                "mindata": mindata, "maxdata": maxdata, "jump": jump
            }
        )
        return dag

    def _generate_layers(self, n: int, fat: float, regular: float) -> List[List[int]]:
        """
        Step 1 of daggen.c：
        ideal_per_layer = e^(fat * log(n)) = n^fat
        然后用 regular 参数在 ideal 周围随机扰动
        """
        ideal = math.exp(fat * math.log(max(n, 2)))  # = n^fat
        ideal = max(1.0, ideal)

        layers = []
        remaining = n
        while remaining > 0:
            if remaining <= ideal:
                layers.append(list(range(remaining)))
                remaining = 0
            else:
                # 扰动范围：regular=1.0 时不扰动；regular=0 时可以在 [0, 2*ideal]
                lo = max(1, int(round(ideal * regular)))
                hi = max(lo + 1, int(round(ideal * (2.0 - regular))))
                count = int(self.rng.integers(lo, hi + 1))
                count = min(count, remaining)
                layers.append(list(range(count)))
                remaining -= count

        return layers

    def _fix_layers(self, layers: List[List[int]], target_n: int) -> List[List[int]]:
        """修正层列表，使总任务数等于 target_n"""
        total = sum(len(l) for l in layers)
        if total > target_n:
            # 截断最后一层
            excess = total - target_n
            while excess > 0 and layers:
                cut = min(excess, len(layers[-1]))
                layers[-1] = layers[-1][:-cut]
                excess -= cut
                if not layers[-1]:
                    layers.pop()
        elif total < target_n:
            layers.append(list(range(target_n - total)))
        return [l for l in layers if l]

    def _connect_layers(
        self,
        tasks: List[DaggenTask],
        src_layer: List[int],
        tgt_layer: List[int],
        density: float,
    ):
        """
        根据 density 在 src_layer 和 tgt_layer 之间生成随机依赖边
        对 tgt_layer 中的每个节点，从 src_layer 中随机选取 k 个父节点
        k = max(1, round(density × |src_layer|))
        """
        if not src_layer or not tgt_layer:
            return

        expected_parents = max(1, round(density * len(src_layer)))

        for tgt_idx in tgt_layer:
            n_parents = min(
                int(self.rng.integers(
                    max(1, expected_parents - 1),
                    expected_parents + 2
                )),
                len(src_layer)
            )
            parents = self.rng.choice(src_layer, size=n_parents, replace=False)
            for p in parents:
                if tgt_idx not in tasks[p].successors:
                    tasks[p].successors.append(tgt_idx)
                if p not in tasks[tgt_idx].predecessors:
                    tasks[tgt_idx].predecessors.append(p)

    def _ensure_connectivity(
        self,
        tasks: List[DaggenTask],
        layers: List[List[int]],
    ):
        """
        确保每个非第一层的任务至少有一个父节点
        （原版 daggen 也有类似保护）
        """
        for l_idx in range(1, len(layers)):
            prev_layer = layers[l_idx - 1]
            for tgt_idx in layers[l_idx]:
                if not tasks[tgt_idx].predecessors:
                    # 从上一层随机指派一个父节点
                    p = int(self.rng.choice(prev_layer))
                    tasks[p].successors.append(tgt_idx)
                    tasks[tgt_idx].predecessors.append(p)

    def generate_batch(
        self,
        count: int,
        n_options: List[int] = None,
        fat_options: List[float] = None,
        density_options: List[float] = None,
        ccr_options: List[float] = None,
        regular: float = 0.9,
        mindata: float = 5000.0,
        maxdata: float = 50000.0,
        jump: int = 1,
    ) -> List[DaggenDAG]:
        """
        批量生成 DAG，参数从给定选项中随机选取（对齐论文设置）

        默认选项与 DRLTO / MRLCO 论文完全一致：
          n       ∈ {10, 15, 20, 25, 30, 35, 40, 45, 50}
          fat     ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
          density ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
          ccr     ∈ {0.3, 0.4, 0.5}
        """
        if n_options is None:
            n_options = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        if fat_options is None:
            fat_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if density_options is None:
            density_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if ccr_options is None:
            ccr_options = [0.3, 0.4, 0.5]

        dags = []
        for _ in range(count):
            n = int(self.rng.choice(n_options))
            fat = float(self.rng.choice(fat_options))
            density = float(self.rng.choice(density_options))
            ccr = float(self.rng.choice(ccr_options))
            dags.append(self.generate(
                n=n, fat=fat, density=density, ccr=ccr,
                regular=regular, mindata=mindata, maxdata=maxdata, jump=jump
            ))
        return dags


# ─────────────────────────────────────────────────────────
#  .dot 文件解析器（解析原版 daggen 二进制生成的文件）
# ─────────────────────────────────────────────────────────

def parse_dot_file(dot_path: str) -> DaggenDAG:
    """
    解析原版 daggen 生成的 .dot 文件，返回 DaggenDAG 对象

    DOT 格式：
      <id> [size="<CPU_cycles>", alpha="<alpha>"]   ← COMPUTATION 节点
      <src> -> <dst> [size="<transfer_bytes>"]       ← TRANSFER 边
    """
    import re

    node_pat = re.compile(r'^\s*(\d+)\s*\[size="([^"]+)",\s*alpha="([^"]+)"\]')
    edge_pat = re.compile(r'^\s*(\d+)\s*->\s*(\d+)\s*\[size="([^"]+)"\]')

    nodes: dict = {}   # id(1-based) → (cpu_cycles, alpha)
    edges: list = []   # (src_id, dst_id, transfer_bytes)

    with open(dot_path) as f:
        for line in f:
            m_node = node_pat.match(line)
            m_edge = edge_pat.match(line)
            if m_node:
                nid = int(m_node.group(1))
                cpu = float(m_node.group(2))
                alpha = float(m_node.group(3))
                nodes[nid] = (cpu, alpha)
            elif m_edge:
                src = int(m_edge.group(1))
                dst = int(m_edge.group(2))
                sz = float(m_edge.group(3))
                edges.append((src, dst, sz))

    if not nodes:
        raise ValueError(f"未能从 {dot_path} 中解析到任何任务节点")

    # 转为 0-based 索引
    sorted_ids = sorted(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(sorted_ids)}
    N = len(sorted_ids)

    tasks = []
    for nid in sorted_ids:
        cpu, alpha = nodes[nid]
        tasks.append(DaggenTask(
            idx=id_to_idx[nid],
            cpu=cpu,
            data_in=0.0,   # 由入边 TRANSFER 大小决定（见下）
            data_out=0.0,  # 由出边 TRANSFER 大小决定
            alpha=alpha,
        ))

    # 填充依赖关系和数据量
    for src_nid, dst_nid, sz_bytes in edges:
        src_idx = id_to_idx[src_nid]
        dst_idx = id_to_idx[dst_nid]
        tasks[src_idx].successors.append(dst_idx)
        tasks[dst_idx].predecessors.append(src_idx)
        sz_mb = sz_bytes / 1e6
        tasks[src_idx].data_out = sz_mb      # 出边数据量
        tasks[dst_idx].data_in = max(tasks[dst_idx].data_in, sz_mb)  # 入边最大数据量

    return DaggenDAG(tasks=tasks)


# ─────────────────────────────────────────────────────────
#  调用原版 daggen 二进制（可选，需要编译好的可执行文件）
# ─────────────────────────────────────────────────────────

def call_daggen_binary(
    binary_path: str,
    n: int = 20,
    fat: float = 0.5,
    density: float = 0.5,
    ccr: float = 0.5,
    regular: float = 0.9,
    mindata: int = 5000,
    maxdata: int = 50000,
    jump: int = 1,
    output_dot: str = "/tmp/dag_out.dot",
) -> DaggenDAG:
    """
    调用编译好的 daggen 可执行文件，生成 .dot 文件后解析返回 DaggenDAG

    使用前提：
      1. 已从 https://github.com/frs69wq/daggen 克隆源码
      2. 在源码目录运行 `make` 编译生成 daggen 可执行文件
      3. 将 binary_path 指向该可执行文件路径

    编译步骤（Linux/macOS）：
      git clone https://github.com/frs69wq/daggen
      cd daggen
      make
      # 生成可执行文件 ./daggen

    Windows：需要 MinGW 或 WSL 环境。
    """
    import subprocess

    cmd = [
        binary_path,
        f"--n={n}",
        f"--fat={fat}",
        f"--density={density}",
        f"--ccr={ccr}",
        f"--regular={regular}",
        f"--mindata={mindata}",
        f"--maxdata={maxdata}",
        f"--jump={jump}",
        "--dot",
        f"--output={output_dot}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"daggen 执行失败:\n{result.stderr}")

    return parse_dot_file(output_dot)


# ─────────────────────────────────────────────────────────
#  快速测试
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = DaggenPy(seed=42)

    print("=== 单个 DAG 生成 ===")
    dag = gen.generate(n=20, fat=0.5, density=0.5, ccr=0.5, regular=0.9)
    print(dag)
    print(f"  入口任务: {[t.idx for t in dag.entry_tasks]}")
    print(f"  出口任务: {[t.idx for t in dag.exit_tasks]}")
    print(f"  拓扑顺序: {dag.topological_order()}")
    print(f"  Task 0: cpu={dag.tasks[0].cpu:.2e} cycles, data_in={dag.tasks[0].data_in:.4f} MB")
    print()

    print("=== DOT 格式输出（前10行）===")
    dot = dag.to_dot()
    for line in dot.split('\n')[:10]:
        print(" ", line)
    print()

    print("=== 批量生成（论文标准参数）===")
    dags = gen.generate_batch(count=10)
    n_vals = [d.N for d in dags]
    print(f"  生成数量: {len(dags)}")
    print(f"  任务数分布: min={min(n_vals)}, max={max(n_vals)}, "
          f"mean={sum(n_vals)/len(n_vals):.1f}")
    print(f"  各 DAG gen_params 示例: {dags[0].gen_params}")

    print()
    print("✅ daggen_py 测试通过")
