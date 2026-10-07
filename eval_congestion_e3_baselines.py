"""
eval_congestion_e3_baselines.py
================================
【新增实验 —— 蓝图第八节 Item 1】E=3 异构多边缘下的经典基线拥塞敏感性对比
（HEFT / Greedy / EdgeOnly / LocalOnly），用于新增的拥塞双面板图 panel (b)。

背景（务必先读，避免重复踩坑）：
  - `claude/结论状态_当前.md`「还差什么」高优先级第2条明确指出：
    "多边缘缺基线对比（无 HEFT / Greedy / EdgeOnly）。baseline.py 异构版已落地可用。"
    本脚本就是补这个缺口——**只跑经典基线，不涉及 RL 模型/checkpoint，不需要训练**，
    属于纯推理，成本极低。
  - 同一份文档同时指出："拥塞曲线只有 3 个有效点（default 与 low 重合），画不出趋势图，
    需加密到 6~8 档"——因此本脚本默认用 6 档而不是沿用 eval_congestion.py 的 3 档
    （Low/Medium/High），以便和 E=1 侧的曲线在同一张图里都有足够的点数。
  - `config.py` 的 `Config()` 默认 `env.num_edges = 3`，即默认就是 E=3 异构多边缘配置，
    不需要额外传参改 E。
  - `baseline.py` 的 HEFT/Greedy/EdgeOnly/LocalOnly 均已是多边缘通用实现
    （用 env.E / env.L 泛化，不再硬编码 3 个位置），因此可以直接调用，
    不需要为 E=3 另写一套基线代码。

运行（在你的服务器上，仓库根目录）：
    python eval_congestion_e3_baselines.py

输出：
    results/congestion_e3_baselines.json   —— 原始数值（含每档 4 个基线的 mean/std/min/max）
    results/figures/fig_congestion_e3_baselines_panelb.pdf —— 可直接作为拥塞双面板图 panel (b) 草图

预计耗时：纯 CPU 推理，6 档 × 4 个基线 × 30 个 DAG，几分钟内可跑完，无需 GPU、
不占用 cuda:0 / cuda:1，不会影响你正在跑的任何训练任务。

【与 PID 19229 的隔离说明】本脚本不使用 CUDA（baseline 全部是 numpy 规则调度器），
不会触碰任何 GPU 进程，可以和其他任务同时跑。
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from daggen_py import DaggenPy
from env import ThreeTierEnv
from baseline import run_all_baselines

os.makedirs("results/figures", exist_ok=True)

# ── 全局 Q_ec 补丁：复用 eval_congestion.py 里验证过的写法 ───────────
# run_all_baselines() 内部会调用 env.reset(dag)，不带 q_ec_range 参数；
# 通过给 ThreeTierEnv.reset 打补丁，让它在没有显式传参时使用我们指定的范围。
_Q_EC_OVERRIDE = None
_orig_reset = ThreeTierEnv.reset


def _patched_reset(self, dag, q_ec_range=None):
    effective = q_ec_range if q_ec_range is not None else _Q_EC_OVERRIDE
    return _orig_reset(self, dag, q_ec_range=effective)


ThreeTierEnv.reset = _patched_reset

# ── 6 档拥塞水平（Q_ec_max = 50 MB，见 config.py EnvConfig.Q_ec_max）───
# 比 eval_congestion.py 的 3 档（Low/Medium/High）更密，用于画出平滑趋势线；
# 与 E=1 侧曲线的档位边界对齐（0/15/35/50 是原 3 档分界），额外插入 2 个分界点。
LEVELS = [
    ("L1 [0-8]",   0.0,  8.0),
    ("L2 [8-16]",  8.0,  16.0),
    ("L3 [16-24]", 16.0, 24.0),
    ("L4 [24-32]", 24.0, 32.0),
    ("L5 [32-40]", 32.0, 40.0),
    ("L6 [40-50]", 40.0, 50.0),
]
N_TEST = 30          # 与 eval_congestion.py 一致，纯基线推理成本低，够用
SEED_DAG = 999       # 与「结论状态_当前.md」的报告集口径一致（seed=999）
SEED_ENV = 0

BASELINES_WANTED = ["LocalOnly", "EdgeOnly", "Greedy", "HEFT"]  # 不含 Random


def run_e3_baseline_congestion():
    global _Q_EC_OVERRIDE
    cfg = Config()
    assert cfg.env.num_edges == 3, (
        f"期望 E=3（默认配置），实际 num_edges={cfg.env.num_edges}；"
        "请确认没有被其他地方 monkeypatch 过 config.py 默认值。"
    )

    gen = DaggenPy(seed=SEED_DAG)
    test_dags = gen.generate_batch(
        N_TEST,
        n_options=cfg.dag.n_options,
        fat_options=cfg.dag.fat_options,
        density_options=cfg.dag.density_options,
        ccr_options=cfg.dag.ccr_options,
        regular=cfg.dag.regular,
        mindata=cfg.dag.mindata,
        maxdata=cfg.dag.maxdata,
        jump=cfg.dag.jump,
    )
    print(f"E={cfg.env.num_edges} 异构多边缘 | 测试集：{N_TEST} 个 DAG（seed={SEED_DAG}）\n")

    results = {"labels": [], "num_edges": cfg.env.num_edges}
    for name in BASELINES_WANTED:
        results[name] = {"mean": [], "std": []}

    for label, lo, hi in LEVELS:
        _Q_EC_OVERRIDE = (lo, hi)
        print(f"── {label}  Q_ec∈[{lo:.0f},{hi:.0f}] MB ──")
        # 注：run_all_baselines() 内部会自己打印每个策略的 mean/std（含 Random），
        # 这里不再重复打印，只挑出我们要的 4 个存进 results。
        bl = run_all_baselines(test_dags, cfg, seed=SEED_ENV)
        _Q_EC_OVERRIDE = None

        results["labels"].append(label)
        for r in bl:
            if r["name"] in BASELINES_WANTED:
                results[r["name"]]["mean"].append(r["mean"])
                results[r["name"]]["std"].append(r["std"])
        print()

    out = "results/congestion_e3_baselines.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存 → {out}")
    return results


def plot_panel_b(results):
    labels = results["labels"]
    x = np.arange(len(labels))
    colors = {"LocalOnly": "#999999", "EdgeOnly": "#4d4d4d",
              "Greedy": "#d6604d", "HEFT": "#2166ac"}

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for name in BASELINES_WANTED:
        vals = results[name]["mean"]
        ax.plot(x, vals, marker="o", markersize=3, linewidth=1.2,
                label=name, color=colors.get(name))

    ax.set_xticks(x)
    ax.set_xticklabels([lb.split(" ")[0] for lb in labels], fontsize=7)
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_xlabel(r"Backhaul congestion level ($Q_{ec}$ initial range, E=3)")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.4, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/fig_congestion_e3_baselines_panelb.pdf")
    plt.close()
    print("[done] fig_congestion_e3_baselines_panelb.pdf")


def sanity_check(results):
    """
    跑完自动做的健全性检查（不是可选项，务必看这段输出）：
      1. LocalOnly 全程只在本地执行，从不触碰边缘/云/回程，因此它的 mean makespan
         理论上应该完全不随 Q_ec 档位变化（每档用同一批 DAG、同一个 env seed）。
      2. EdgeOnly 全程只在边缘执行，同样从不使用回程链路，mean makespan 也应该
         基本不随 Q_ec 变化。
      3. HEFT / Greedy 会视情况把部分任务路由到云端，因此**理论上**拥塞越高、
         走云端的代价越大，可能出现两种合理情况：(a) mean makespan 随拥塞档位
         上升而上升（如果它们仍然选择走云）；(b) mean makespan 基本不变（如果
         它们本来就很少选云端路径，异构多边缘下边缘算力本身可能已经比 E=1 时
         更有吸引力）。这两种都合理，只有"随拥塞上升而下降"这种反直觉情况
         需要警惕。
    如果 1、2 两条不成立（数值肉眼可见地随拥塞档位变化，而不是浮点误差级别的
    抖动），大概率是 `_Q_EC_OVERRIDE` 补丁没生效或 env.reset 签名不匹配，
    需要停下来排查，不要继续跑后面的正式实验。
    """
    print("=" * 60)
    print("健全性检查（务必核对，不是可跳过的例行日志）")
    print("=" * 60)
    for name in ("LocalOnly", "EdgeOnly"):
        vals = results[name]["mean"]
        spread = max(vals) - min(vals)
        flag = "✅ 通过" if spread < 1e-6 else "❌ 异常"
        print(f"  {name:10s} 跨 6 档拥塞的 mean makespan 极差 = {spread:.6f}s  {flag}"
              f"（期望 ≈ 0，因为该策略从不使用回程链路）")
    for name in ("HEFT", "Greedy"):
        vals = results[name]["mean"]
        print(f"  {name:10s} 各档 mean makespan：" +
              "  ".join(f"{v:.3f}" for v in vals))
    print("=" * 60)


if __name__ == "__main__":
    results = run_e3_baseline_congestion()
    sanity_check(results)
    plot_panel_b(results)
    print("\n完成。")
