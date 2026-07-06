"""
plot_results.py — 结果可视化脚本

读取 results/ 目录下的 JSON 文件，生成论文所需图表：
  Figure 1: 训练曲线（makespan + eps_in 变化）
  Figure 2: 消融实验柱状图
  Figure 3: 基线对比柱状图
  Figure 4: 快速适应曲线（adapt_steps vs makespan）
  Figure 5: CCSM 软掩码动态响应（Q_ec vs w_cloud）

用法：python plot_results.py [--fig all|train|ablation|compare|adapt|ccsm]
所有图表保存到 results/figures/ 目录。
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")             # 非交互后端，适合服务器/PyCharm 运行
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 全局绘图风格 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "main":    "#2563EB",   # 本工作-蓝
    "ablA":    "#DC2626",   # 消融A-红
    "ablB1":   "#D97706",   # 消融B1-橙
    "ablB2":   "#7C3AED",   # 消融B2-紫
    "ablC":    "#059669",   # 消融C-绿
    "heft":    "#6B7280",   # HEFT-灰
    "greedy":  "#F59E0B",   # Greedy-黄
    "local":   "#9CA3AF",
    "edge":    "#34D399",
    "random":  "#FCA5A5",
    "eps_in":  "#F97316",   # eps_in 曲线
}

os.makedirs("results/figures", exist_ok=True)


# ─────────────────────────────────────────────────────────
#  Figure 1：训练曲线
# ─────────────────────────────────────────────────────────

def plot_training_curve(log_path: str = "logs/training_log.json"):
    if not os.path.exists(log_path):
        print(f"  [跳过] 未找到训练日志: {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    iters = log["outer_iter"]
    makespans = log["mean_makespan"]
    eps_ins = log["eps_in"]
    g_emas = log["g_ema"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.suptitle("ADC Meta-RL Training Curves", fontweight="bold", y=1.02)

    # 平滑（滑动窗口）
    def smooth(x, w=10):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # (a) Makespan
    ax = axes[0]
    raw = np.array(makespans)
    sm = smooth(raw)
    ax.plot(iters, raw, alpha=0.25, color=COLORS["main"], linewidth=0.8)
    ax.plot(iters[:len(sm)], sm, color=COLORS["main"], linewidth=2, label="Makespan (smoothed)")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_title("(a) Makespan vs. Iteration")
    ax.legend()

    # (b) ε_in 变化
    ax = axes[1]
    ax.plot(iters, eps_ins, color=COLORS["eps_in"], linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=1, label=r"$\varepsilon_{out}=0.1$")
    ax.axhline(y=0.2, color="navy", linestyle="--", linewidth=1, label=r"$\varepsilon_{base}=0.2$")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"$\varepsilon_{in}$")
    ax.set_title(r"(b) Adaptive $\varepsilon_{in}$ Dynamics")
    ax.set_ylim([0, 0.45])
    ax.legend()

    # (c) EMA 梯度范数
    ax = axes[2]
    ax.plot(iters, g_emas, color="#10B981", linewidth=1.5)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"$\bar{g}$ (EMA Grad Norm)")
    ax.set_title(r"(c) Gradient Norm EMA $\bar{g}$")

    plt.tight_layout()
    out = "results/figures/fig1_training_curve.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 1 已保存 → {out}")


# ─────────────────────────────────────────────────────────
#  Figure 2：消融实验
# ─────────────────────────────────────────────────────────

def plot_ablation(ablation_path: str = "results/ablation.json"):
    if not os.path.exists(ablation_path):
        print(f"  [跳过] 未找到消融结果: {ablation_path}")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    labels = {
        "Full_ADC_MetaRL": "Full Model\n(Ours)",
        "w/o_CCSM":        "w/o CCSM",
        "Fixed_eps_in":    r"Fixed $\varepsilon_{in}$",
        "Single_eps":      r"Single $\varepsilon$",
        "No_Adaptation":   "No Adaptation",
    }
    colors_list = [COLORS["main"], COLORS["ablA"], COLORS["ablB1"],
                   COLORS["ablB2"], COLORS["ablC"]]

    keys = list(labels.keys())
    means = [data[k]["mean"] for k in keys]
    stds = [data[k]["std"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors_list, alpha=0.85, edgecolor="white", linewidth=1.2)

    # 标注具体数值
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.2f}s", ha="center", va="bottom", fontsize=9)

    # 标出完整模型基准线
    ax.axhline(y=means[0], color=COLORS["main"], linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Full Model ({means[0]:.2f}s)")

    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], fontsize=10)
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_title("Ablation Study: Component Contribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    out = "results/figures/fig2_ablation.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 2 已保存 → {out}")


# ─────────────────────────────────────────────────────────
#  Figure 3：基线对比
# ─────────────────────────────────────────────────────────

def plot_comparison(compare_path: str = "results/comparison.json"):
    if not os.path.exists(compare_path):
        print(f"  [跳过] 未找到对比结果: {compare_path}")
        return

    with open(compare_path) as f:
        data = json.load(f)

    name_map = {
        "LocalOnly":   "Local\nOnly",
        "EdgeOnly":    "Edge\nOnly",
        "Random":      "Random",
        "Greedy":      "Greedy",
        "HEFT":        "HEFT",
        "ADC_MetaRL":  "ADC Meta-RL\n(Ours)",
    }
    color_map = {
        "LocalOnly": COLORS["local"], "EdgeOnly": COLORS["edge"],
        "Random": COLORS["random"],   "Greedy": COLORS["greedy"],
        "HEFT": COLORS["heft"],       "ADC_MetaRL": COLORS["main"],
    }

    keys = [k for k in name_map if k in data]
    means = [data[k]["mean"] for k in keys]
    stds  = [data[k]["std"]  for k in keys]
    clrs  = [color_map[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=clrs, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([name_map[k] for k in keys], fontsize=10)
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_title("Performance Comparison with Baselines", fontweight="bold")

    # 标出本工作
    ours_idx = keys.index("ADC_MetaRL")
    bars[ours_idx].set_edgecolor("#1D4ED8")
    bars[ours_idx].set_linewidth(2.5)

    plt.tight_layout()
    out = "results/figures/fig3_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 3 已保存 → {out}")


# ─────────────────────────────────────────────────────────
#  Figure 4：快速适应曲线
# ─────────────────────────────────────────────────────────

def plot_adaptation_curve(adapt_path: str = "results/adaptation_curve.json"):
    if not os.path.exists(adapt_path):
        print(f"  [跳过] 未找到适应曲线数据: {adapt_path}")
        return

    with open(adapt_path) as f:
        data = json.load(f)

    steps = data["adapt_steps"]
    meta_ms = data["ADC_MetaRL"]
    rand_ms = data["RandomInit"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, meta_ms, "o-", color=COLORS["main"], linewidth=2,
            markersize=6, label="ADC Meta-RL (Ours)")
    ax.plot(steps, rand_ms, "s--", color=COLORS["heft"], linewidth=1.8,
            markersize=6, label="Random Init (Fine-tune)")

    ax.set_xlabel("Number of Gradient Steps (K)")
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_title("Fast Adaptation: Meta-RL vs. Fine-tuning from Scratch",
                 fontweight="bold")
    ax.set_xticks(steps)
    ax.legend()
    plt.tight_layout()

    out = "results/figures/fig4_adaptation_curve.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 4 已保存 → {out}")


# ─────────────────────────────────────────────────────────
#  Figure 5：CCSM 软掩码动态响应
# ─────────────────────────────────────────────────────────

def plot_ccsm_response():
    """展示软掩码权重随 Q_ec 变化的曲线（无需数据文件）"""
    Q_max = 50.0
    q_range = np.linspace(0, Q_max, 200)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # (a) 不同 alpha 的软掩码曲线
    ax = axes[0]
    for alpha, lw, ls in [(1.0, 1.5, ":"), (3.0, 2.0, "-"), (5.0, 1.5, "--")]:
        w = np.exp(-alpha * q_range / Q_max)
        ax.plot(q_range, w, linewidth=lw, linestyle=ls,
                label=fr"$\alpha={alpha}$")
    ax.set_xlabel(r"$Q_{ec}$ (MB)")
    ax.set_ylabel(r"Soft Mask Weight $w^{cloud}$")
    ax.set_title("(a) CCSM Weight vs. Queue Length", fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.axvline(x=Q_max * 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(Q_max * 0.52, 0.5, r"$Q_{max}/2$", color="gray", fontsize=9)
    ax.legend()

    # (b) 软掩码 vs 硬掩码的概率分布对比示意
    ax = axes[1]
    actions = ["Local", "Edge", "Cloud"]
    raw_logits = np.array([0.5, 0.8, 0.7])   # 示例原始 logit

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    # 无拥塞（Q_ec=0）
    w_low = np.exp(-3.0 * 0.0 / Q_max)
    logits_low = raw_logits.copy()
    logits_low[2] += np.log(w_low + 1e-9)
    p_low = softmax(logits_low)

    # 中等拥塞（Q_ec=Q_max/2）
    w_mid = np.exp(-3.0 * 0.5)
    logits_mid = raw_logits.copy()
    logits_mid[2] += np.log(w_mid + 1e-9)
    p_mid = softmax(logits_mid)

    # 高度拥塞（Q_ec=Q_max）
    w_high = np.exp(-3.0 * 1.0)
    logits_high = raw_logits.copy()
    logits_high[2] += np.log(w_high + 1e-9)
    p_high = softmax(logits_high)

    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, p_low,  width, label=r"$Q_{ec}=0$",           color="#93C5FD")
    ax.bar(x,         p_mid,  width, label=r"$Q_{ec}=Q_{max}/2$",   color="#3B82F6")
    ax.bar(x + width, p_high, width, label=r"$Q_{ec}=Q_{max}$",     color="#1E3A8A")
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel("Action Probability")
    ax.set_title("(b) CCSM Effect on Action Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = "results/figures/fig5_ccsm.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 5 已保存 → {out}")


# ─────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADC Meta-RL 结果可视化")
    parser.add_argument("--fig", default="all",
                        choices=["all", "train", "ablation", "compare", "adapt", "ccsm"],
                        help="要绘制的图表")
    args = parser.parse_args()

    print("绘制图表中...")

    if args.fig in ("all", "train"):
        plot_training_curve()
    if args.fig in ("all", "ablation"):
        plot_ablation()
    if args.fig in ("all", "compare"):
        plot_comparison()
    if args.fig in ("all", "adapt"):
        plot_adaptation_curve()
    if args.fig in ("all", "ccsm"):
        plot_ccsm_response()

    print("\n✅ 所有图表保存在 results/figures/ 目录")
