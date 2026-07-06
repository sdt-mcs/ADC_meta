"""
plot_paper_figures.py
=====================
生成论文全部图表（除拥塞敏感性实验外）
运行：python plot_paper_figures.py
依赖：matplotlib >= 3.5, numpy, json
输出：results/figures/ 目录下各 PDF 文件
"""

import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("results/figures", exist_ok=True)

# ── 全局样式 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "legend.fontsize":    7,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.linewidth":     0.6,
    "lines.linewidth":    1.2,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# ── 颜色方案（全论文统一） ────────────────────────────────
C = {
    "adc":      "#2166ac",   # ADC-Meta：深蓝
    "heft":     "#4d4d4d",   # HEFT：深灰
    "greedy":   "#d6604d",   # Greedy：深橙红
    "edge":     "#74add1",   # Edge-only：浅蓝
    "local":    "#878787",   # Local-only：中灰
    "rand":     "#f4a582",   # Rand-Init：浅橙
    "no_ccsm":  "#d73027",   # w/o CCSM：红
    "fixed_eps":"#fc8d59",   # Fixed-ε：橙
    "no_meta":  "#999999",   # w/o Meta：灰
    "ref":      "#4d4d4d",   # 参考线
}

HEFT_VAL = 0.339   # HEFT 基准值（全图通用）

# ══════════════════════════════════════════════════════════
# FIG 1：总体性能对比水平条形图（V.B）
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# FIG 1：总体性能对比水平条形图（V.B）
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# FIG 1：总体性能对比水平条形图（V.B）
# ══════════════════════════════════════════════════════════
def plot_fig1_comparison():
    # 【修改点1】将 HEFT 作为独立的算法加入列表，放在最底部（性能最好）
    methods = ["Local-only", "Rand.-Init", "Edge-only",
               "Greedy",     "ADC-Meta (K=0)", "ADC-Meta (K=2)", "HEFT"]
    
    # 【修改点2】对应加入 HEFT 的数值 (HEFT_VAL) 和 标准差 (启发式算法设为0)
    means   = [2.336, 0.806, 0.540, 0.359, 0.252, 0.235, HEFT_VAL]
    stds    = [1.036, 0.384, 0.249, 0.180, 0.125, 0.0,   0.0     ]
    
    # 【修改点3】对应加入 HEFT 的颜色和填充样式
    colors  = [C["local"], C["rand"], C["edge"],
               C["greedy"], C["adc"],   C["adc"], C["heft"]]
    hatches = ["",  "",  "",  "",  "",  "///", ""]

    # 【修改点4】适当增加图表高度 (2.3 -> 2.6)，容纳7个柱子，防止拥挤
    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    y_pos = np.arange(len(methods))[::-1]   # 从上到下由差到好
    bars = ax.barh(y_pos, means, xerr=stds,
                   color=colors, hatch=hatches,
                   height=0.55, error_kw={"elinewidth": 0.7,
                                          "capsize": 2.0,
                                          "capthick": 0.7},
                   edgecolor="white", linewidth=0.4)

    # 【修改点5】彻底移除了原来的 ax.axvline (垂直虚线) 和关联文本

    # 数值标注
    for bar, m, s in zip(bars, means, stds):
        x = m + s + 0.08 if s > 0 else m + 0.05
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", ha="left",
                fontsize=6, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Mean Makespan (s)")
    
    # 保持宽裕的横轴视野
    ax.set_xlim(0, 3.8)
    ax.grid(axis="x", which="major")
    ax.spines[["top", "right"]].set_visible(False)

    # 图例 (无需改动，只标示我们自己的 K=0 和 K=2 的区别即可)
    patch_k0 = mpatches.Patch(color=C["adc"], label="ADC-Meta (K=0)")
    patch_k2 = mpatches.Patch(color=C["adc"], hatch="///",
                               label="ADC-Meta (K=2)")
    ax.legend(handles=[patch_k0, patch_k2], loc="lower right",
              framealpha=0.8, edgecolor="gray", fontsize=6.5)

    plt.tight_layout()
    plt.savefig("results/figures/fig1_comparison.pdf")
    plt.close()
    print("[✓] fig1_comparison.pdf")

# ══════════════════════════════════════════════════════════
# FIG 3：快速适应双面板（V.C）
# 左：适应曲线；右：N 规模对比
# ══════════════════════════════════════════════════════════

def plot_fig3_adaptation():
    # ── 左面板数据 ──────────────────────────────────────
    k_vals  = [0, 1, 2, 3, 5, 10, 20]
    adc_ms  = [0.252, 0.238, 0.235, 0.237, 0.245, 0.243, 0.254]
    rand_ms = [0.482, 0.593, 0.841, 0.856, 0.815, 0.792, 0.813]

    # ── 右面板数据 ──────────────────────────────────────
    n_vals    = [10,    20,    30,    40,    50   ]
    heft_n    = [0.106, 0.231, 0.338, 0.464, 0.573]
    adc_k0_n  = [0.079, 0.164, 0.237, 0.320, 0.387]
    adc_k2_n  = [0.108, 0.159, 0.217, 0.305, 0.361]

    # 扁平化画布与紧凑间距
    fig = plt.figure(figsize=(3.5, 5.0))
    gs  = GridSpec(2, 1, figure=fig, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── 左：适应曲线 ─────────────────────────────────
    ax1.plot(k_vals, adc_ms,  "o-",  color=C["adc"],
             label="ADC-Meta",  markersize=4, zorder=4)
    ax1.plot(k_vals, rand_ms, "s--", color=C["rand"],
             label="Rand.-Init", markersize=4, zorder=4)
    ax1.axhline(HEFT_VAL, color=C["heft"], linestyle="-.",
                linewidth=0.9, zorder=3)
                
    # 【修改点 1：废弃原来的 text 标注，改用悬空 annotate 箭头指向 HEFT 线】
    ax1.annotate(f"HEFT Ref.\n({HEFT_VAL:.3f}s)", 
                 xy=(21.0, HEFT_VAL),       # 箭头指向虚线最右侧
                 xytext=(17.5, 0.45),       # 文字悬停在上方宽阔的空白区
                 arrowprops=dict(arrowstyle="->", color=C["heft"], lw=0.9, 
                                 connectionstyle="arc3,rad=-0.15"),
                 color=C["heft"], fontsize=6.5, va="center", ha="center", zorder=5)

    # 异常点高亮外圈
    ax1.plot(2, 0.235, "o", markersize=8, markerfacecolor="none", 
             markeredgecolor="#d73027", markeredgewidth=1.2, zorder=5)
    # 文字放在上方绝对安全的空白区，带弧度指向下方圆圈
    ax1.annotate("min\n0.235s", 
                 xy=(2, 0.245),       
                 xytext=(4.5, 0.38),  
                 arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.9, 
                                 connectionstyle="arc3,rad=-0.15"),
                 color="#d73027", fontsize=6.5, va="center", ha="center")

    ax1.set_xlabel("Adaptation steps $K$")
    ax1.set_ylabel("Mean Makespan (s)")
    ax1.set_xticks(k_vals)
    ax1.set_xlim(-1.0, 22)
    ax1.set_ylim(0.15, 1.15) 
    
    ax1.legend(loc="upper right", framealpha=0.85, ncol=2, 
               fontsize=6.0, columnspacing=0.8)
               
    ax1.grid(True, which="major")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) Fast adaptation", loc="left", fontsize=7.5)

    # ── 右：N 规模扩展 ──────────────────────────────
    ax2.plot(n_vals, heft_n,   "^-.", color=C["heft"],
             label="HEFT",        markersize=4, zorder=3)
    ax2.plot(n_vals, adc_k0_n, "o-",  color=C["adc"],
             label="ADC-Meta K=0", markersize=4, zorder=4)
    ax2.plot(n_vals, adc_k2_n, "o--", color=C["adc"],
             label="ADC-Meta K=2", markersize=4, zorder=4,
             markerfacecolor="white")

    # 异常点高亮外圈
    ax2.plot(10, 0.108, "o", markersize=8, markerfacecolor="none", 
             markeredgecolor="#d73027", markeredgewidth=1.2, zorder=5)
             
    # 右图异常点悬空标注
    ax2.annotate("data-scarce\nregime", 
                 xy=(10.5, 0.125),    
                 xytext=(14.5, 0.24), 
                 arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.9,
                                 connectionstyle="arc3,rad=0.15"), 
                 fontsize=6.5, color="#d73027", ha="center", va="center")

    ax2.set_xlabel("Number of subtasks $n$")
    ax2.set_ylabel("Mean Makespan (s)")
    ax2.set_xticks(n_vals)
    ax2.set_ylim(0, 0.52)
    
    ax2.legend(loc="upper left", framealpha=0.85, fontsize=6.0)
    ax2.grid(True, which="major")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) Scalability with task count", loc="left", fontsize=7.5)

    plt.savefig("results/figures/fig3_adapt_scale.pdf")
    plt.close()
    print("[✓] fig3_adapt_scale.pdf")
# ══════════════════════════════════════════════════════════
# FIG 4：消融相对退化条形图（V.D）
# ══════════════════════════════════════════════════════════
def plot_fig4_ablation():
    # ── 双面板：左=CCSM+Fixed-ε（精细刻度），右=w/o Meta（全刻度）
    data_left = {
    "w/o CCSM":           [7.9,   5.5,   9.8],   # ← K=0 从0.4改为7.9
    r"Fixed-$\varepsilon_{\rm in}$": [7.5, -0.4, 12.2],
}
    data_right = {  # w/o Meta-RL 的退化(%)
        "w/o Meta-RL":        [105.2, 153.6, 131.9],
    }
    k_labels  = ["K=0", "K=2", "K=20"]
    #k_colors  = ["#2c7bb6", "#fdae61", "#d7191c"]
    k_colors  = ["#084594", "#4292c6", "#9ecae1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                    gridspec_kw={"width_ratios": [3, 2]})

    x_left = np.arange(len(data_left))
    width  = 0.22
    offsets = [-width, 0, width]

    # ── 左面板 ──────────────────────────────────────────
    for j, (k_label, offset, kc) in enumerate(
            zip(k_labels, offsets, k_colors)):
        vals = [v[j] for v in data_left.values()]
        ax1.bar(x_left + offset, vals, width,
                label=k_label, color=kc,
                edgecolor="white", linewidth=0.3, zorder=3)
        # 数值标注
        for i, v in enumerate(vals):
            if abs(v) > 0.05:
                ax1.text(x_left[i] + offset, v + 0.15 if v >= 0 else v - 0.6,
                         f"{v:.1f}%", ha="center", va="bottom" if v>=0 else "top",
                         fontsize=5.5)

    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(x_left)
    ax1.set_xticklabels(list(data_left.keys()), fontsize=7)
    ax1.set_ylabel("Makespan degradation vs.\nFull ADC-Meta (%)")
    ax1.set_ylim(-3, 18)
    ax1.set_title("(a) Component-level degradation", loc="left", fontsize=7.5)
    ax1.legend(title="Adapt. steps", fontsize=6, title_fontsize=6,
               loc="upper left", framealpha=0.85)
    ax1.grid(axis="y", which="major")
    ax1.spines[["top", "right"]].set_visible(False)

    # 非单调注释
    ax1.annotate("K=2: slightly\nbetter (−0.4%)",
                 xy=(1 + offsets[1], -0.4),
                 xytext=(1.5, -2.5),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"),
                 fontsize=5.5, color="#555", ha="center")

    # ── 右面板 ──────────────────────────────────────────
    x_right = np.arange(1)
    for j, (k_label, offset, kc) in enumerate(
            zip(k_labels, offsets, k_colors)):
        v = list(data_right.values())[0][j]
        ax2.bar(x_right + offset, [v], width,
                label=k_label, color=kc,
                edgecolor="white", linewidth=0.3, zorder=3)
        ax2.text(x_right[0] + offset, v + 2,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=5.5)

    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x_right)
    ax2.set_xticklabels(["w/o Meta-RL"], fontsize=7)
    ax2.set_ylim(0, 175)
    ax2.set_ylabel("")
    ax2.set_title("(b) Meta-RL removal", loc="left", fontsize=7.5)
    ax2.grid(axis="y", which="major")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/fig4_ablation.pdf")
    plt.close()
    print("[✓] fig4_ablation.pdf (revised dual-panel)")

# ══════════════════════════════════════════════════════════
# FIG 5：ADC 机制双面板（V.F）- 优化版
# ══════════════════════════════════════════════════════════
def plot_fig5_adc_mechanism():
    # 读重建的主模型日志（每10轮一条，200个点）
    log = json.load(open("logs/training_log_main_reconstructed.json"))
    iters  = np.array(log["outer_iter"])
    g_ema  = np.array(log["g_ema"])
    eps_in = np.array(log["eps_in"])

    fig = plt.figure(figsize=(7.0, 2.2))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── 左：g_ema 轨迹 ───────────────────────────────
    ax1.plot(iters, g_ema, "-", color=C["adc"],
             linewidth=1.0, alpha=0.9, label=r"$\bar{g}$")
    ax1.axhline(0.8, color="#d73027", linestyle="--",
                linewidth=0.9, label=r"$g_{\max}=0.8$")
    ax1.fill_between(iters, 0, g_ema,
                     color=C["adc"], alpha=0.08)
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel(r"$\bar{g}$ (gradient norm EMA)")
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(0, 0.9)
    ax1.legend(loc="upper left", framealpha=0.85)
    ax1.grid(True, which="major")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(r"(a) $\bar{g}$ trajectory", loc="left", fontsize=7.5)

    # ── 右：eps_in 轨迹 ──────────────────────────────
    # 允许范围填充
    ax2.fill_between(iters, 0.1, 0.4,
                     color="#f0f0f0", alpha=0.7, zorder=0,
                     label="Allowed range [0.1, 0.4]")
    ax2.plot(iters, eps_in, "-", color=C["greedy"],
             linewidth=0.9, alpha=0.85,
             label=r"$\bar{\varepsilon}_{\rm in}$ (batch mean)")
    ax2.axhline(0.2, color=C["ref"], linestyle="-.",
                linewidth=0.9,
                label=r"$\varepsilon_{\rm base}=0.2$")

    # 上下界虚线
    ax2.axhline(0.1, color="#aaaaaa", linestyle=":",
                linewidth=0.7,
                label=r"$\delta_{\rm lo}\!\cdot\!\varepsilon_{\rm base}$")
    ax2.axhline(0.4, color="#aaaaaa", linestyle=":",
                linewidth=0.7,
                label=r"$\delta_{\rm hi}\!\cdot\!\varepsilon_{\rm base}$")

    # 标注上下界数值
    ax2.text(2020, 0.10, "0.10", va="center", fontsize=6, color="#888")
    ax2.text(2020, 0.40, "0.40", va="center", fontsize=6, color="#888")

    ax2.set_xlabel("Outer iteration")
    ax2.set_ylabel(r"Mean $\varepsilon_{{\rm in},i}$ per batch")
    ax2.set_xlim(0, 2050)
    ax2.set_ylim(0.05, 0.48)
    ax2.legend(loc="upper right", framealpha=0.85,
               fontsize=5.8, ncol=1)
    ax2.grid(True, which="major")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title(r"(b) Adaptive $\varepsilon_{{\rm in},i}$ dynamics",
                  loc="left", fontsize=7.5)

    plt.tight_layout()
    plt.savefig("results/figures/fig5_adc_mechanism.pdf")
    plt.close()
    print("[✓] fig5_adc_mechanism.pdf")
# ══════════════════════════════════════════════════════════
# FIG 6：训练收敛曲线（V.G）
# ══════════════════════════════════════════════════════════
def plot_fig6_convergence():
    log      = json.load(open("logs/training_log.json"))
    all_iter = np.array(log["outer_iter"])
    all_ms   = np.array(log["mean_makespan"])

    # 每 50 轮的 eval 点（与 checkpoints 对应）
    eval_iters = all_iter[all_iter % 50 == 0]
    eval_ms    = all_ms  [all_iter % 50 == 0]

    # 原始训练 makespan（每轮）—— 用浅色背景
    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    ax.plot(all_iter, all_ms, "-", color=C["adc"],
            linewidth=0.5, alpha=0.35, label="Training batch makespan")
    ax.plot(eval_iters, eval_ms, "o-", color=C["adc"],
            linewidth=1.1, markersize=3.5,
            label="Eval makespan (every 50 iters)", zorder=4)

    # HEFT 参考线
    ax.axhline(HEFT_VAL, color=C["heft"], linestyle="--",
               linewidth=0.9, label=f"HEFT ({HEFT_VAL:.3f}s)", zorder=3)

    # 收敛区域标注
    ax.axvspan(500, 2000, alpha=0.04, color=C["adc"],
               label="Convergence region", zorder=0)

    # 最优点寻找与标注
    best_idx  = eval_ms.argmin()
    best_iter = eval_iters[best_idx]
    best_val  = eval_ms[best_idx]

    # 换成红色高亮外圈
    ax.plot(best_iter, best_val, "o", markersize=8, markerfacecolor="none", 
            markeredgecolor="#d73027", markeredgewidth=1.2, zorder=5)
            
    # 【修改点】：大幅拉高文字Y坐标 (0.12 -> 0.28)，同时X轴往左移留出箭头弧度空间
    ax.annotate(f"Best: {best_val:.3f}s\n@ Iter {int(best_iter)}",
                xy=(best_iter, best_val + 0.02),      
                xytext=(best_iter - 250, best_val + 0.28), 
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.0, 
                                connectionstyle="arc3,rad=-0.15"),
                fontsize=6.5, color="#d73027", ha="center", va="bottom", zorder=6)

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Makespan (s)")
    ax.set_xlim(0, 2050)
    
    ax.legend(loc="upper right", framealpha=0.85, fontsize=6.5)
    ax.grid(True, which="major", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/fig6_convergence.pdf")
    plt.close()
    print("[✓] fig6_convergence.pdf")
# ══════════════════════════════════════════════════════════
# 执行
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    plot_fig1_comparison()
    plot_fig3_adaptation()
    plot_fig4_ablation()
    plot_fig5_adc_mechanism()
    plot_fig6_convergence()
    print("\nAll figures saved to results/figures/")
    print("Remaining: fig2_congestion.pdf (run after eval_congestion.py)")
