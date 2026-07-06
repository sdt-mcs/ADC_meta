"""
eval_congestion.py
==================
回程拥塞敏感性实验（论文 Fig. 2 数据来源）

前置条件：
  env.py     第46行 reset() 签名已加 q_ec_range=None
  env.py     第55行 Q_ec 初始化已使用 q_ec_range
  evaluate.py eval_model_on_dags 已加 q_ec_range 参数

运行：
  python eval_congestion.py

输出：
  results/congestion_results.json
  results/figures/fig2_congestion.pdf
"""

import json, os, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config    import Config
from daggen_py import DaggenPy
from env       import ThreeTierEnv
from evaluate  import eval_model_on_dags, load_model, run_all_baselines

os.makedirs("results/figures", exist_ok=True)

# ── 全局 Q_ec 补丁（让 run_all_baselines 也能用受控范围）────
_Q_EC_OVERRIDE = None
_orig_reset = ThreeTierEnv.reset

def _patched_reset(self, dag, q_ec_range=None):
    effective = q_ec_range if q_ec_range is not None else _Q_EC_OVERRIDE
    return _orig_reset(self, dag, q_ec_range=effective)

ThreeTierEnv.reset = _patched_reset   # 打补丁（不影响 q_ec_range 已有逻辑）

# ── 绘图全局样式 ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.4,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"adc": "#2166ac", "no_ccsm": "#d73027",
     "greedy": "#d6604d", "heft": "#4d4d4d"}

# ── 拥塞档位 ─────────────────────────────────────────────
LEVELS = [
    ("Low [0–15 MB]",     0.0,  15.0),
    ("Medium [15–35 MB]", 15.0, 35.0),
    ("High [35–50 MB]",   35.0, 50.0),
]
N_TEST   = 30
SEED_DAG = 999
SEED_ENV = 0


def run_congestion_experiment():
    global _Q_EC_OVERRIDE
    cfg = Config()

    # 生成固定测试集
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
    print(f"测试集：{N_TEST} 个 DAG（seed={SEED_DAG}）\n")

    model_adc    = load_model("checkpoints/model_iter1750.pt",   cfg)
    model_noccsm = load_model("checkpoints/ablation_no_ccsm.pt", cfg)

    results = {
        "labels":   [],
        "ADC_Meta": [], "w/o_CCSM": [],
        "Greedy":   [], "HEFT":     [],
    }

    for label, lo, hi in LEVELS:
        q_range = (lo, hi)
        print(f"── {label.replace(chr(10), ' ')}  Q_ec∈[{lo:.0f},{hi:.0f}] MB ──")

        # ① ADC-Meta K=0（直接用 q_ec_range 参数）
        ms = eval_model_on_dags(model_adc, test_dags, cfg,
                                adapt_steps=0, seed=SEED_ENV,
                                q_ec_range=q_range)
        mean_adc = float(np.mean(ms))
        print(f"  ADC-Meta : {mean_adc:.3f}s")

        # ② w/o CCSM K=0
        ms = eval_model_on_dags(model_noccsm, test_dags, cfg,
                                adapt_steps=0, seed=SEED_ENV,
                                q_ec_range=q_range)
        mean_noccsm = float(np.mean(ms))
        print(f"  w/o CCSM : {mean_noccsm:.3f}s")

        # ③ Greedy + HEFT（通过全局补丁控制 Q_ec）
        _Q_EC_OVERRIDE = q_range
        bl = run_all_baselines(test_dags, cfg)
        _Q_EC_OVERRIDE = None          # 恢复
        mean_greedy = next(r["mean"] for r in bl if r["name"] == "Greedy")
        mean_heft   = next(r["mean"] for r in bl if r["name"] == "HEFT")
        print(f"  Greedy   : {mean_greedy:.3f}s")
        print(f"  HEFT     : {mean_heft:.3f}s\n")

        results["labels"].append(label.replace("\n", " "))
        results["ADC_Meta"].append(mean_adc)
        results["w/o_CCSM"].append(mean_noccsm)
        results["Greedy"].append(float(mean_greedy))
        results["HEFT"].append(float(mean_heft))

    out = "results/congestion_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存 → {out}")
    return results


def plot_fig2(results):
    labels  = results["labels"]
    methods = ["HEFT",          "ADC_Meta",          "w/o_CCSM",  "Greedy"]
    # 【修改点1】精简图例名称，去除后缀，确保 1 行 4 列能完美放得下
    display = ["HEFT", "ADC-Meta", "w/o CCSM", "Greedy"]
    colors  = [C["heft"],        C["adc"],              C["no_ccsm"], C["greedy"]]
    hatches = ["//",             "",                    "xx",         ""]

    x      = np.arange(len(labels))
    width  = 0.17
    n      = len(methods)
    starts = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)

    # 【修改点2】稍微把画布加宽一点点 (3.5 -> 3.8)，给上方并排的 4 个图例留足空间
    fig, ax = plt.subplots(figsize=(3.8, 2.4))

    max_val = 0  # 追踪最大值，用于后续动态设置 Y 轴上限

    for m, disp, c, h, s in zip(methods, display, colors, hatches, starts):
        vals = results[m]
        max_val = max(max_val, max(vals))
        
        # zorder=3 让柱子在网格线上方
        bars = ax.bar(x + s, vals, width, label=disp,
                      color=c, hatch=h,
                      edgecolor="white", linewidth=0.3, zorder=3)
                      
        # 仅标注 ADC-Meta 和 w/o CCSM 的数值
        if m in ("ADC_Meta", "w/o_CCSM"):
            for bar, v in zip(bars, vals):
                # 【修改点3】增大文字与柱顶的距离 (0.003 -> 0.015)，让呼吸感更好
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + 0.015, f"{v:.3f}",
                        ha="center", va="bottom",
                        fontsize=5.5, color="#222", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([lb.replace(" ", "\n") for lb in labels], fontsize=7)
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_xlabel(r"Backhaul congestion level ($Q_{ec}$ initial range)")
    
    # 【修改点4】动态设置 Y 轴上限留白 25%，防止柱体上方的数字被切掉
    ax.set_ylim(0, max_val * 1.25)
    
    # 【修改点5】将图例移出图表，放在外侧正上方，1 行 4 列排开
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.90),
              ncol=4, framealpha=0.9, fontsize=6.0,
              columnspacing=1.0, handletextpad=0.4)
              
    ax.grid(axis="y", which="major", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/fig2_congestion.pdf")
    plt.close()
    print("[✓] fig2_congestion.pdf")

if __name__ == "__main__":
    results = run_congestion_experiment()
    plot_fig2(results)
    print("\n完成。")
