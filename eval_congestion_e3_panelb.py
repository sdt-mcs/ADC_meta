"""
eval_congestion_e3_panelb.py
=============================
【蓝图第八节 实验 1c —— 拥塞双面板图 panel (b) 的最终数据与作图】

一句话：把 `paired_eval.py` 已经算好的三条学习模型曲线，和**在完全相同测试集、
完全相同逐DAG种子下重算的**经典基线曲线合并成 panel (b)。

────────────────────────────────────────────────────────────────────
为什么这个脚本取代了 eval_congestion_e3_baselines_3level.py
────────────────────────────────────────────────────────────────────
读了服务器上的 `paired_eval.py` 之后发现两件事：

1. **学习模型的三档数据其实已经有了**，不需要新写评估脚本。
   `paired_eval.py` 的 LEVELS 就是 {default, low[0,15], medium[15,35],
   high[35,50]}，和 `eval_congestion.py`（E=1 侧 Table IV 用的那套）逐字相同，
   而且它已经把六个配置（含 none / ccsm_cloud_only / no_ccsm）在这四档上
   全部跑过，结果存在 `results/paired_eval_E3v3.json` 的 `raw_makespans` 里，
   是逐DAG的原始 makespan 列表，不是只有均值。panel (b) 要的三条模型曲线
   直接从这里读即可。

2. **但基线不能沿用我上一版脚本的口径。** `paired_eval.py` 用的是
   n=200、seed=999 的测试集，且每个 DAG 用固定的 `np.random.default_rng(1000+i)`
   逐DAG配对；而 `eval_congestion_e3_baselines_3level.py` 用的是 n=30、
   整批一个 `seed=0` 的环境种子。两者不是同一批 DAG、也不是同一套信道实现，
   放进同一个 panel 里比较是不成立的。所以本脚本把四个经典基线**重算一遍**，
   逐行对齐 `paired_eval.py` 的测试集与逐DAG种子：

       测试集   ev.generate_test_dags(Config(), n=<json里的n_test>, seed=999)
       环境种子 ThreeTierEnv(cfg, np.random.default_rng(1000 + i))   # i = DAG 序号
       档位定义 直接 from paired_eval import LEVELS（同一个对象，杜绝抄错）

   `eval_congestion_e3_baselines_3level.py` 与 6 档版本一起，降级为附录/内部
   探索性数据，不再是 panel (b) 的数据来源。

────────────────────────────────────────────────────────────────────
一个必须知道的配对口径细节（与 paired_eval.py 完全一致的已知局限）
────────────────────────────────────────────────────────────────────
模型侧 K>0 时，内循环适配的 rollout 会额外消耗环境随机数，所以模型最终那一次
贪心 episode 的信道实现，和基线第一次 episode 的信道实现并不相同。也就是说
**配对是配在 DAG 身份上，不是配在信道实现上**。`paired_eval.py` 的文档里已经
写明这一点，并指出这会让检验略偏保守、不会制造假阳性。本脚本沿用同一口径，
不做额外处理，保持两边可比。

────────────────────────────────────────────────────────────────────
运行
────────────────────────────────────────────────────────────────────
前置条件：`results/paired_eval_E3v3.json` 必须已经存在（即 `paired_eval.py`
已经跑完）。如果没有，先跑：

    python -u paired_eval.py

然后：

    python -u eval_congestion_e3_panelb.py

可选参数：
    --paired  results/paired_eval_E3v3.json   # 换一个 paired_eval 结果文件
    --levels  low,medium,high                 # 默认就是这三档（正式图用）
    --out     results/congestion_e3_panelb.json

输出：
    results/congestion_e3_panelb.json                      —— 合并后的逐DAG原始数据 + 均值 + 配对统计
    results/figures/fig_congestion_e3_panelb.pdf           —— panel (b) 正式草图

成本：纯 CPU 规则调度器推理，n=200 × 4档 × 4个基线 ≈ 3200 个 episode，
通常几分钟内跑完。**不使用 CUDA，不占用 cuda:0 / cuda:1，与 PID 19229 及你
正在跑的任何训练完全隔离。**

────────────────────────────────────────────────────────────────────
跑完必看的三条健全性检查（脚本自动执行并打印）
────────────────────────────────────────────────────────────────────
1. LocalOnly / EdgeOnly 跨三档的 mean makespan 极差应 ≈ 0
   —— 这两个策略从不使用回程链路，理论上不随 Q_ec 变化。不通过 = 补丁没生效。
2. default 档与 low 档的逐DAG makespan 应当**逐位完全相同**
   —— `结论状态_当前.md` 明确记录"`env.reset()` 不传 q_ec_range 时默认就是
   Uniform(0,15)MB，与 low 档数据逐位相同"。这条同时验证了两件事：
   `_Q_EC_OVERRIDE` 补丁确实生效，且那条文档结论在当前代码下依然成立。
3. 从 json 读到的 DAG 数量与本脚本重新生成的 DAG 数量一致
   —— 防止 paired_eval 跑的是 n=100、而这里默认按 200 重算导致错位配对。
任何一条不通过，脚本会明确打印 ❌ 并说明该查什么，此时**不要**用这份数据作图。
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from env import ThreeTierEnv
import evaluate as ev
from baseline import (
    LocalOnlyPolicy, EdgeOnlyPolicy, GreedyPolicy, HEFTPolicy,
)

# 直接复用 paired_eval 的档位定义与自助法实现，保证与模型侧是同一套口径。
# （paired_eval.py 的 main() 在 __main__ 保护下，import 它不会触发任何评估）
from paired_eval import LEVELS, paired_bootstrap_ci

os.makedirs("results/figures", exist_ok=True)

# ── 全局 Q_ec 补丁：与 eval_congestion.py / paired_eval 的 q_ec_range 语义一致 ──
# baseline 的 run_episode() 内部调用 env.reset(dag)，不带 q_ec_range 参数，
# 所以这里给 reset 打补丁，让它在没有显式传参时使用我们指定的范围。
_Q_EC_OVERRIDE = None
_orig_reset = ThreeTierEnv.reset


def _patched_reset(self, dag, q_ec_range=None):
    effective = q_ec_range if q_ec_range is not None else _Q_EC_OVERRIDE
    return _orig_reset(self, dag, q_ec_range=effective)


ThreeTierEnv.reset = _patched_reset

# panel (b) 要画的三条模型曲线，键名 = paired_eval.py 里的 tag
MODEL_SERIES = {
    "none":             "ADC-Meta (ours)",
    "ccsm_cloud_only":  "CCSM cloud-only",
    "no_ccsm":          "w/o CCSM",
}
# 重算的经典基线（Random 不进图，与 E=1 侧 Table IV 的呈现方式保持一致）
BASELINE_SERIES = ["LocalOnly", "EdgeOnly", "Greedy", "HEFT"]
# 实际画进 panel (b) 的基线（LocalOnly ≈ 2.3s，画进去会把纵轴压平，只留在 json 里）
BASELINE_PLOTTED = ["Greedy", "HEFT"]

# 与 paired_eval.py 的 eval_one_dag 一致：第 i 个 DAG 用 np 种子 1000+i
SEED_ENV_BASE = 1000


def build_policies(cfg):
    """每档、每个 DAG 都重新构造，避免 HEFT 的 _priority_ptr 等内部状态串档。"""
    return [LocalOnlyPolicy(), EdgeOnlyPolicy(), GreedyPolicy(), HEFTPolicy(cfg)]


def run_baselines_paired(cfg, dags, level_names):
    """
    逐DAG、逐档位重算四个经典基线，种子口径与 paired_eval.py 逐行对齐。
    返回 {policy_name: {level: [每个DAG的makespan]}}
    """
    global _Q_EC_OVERRIDE
    out = {name: {lv: [] for lv in level_names} for name in BASELINE_SERIES}

    for lv in level_names:
        _Q_EC_OVERRIDE = LEVELS[lv]          # None（default档）也是合法值，走 env 自身默认
        print(f"  [基线] {lv:<8} Q_ec={LEVELS[lv]} ...", end="", flush=True)
        for i, dag in enumerate(dags):
            # 关键：每个策略都用**各自独立、同样以 1000+i 起始**的 rng 新建 env。
            # 不能四个策略共用一个 env —— 那样第二个策略拿到的信道实现会是第一个
            # 策略消耗过随机数之后的下一批，与模型侧 eval_model_on_dags 内部
            # 「每次评估都新建 ThreeTierEnv(cfg, default_rng(1000+i))」的口径对不上。
            for policy in build_policies(cfg):
                env = ThreeTierEnv(cfg, np.random.default_rng(SEED_ENV_BASE + i))
                ms = policy.run_episode(dag, env)
                out[policy.name][lv].append(float(ms))
        means = "  ".join(f"{n}={np.mean(out[n][lv]):.3f}s" for n in BASELINE_SERIES)
        print(f" 完成   {means}", flush=True)
    _Q_EC_OVERRIDE = None
    return out


def sanity_checks(baseline_raw, model_raw, n_dags_json, n_dags_regen, level_names):
    print("\n" + "=" * 78)
    print("健全性检查（三条都必须通过，任何一条 ❌ 都不要拿这份数据作图）")
    print("=" * 78)
    ok = True

    # ① LocalOnly / EdgeOnly 不应随拥塞档位变化
    curve_levels = [lv for lv in level_names if lv != "default"]
    for name in ("LocalOnly", "EdgeOnly"):
        means = [float(np.mean(baseline_raw[name][lv])) for lv in curve_levels]
        spread = max(means) - min(means)
        flag = "✅ 通过" if spread < 1e-6 else "❌ 异常"
        if spread >= 1e-6:
            ok = False
        print(f"  ① {name:<10} 跨 {len(curve_levels)} 档极差 = {spread:.6f}s  {flag}"
              f"（该策略从不使用回程链路，期望 ≈ 0）")

    # ② default 档应与 low 档逐位相同
    if "default" in level_names and "low" in level_names:
        same_all = True
        for name in BASELINE_SERIES:
            a = np.asarray(baseline_raw[name]["default"])
            b = np.asarray(baseline_raw[name]["low"])
            if not np.allclose(a, b, rtol=0, atol=0):
                same_all = False
        flag = "✅ 通过" if same_all else "❌ 异常"
        if not same_all:
            ok = False
        print(f"  ② default 档与 low 档逐DAG makespan 完全相同：{flag}"
              f"（验证 env.reset() 默认即 Uniform(0,15)MB，与文档结论一致）")
    else:
        print("  ② 跳过（本次未同时计算 default 与 low 两档）")

    # ③ DAG 数量一致
    flag = "✅ 通过" if n_dags_json == n_dags_regen else "❌ 异常"
    if n_dags_json != n_dags_regen:
        ok = False
    print(f"  ③ paired_eval 的 DAG 数({n_dags_json}) 与本脚本重算的 DAG 数({n_dags_regen})"
          f"：{flag}")

    print("=" * 78)
    if not ok:
        print("⚠️  有检查未通过。最常见原因：")
        print("    ①不过 → ThreeTierEnv.reset 的签名变了，_Q_EC_OVERRIDE 补丁没生效；")
        print("    ②不过 → env.py 里 Q_ec 的默认初始化范围被改过，不再是 Uniform(0,15)；")
        print("    ③不过 → paired_eval.py 当时用的 --n_test 不是 200，需用同样的 n 重跑本脚本。")
    return ok


def paired_vs_ours(baseline_raw, model_raw, level_names):
    """
    按 DAG 配对，比较 HEFT / Greedy 相对本方法(none)的差值，
    统计方法与 paired_eval.py 完全相同（配对自助法 + Bonferroni 校正）。
    正数 = 该基线比本方法更慢 = 本方法更好。
    """
    curve_levels = [lv for lv in level_names if lv != "default"]
    n_tests = len(BASELINE_PLOTTED) * len(curve_levels)
    bonf_conf = 1.0 - 0.05 / n_tests

    print("\n" + "=" * 78)
    print("本方法(none) vs 经典基线：按DAG配对差值")
    print(f"正数 = 基线更慢 = 本方法更好；共 {n_tests} 次比较，"
          f"Bonferroni 后单次置信水平 {bonf_conf * 100:.2f}%")
    print("=" * 78)

    analysis = {}
    for lv in curve_levels:
        ours = np.asarray(model_raw["none"][lv], dtype=float)
        print(f"\n  [{lv}]  本方法 = {ours.mean():.4f}s")
        print(f"    {'基线':<12} {'基线均值':>10} {'差值':>10} {'相对提升':>10} "
              f"{'Bonf.CI':>22}")
        for name in BASELINE_PLOTTED:
            arr = np.asarray(baseline_raw[name][lv], dtype=float)
            d = arr - ours                       # 正数 = 基线更慢
            lo95, hi95 = paired_bootstrap_ci(d, conf=0.95)
            lob, hib = paired_bootstrap_ci(d, conf=bonf_conf)
            rel = d.mean() / arr.mean() * 100.0  # 本方法相对该基线缩短的百分比
            if lob > 0:
                verdict = "本方法更优(Bonf显著)"
            elif hib < 0:
                verdict = "基线更优(Bonf显著)"
            elif lo95 > 0:
                verdict = "本方法更优(仅95%)"
            elif hi95 < 0:
                verdict = "基线更优(仅95%)"
            else:
                verdict = "分辨不出"
            analysis[f"{lv}|{name}"] = {
                "baseline_mean": float(arr.mean()),
                "ours_mean": float(ours.mean()),
                "mean_diff": float(d.mean()),
                "rel_improvement_pct": float(rel),
                "ci95": [lo95, hi95], "ci_bonf": [lob, hib],
                "verdict": verdict,
            }
            print(f"    {name:<12} {arr.mean():>10.4f} {d.mean():>+10.4f} "
                  f"{rel:>9.1f}% [{lob:>+9.4f},{hib:>+9.4f}]  {verdict}")

    print("\n  注：这里的配对口径与 paired_eval.py 完全一致——配在 DAG 身份上，")
    print("      不配在信道实现上（模型侧 K>0 的适配 rollout 会多消耗随机数）。")
    print("      该口径略偏保守，不会制造假阳性。")
    return analysis


def plot_panel_b(model_raw, baseline_raw, level_names, out_pdf):
    curve_levels = [lv for lv in level_names if lv != "default"]
    x = np.arange(len(curve_levels))

    # 与 eval_congestion.py 的配色保持同一体系，便于 (a)(b) 两个 panel 视觉统一
    style = {
        "ADC-Meta (ours)":  ("#2166ac", "o", "-"),
        "CCSM cloud-only":  ("#4393c3", "s", "--"),
        "w/o CCSM":         ("#d73027", "^", "--"),
        "Greedy":           ("#d6604d", "v", ":"),
        "HEFT":             ("#4d4d4d", "D", ":"),
    }

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    fig, ax = plt.subplots(figsize=(3.8, 2.4))

    for tag, label in MODEL_SERIES.items():
        vals = [float(np.mean(model_raw[tag][lv])) for lv in curve_levels]
        c, m, ls = style[label]
        ax.plot(x, vals, marker=m, markersize=3.2, linestyle=ls,
                color=c, label=label)

    for name in BASELINE_PLOTTED:
        vals = [float(np.mean(baseline_raw[name][lv])) for lv in curve_levels]
        c, m, ls = style[name]
        ax.plot(x, vals, marker=m, markersize=3.2, linestyle=ls,
                color=c, label=name)

    # x 轴标签与 E=1 侧 Table IV / Fig.2 的三档一致
    pretty = {"low": "Low\n[0–15]", "medium": "Medium\n[15–35]", "high": "High\n[35–50]"}
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(lv, lv) for lv in curve_levels], fontsize=7)
    ax.set_ylabel("Mean Makespan (s)")
    ax.set_xlabel(r"Backhaul congestion level ($Q_{ec}$ initial range, MB)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
              framealpha=0.9, columnspacing=1.0, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.4, linewidth=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"\n[done] {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paired", type=str, default="results/paired_eval_E3v3.json",
                   help="paired_eval.py 的输出文件")
    p.add_argument("--levels", type=str, default="default,low,medium,high",
                   help="要计算的档位；default 档只用于健全性检查②，不进图")
    p.add_argument("--out", type=str, default="results/congestion_e3_panelb.json")
    args = p.parse_args()

    # ── 读 paired_eval 结果 ────────────────────────────────────────
    if not os.path.isfile(args.paired):
        print(f"✗ 找不到 {args.paired}。panel (b) 的三条模型曲线来自 paired_eval.py 的结果，")
        print("  请先在同一目录下跑：python -u paired_eval.py")
        return
    with open(args.paired) as f:
        pe = json.load(f)

    model_raw = pe["raw_makespans"]
    n_test = int(pe["n_test"])
    pe_levels = list(pe["levels"].keys())

    missing_tag = [t for t in MODEL_SERIES if t not in model_raw]
    if missing_tag:
        print(f"✗ {args.paired} 里缺少配置 {missing_tag} 的结果，panel (b) 画不全。")
        print(f"  现有配置：{list(model_raw)}")
        return

    level_names = [s.strip() for s in args.levels.split(",") if s.strip()]
    missing_lv = [lv for lv in level_names if lv not in pe_levels]
    if missing_lv:
        print(f"✗ {args.paired} 里没有档位 {missing_lv}（它当时用的是 --levels "
              f"{','.join(pe_levels)}）。")
        print(f"  请先用 python -u paired_eval.py --levels {args.levels} 重跑，"
              f"或把本脚本的 --levels 缩小到 {','.join(pe_levels)}。")
        return

    print("=" * 78)
    print("panel (b) 数据合成")
    print("=" * 78)
    print(f"  模型曲线来源 : {args.paired}")
    print(f"    n_test={n_test}  seed={pe['seed']}  adapt_lr={pe['adapt_lr']}  "
          f"adapt_episodes={pe['adapt_episodes']}")
    for tag, label in MODEL_SERIES.items():
        s = pe["setups"][tag]
        print(f"    {tag:<17} → {label:<18} iter{s['iter']}  K={s['K']}  {s['ckpt']}")
    print(f"  档位         : {[(lv, LEVELS[lv]) for lv in level_names]}")

    # ── 用与 paired_eval 完全相同的方式重建测试集 ──────────────────
    cfg = Config()            # 基线不涉及消融，用默认（= none 配置）的环境
    assert cfg.env.num_edges == 3, f"期望 E=3，实际 {cfg.env.num_edges}"
    dags = ev.generate_test_dags(cfg, n=n_test, seed=pe["seed"])
    print(f"  测试集       : ev.generate_test_dags(Config(), n={n_test}, "
          f"seed={pe['seed']})  → {len(dags)} 个 DAG")
    print(f"  逐DAG环境种子: np.random.default_rng({SEED_ENV_BASE}+i)"
          f"（与 paired_eval.py 的 eval_one_dag 一致）\n")

    # ── 重算基线 ──────────────────────────────────────────────────
    baseline_raw = run_baselines_paired(cfg, dags, level_names)

    # ── 健全性检查 ────────────────────────────────────────────────
    n_dags_json = len(model_raw["none"][level_names[0]])
    ok = sanity_checks(baseline_raw, model_raw, n_dags_json, len(dags), level_names)

    # ── 绝对值总表 ────────────────────────────────────────────────
    curve_levels = [lv for lv in level_names if lv != "default"]
    print("\n" + "=" * 78)
    print("panel (b) 绝对 makespan 均值（秒）")
    print("=" * 78)
    print(f"  {'系列':<20} " + " ".join(f"{lv:>10}" for lv in curve_levels))
    for tag, label in MODEL_SERIES.items():
        row = " ".join(f"{np.mean(model_raw[tag][lv]):>10.4f}" for lv in curve_levels)
        print(f"  {label:<20} {row}")
    for name in BASELINE_SERIES:
        row = " ".join(f"{np.mean(baseline_raw[name][lv]):>10.4f}" for lv in curve_levels)
        extra = "" if name in BASELINE_PLOTTED else "   (仅存 json，不进图)"
        print(f"  {name:<20} {row}{extra}")

    # ── 配对统计 ──────────────────────────────────────────────────
    analysis = paired_vs_ours(baseline_raw, model_raw, level_names)

    # ── 落盘 ──────────────────────────────────────────────────────
    payload = {
        "source_paired_eval": args.paired,
        "n_test": n_test,
        "seed": pe["seed"],
        "levels": {lv: LEVELS[lv] for lv in level_names},
        "plotted_levels": curve_levels,
        "seed_env_base": SEED_ENV_BASE,
        "model_setups": {t: pe["setups"][t] for t in MODEL_SERIES},
        "model_raw_makespans": {t: {lv: model_raw[t][lv] for lv in level_names}
                                for t in MODEL_SERIES},
        "baseline_raw_makespans": baseline_raw,
        "means": {
            **{MODEL_SERIES[t]: {lv: float(np.mean(model_raw[t][lv]))
                                 for lv in level_names} for t in MODEL_SERIES},
            **{n: {lv: float(np.mean(baseline_raw[n][lv])) for lv in level_names}
               for n in BASELINE_SERIES},
        },
        "paired_vs_ours": analysis,
        "sanity_checks_passed": bool(ok),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n结果已保存 → {args.out}")

    # ── 作图 ──────────────────────────────────────────────────────
    if ok:
        plot_panel_b(model_raw, baseline_raw, level_names,
                     "results/figures/fig_congestion_e3_panelb.pdf")
    else:
        print("\n⚠️  健全性检查未全部通过，已跳过作图。请先按上面的提示排查。")

    print("\n完成。")


if __name__ == "__main__":
    main()
