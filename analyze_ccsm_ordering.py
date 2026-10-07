"""
analyze_ccsm_ordering.py
=========================
【定稿主线的决定性统计补充】三种 CCSM 形式在 E=3 三档拥塞下的**两两**配对比较

为什么需要这个脚本
------------------
`results/paired_eval_E3v3.json` 里的 `paired_analysis` 只做了"每个消融 vs 完整方法(none)"
这一组比较，**没有做 `ccsm_cloud_only` vs `no_ccsm` 这一对**。而恰恰是这一对，
承载了论文多边缘一节最核心的那句话：

    「维数错配的拥塞先验，比完全不用先验更糟。」

从 json 的均值能看出苗头（high 档：单维 0.2470 vs 无先验 0.2241 vs 逐位置 0.2264），
但苗头不是证据——必须按 DAG 配对、给出自助法置信区间、并做 Bonferroni 校正，
才能写进论文。这个脚本就是补这一刀。

它做什么
--------
读 `results/paired_eval_E3v3.json` 的 `raw_makespans`，对 low / medium / high 三档，
在三个配置 {none(逐位置CCSM), ccsm_cloud_only(v2单维), no_ccsm(无先验)} 之间做
**全部 3 组两两配对比较**（共 3 档 × 3 对 = 9 次比较），统一 Bonferroni 校正，
输出一张可以直接誊进论文的完整序关系表。

统计口径与 paired_eval.py / eval_congestion_e3_panelb.py **完全一致**：
逐 DAG 配对求差 → 配对自助法（n_boot=10000, seed=12345）→ Bonferroni 校正。
不引入任何新的统计方法，保证与已有结果口径统一、可以并排陈述。

运行（服务器仓库根目录）：
    python -u analyze_ccsm_ordering.py

输出：
    results/ccsm_ordering_E3v3.json
    控制台打印每档的完整序关系与显著性判定

成本：纯读文件 + numpy 自助法，几秒钟跑完，不碰 GPU、不碰 PID 19229、
不需要任何 checkpoint。**这个脚本不重新评估任何模型，只是把已有的逐DAG数据
换一个角度做统计**，所以结果与已报告的数字之间不会有任何不一致。
"""

import json
import os
import itertools

import numpy as np

# 复用 paired_eval.py 里那一个自助法实现，保证与全部已报告结果同源同口径
from paired_eval import paired_bootstrap_ci

PAIRED_JSON = "results/paired_eval_E3v3.json"
OUT_JSON = "results/ccsm_ordering_E3v3.json"

# 三个 CCSM 形式（论文里的命名 → json 里的 tag）
CONFIGS = {
    "none":            "Per-position CCSM (ours)",
    "ccsm_cloud_only": "Cloud-only CCSM (v2 single-dim)",
    "no_ccsm":         "No CCSM",
}
LEVELS = ["low", "medium", "high"]


def main():
    if not os.path.isfile(PAIRED_JSON):
        print(f"✗ 找不到 {PAIRED_JSON}，请先确认 paired_eval.py 已跑完。")
        return

    with open(PAIRED_JSON) as f:
        pe = json.load(f)

    raw = pe["raw_makespans"]
    for tag in CONFIGS:
        if tag not in raw:
            print(f"✗ {PAIRED_JSON} 里没有配置 {tag}，无法做完整两两比较。")
            return

    n_dags = len(raw["none"]["low"])
    pairs = list(itertools.combinations(CONFIGS.keys(), 2))
    n_tests = len(pairs) * len(LEVELS)
    bonf_conf = 1.0 - 0.05 / n_tests

    print("=" * 84)
    print("E=3 异构多边缘：三种拥塞先验形式的两两配对比较")
    print("=" * 84)
    print(f"  测试集 n={n_dags}（seed={pe['seed']}），逐DAG配对 + 配对自助法 + Bonferroni")
    print(f"  共 {n_tests} 次比较（{len(pairs)} 对 × {len(LEVELS)} 档），"
          f"单次置信水平 {bonf_conf*100:.2f}%")
    print(f"  checkpoint：" + "  ".join(
        f"{t}=iter{pe['setups'][t]['iter']}" for t in CONFIGS))
    print()

    out = {
        "n_test": n_dags,
        "seed": pe["seed"],
        "n_tests": n_tests,
        "bonferroni_conf": bonf_conf,
        "means": {},
        "comparisons": {},
    }

    for lv in LEVELS:
        means = {t: float(np.mean(raw[t][lv])) for t in CONFIGS}
        out["means"][lv] = means

        # 按均值从快到慢排序，给出该档的序关系
        order = sorted(means, key=lambda t: means[t])
        order_str = "  <  ".join(f"{CONFIGS[t]}({means[t]:.4f}s)" for t in order)

        print("-" * 84)
        print(f"[{lv}]  按均值排序（越靠前越快）：")
        print(f"    {order_str}")
        print()

        for a, b in pairs:
            arr_a = np.asarray(raw[a][lv], dtype=float)
            arr_b = np.asarray(raw[b][lv], dtype=float)
            d = arr_b - arr_a                     # 正数 = b 更慢 = a 更好
            lo95, hi95 = paired_bootstrap_ci(d, conf=0.95)
            lob, hib = paired_bootstrap_ci(d, conf=bonf_conf)
            rel = d.mean() / arr_b.mean() * 100.0  # a 相对 b 缩短的百分比

            if lob > 0:
                verdict = f"{CONFIGS[a]} 更优 (Bonf显著)"
            elif hib < 0:
                verdict = f"{CONFIGS[b]} 更优 (Bonf显著)"
            elif lo95 > 0:
                verdict = f"{CONFIGS[a]} 更优 (仅95%)"
            elif hi95 < 0:
                verdict = f"{CONFIGS[b]} 更优 (仅95%)"
            else:
                verdict = "分辨不出"

            # worse_rate：b 比 a 慢的 DAG 占比，用来说明差异不是被少数极端值带偏的
            worse_rate = float(np.mean(d > 0))

            key = f"{lv}|{a}_vs_{b}"
            out["comparisons"][key] = {
                "mean_a": float(arr_a.mean()),
                "mean_b": float(arr_b.mean()),
                "mean_diff_b_minus_a": float(d.mean()),
                "rel_pct": float(rel),
                "ci95": [lo95, hi95],
                "ci_bonf": [lob, hib],
                "worse_rate_b_slower": worse_rate,
                "verdict": verdict,
            }

            print(f"    {a:<16} vs {b:<16}  差值={d.mean():+.4f}s ({rel:+.1f}%)  "
                  f"Bonf.CI=[{lob:+.4f},{hib:+.4f}]  b更慢占比={worse_rate:.1%}")
            print(f"        → {verdict}")
        print()

    # ── 把最核心的那条结论单独拎出来打印，避免被淹没在表里 ──────────────
    print("=" * 84)
    print("核心结论自检：「维数错配的先验是否比不用先验更糟」")
    print("=" * 84)
    for lv in LEVELS:
        k = f"{lv}|ccsm_cloud_only_vs_no_ccsm"
        c = out["comparisons"][k]
        # 这里 a=ccsm_cloud_only, b=no_ccsm，差值 = no_ccsm − cloud_only
        # 负数 = no_ccsm 更快 = 单维先验比不用先验还差
        d = c["mean_diff_b_minus_a"]
        if d < 0:
            tag = (f"单维先验比无先验**更差** {abs(d):.4f}s "
                   f"（{abs(c['rel_pct']):.1f}%）")
        else:
            tag = f"单维先验仍优于无先验 {d:.4f}s"
        print(f"  [{lv:<6}] 单维CCSM={c['mean_a']:.4f}s  无CCSM={c['mean_b']:.4f}s  "
              f"→ {tag}")
        print(f"           判定：{c['verdict']}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存 → {OUT_JSON}")
    print("\n完成。")


if __name__ == "__main__":
    main()
