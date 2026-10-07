"""
eval_congestion_e3_heft_meanparam.py
=====================================
【承接 panelb_finding.md 第三节的"待核实项 A"，现已核实，这是核实后的下一步】

已确认的事实（你贴的 sed 输出）：当前 baseline.py 的 HEFTPolicy._heft_exec
边缘分支用 `task.cpu / st.f_edge[e]` —— 逐边真实算力，不是均值。
而 iotj.tex Section V.A 描述、以及 panel (a)（E=1）的 HEFT，用的是均值参数口径。
两个 panel 的 HEFT 不是同一个口径，这一点现在不再是假设。

这个脚本做且只做一件事：构造一个"其余逻辑完全不变，只把边缘算力从
逐边真值 st.f_edge[e] 换成均值 np.mean(st.f_edge)"的 HEFT 变体
（HEFTPolicy_MeanEdge），在与 congestion_e3_panelb.json **完全相同**的测试集
/ 逐DAG种子 / 拥塞档位下重算一遍，输出三条 HEFT 数字放在一起看：

    HEFT-true   —— 现有 baseline.py 的 HEFT（逐边真值，congestion_e3_panelb.json 里已有）
    HEFT-mean   —— 本脚本新算的变体（只把 f_edge 换成均值，其余不动，
                    包括回程带宽仍沿用现有代码已经在用的 mean_rate_ec 处理）
    ADC-Meta    —— 本方法（none），同样已在 congestion_e3_panelb.json 里

**为什么只改 f_edge、不改回程带宽**：现有 baseline.py 的云端分支本来就已经用
`mean_rate_ec = np.mean(st.rate_ec_max)`（均值），没有用逐边真值——不对称只发生在
"边缘算力"这一项上。所以把 f_edge 也换成同样的"对本次采样取均值"处理，
是让 HEFT 内部对"边缘"和"回程"两类参数使用**同一种口径**，而不是引入一个新口径。
这样 HEFT-mean 与 HEFT-true 之间的差距，就精确等于"HEFT 是否能看见/利用
边缘间算力异构性"这一件事造成的差距，其余（回程口径、拥塞档位、测试集、种子）
全部对齐，可以干净地做减法：
    (HEFT-mean 与 HEFT-true 的差距)              → 有多少差距来自"口径"
    (ADC-Meta 与 HEFT-mean 的差距)                → 剩下多少差距来自"策略本身"

【自律声明，与 panelb_finding.md 第三节一致】本脚本不是为了给 ADC-Meta 找一个
更好看的基线换上去。两个口径都算、都报，congestion_e3_panelb.json 里
HEFT-true 的数字不删除、不替换，只是在旁边多放一条 HEFT-mean 作对照。
最终报哪个/两个都报，由你决定，这个脚本本身不预设结论。

运行（仓库根目录，results/congestion_e3_panelb.json 必须已存在）：
    python -u eval_congestion_e3_heft_meanparam.py

输出：
    results/congestion_e3_heft_meanparam.json
    （控制台会打印一张三行对照表：HEFT-true / HEFT-mean / ADC-Meta，
      并对 HEFT-true vs HEFT-mean 做配对 bootstrap+Bonferroni 检验，
      判断"逐边真值 vs 均值"这一处口径差异本身是否统计显著）

成本：与 eval_congestion_e3_panelb.py 的基线部分完全一样量级
（纯 CPU 规则调度器，n_test × 3 档，几分钟内跑完），只多算一个策略（HEFT-mean），
不使用 CUDA，不碰 PID 19229 或你正在跑的任何训练。
"""

import json
import os

import numpy as np

from config import Config
from env import ThreeTierEnv
import evaluate as ev
from baseline import HEFTPolicy

from paired_eval import LEVELS, paired_bootstrap_ci

# ── 与 eval_congestion_e3_panelb.py 完全一致的 Q_ec 补丁 ──────────────────
_Q_EC_OVERRIDE = None
_orig_reset = ThreeTierEnv.reset


def _patched_reset(self, dag, q_ec_range=None):
    effective = q_ec_range if q_ec_range is not None else _Q_EC_OVERRIDE
    return _orig_reset(self, dag, q_ec_range=effective)


ThreeTierEnv.reset = _patched_reset

SEED_ENV_BASE = 1000   # 与 paired_eval.py / eval_congestion_e3_panelb.py 一致
CURVE_LEVELS = ["low", "medium", "high"]


class HEFTPolicy_MeanEdge(HEFTPolicy):
    """
    与 baseline.HEFTPolicy 唯一的区别：_heft_exec 的边缘分支用
    float(np.mean(st.f_edge)) 代替 st.f_edge[e]。
    其余（本地分支、云端分支、_compute_priority、select_action、run_episode）
    全部继承自 HEFTPolicy，一字不改——保证除了这一处口径之外，其他任何行为
    都不会意外产生差异。
    """
    name = "HEFT-mean"

    def _heft_exec(self, task, x: int, env: ThreeTierEnv) -> float:
        ec = self.cfg.env
        st = env.state
        avg_rate_ue = float(np.mean(list(ec.rate_ue_options)))
        if x == env.LOC_LOCAL:
            return task.cpu / ec.f_local
        if 1 <= x <= env.E:
            # 唯一改动：逐边真值 st.f_edge[e]  →  本次采样的边缘算力均值
            mean_f_edge = float(np.mean(st.f_edge))
            return (task.data_in / avg_rate_ue
                    + task.cpu / mean_f_edge
                    + task.data_out / ec.rate_eu)
        # 云端分支：与现有 HEFTPolicy 完全相同（本来就已经是均值口径）
        mean_rate_ec = float(np.mean(st.rate_ec_max))
        r_ec = env._eff_ec_rate(0.0, mean_rate_ec)
        return (task.data_in / avg_rate_ue
                + task.data_in / r_ec
                + task.cpu / ec.f_cloud
                + task.data_out / ec.rate_ce
                + task.data_out / ec.rate_eu)


def run_heft_mean(cfg, dags):
    global _Q_EC_OVERRIDE
    out = {lv: [] for lv in CURVE_LEVELS}
    for lv in CURVE_LEVELS:
        _Q_EC_OVERRIDE = LEVELS[lv]
        print(f"  [HEFT-mean] {lv:<8} Q_ec={LEVELS[lv]} ...", end="", flush=True)
        for i, dag in enumerate(dags):
            env = ThreeTierEnv(cfg, np.random.default_rng(SEED_ENV_BASE + i))
            policy = HEFTPolicy_MeanEdge(cfg)   # 每个 DAG 重新构造，避免内部状态串档
            ms = policy.run_episode(dag, env)
            out[lv].append(float(ms))
        print(f" 完成  mean={np.mean(out[lv]):.4f}s", flush=True)
    _Q_EC_OVERRIDE = None
    return out


def main():
    panelb_path = "results/congestion_e3_panelb.json"
    if not os.path.isfile(panelb_path):
        print(f"✗ 找不到 {panelb_path}，请先跑 eval_congestion_e3_panelb.py。")
        return
    with open(panelb_path) as f:
        pb = json.load(f)

    n_test = int(pb["n_test"])
    seed = int(pb["seed"])
    cfg = Config()
    assert cfg.env.num_edges == 3, f"期望 E=3，实际 {cfg.env.num_edges}"

    # 与 congestion_e3_panelb.json 完全一致的方式重建测试集，保证逐DAG可比
    dags = ev.generate_test_dags(cfg, n=n_test, seed=seed)
    n_dags_json = len(pb["baseline_raw_makespans"]["HEFT"]["low"])
    print(f"测试集：n_test={n_test} seed={seed} → 重算得到 {len(dags)} 个 DAG"
          f"（json 里 HEFT-true 的 DAG 数 = {n_dags_json}）")
    if len(dags) != n_dags_json:
        print("❌ DAG 数量对不上，停止——说明测试集生成方式与 congestion_e3_panelb.json "
              "不一致，不能配对比较。")
        return

    heft_mean_raw = run_heft_mean(cfg, dags)

    heft_true_raw = {lv: pb["baseline_raw_makespans"]["HEFT"][lv] for lv in CURVE_LEVELS}
    ours_raw = {lv: pb["model_raw_makespans"]["none"][lv] for lv in CURVE_LEVELS}

    print("\n" + "=" * 78)
    print("三条 HEFT/本方法对照（均值，秒）")
    print("=" * 78)
    print(f"  {'系列':<14} " + " ".join(f"{lv:>10}" for lv in CURVE_LEVELS))
    for label, raw in [("HEFT-true (逐边真值)", heft_true_raw),
                        ("HEFT-mean (均值口径)", heft_mean_raw),
                        ("ADC-Meta (本方法)", ours_raw)]:
        row = " ".join(f"{np.mean(raw[lv]):>10.4f}" for lv in CURVE_LEVELS)
        print(f"  {label:<14} {row}")

    # 配对检验：HEFT-true vs HEFT-mean（隔离"口径"这一项本身有多大）
    # 以及 ADC-Meta vs HEFT-mean（换成均值口径之后，本方法是否还输）
    n_tests = 2 * len(CURVE_LEVELS)
    bonf_conf = 1.0 - 0.05 / n_tests
    print(f"\n配对差值（正数=前者更慢）；Bonferroni 校正 {n_tests} 次比较，"
          f"单次置信 {bonf_conf*100:.2f}%")
    print("-" * 78)
    analysis = {}
    for lv in CURVE_LEVELS:
        true_arr = np.asarray(heft_true_raw[lv], dtype=float)
        mean_arr = np.asarray(heft_mean_raw[lv], dtype=float)
        ours_arr = np.asarray(ours_raw[lv], dtype=float)

        d1 = true_arr - mean_arr          # 正数 = HEFT-true 更慢 = 均值口径反而更强
        lo1, hi1 = paired_bootstrap_ci(d1, conf=bonf_conf)
        v1 = ("真值口径更慢(Bonf显著)" if lo1 > 0 else
              "均值口径更慢(Bonf显著)" if hi1 < 0 else "分辨不出")

        d2 = mean_arr - ours_arr          # 正数 = 均值口径HEFT更慢 = 本方法此时更好
        lo2, hi2 = paired_bootstrap_ci(d2, conf=bonf_conf)
        v2 = ("本方法更优(Bonf显著)" if lo2 > 0 else
              "HEFT-mean更优(Bonf显著)" if hi2 < 0 else "分辨不出")

        print(f"  [{lv}]")
        print(f"    HEFT-true − HEFT-mean = {d1.mean():+.4f}s  "
              f"[{lo1:+.4f},{hi1:+.4f}]  {v1}")
        print(f"    HEFT-mean − ADC-Meta  = {d2.mean():+.4f}s  "
              f"[{lo2:+.4f},{hi2:+.4f}]  {v2}")
        analysis[lv] = {
            "heft_true_mean": float(true_arr.mean()),
            "heft_mean_mean": float(mean_arr.mean()),
            "ours_mean": float(ours_arr.mean()),
            "true_minus_mean": {"diff": float(d1.mean()), "ci_bonf": [lo1, hi1], "verdict": v1},
            "mean_minus_ours": {"diff": float(d2.mean()), "ci_bonf": [lo2, hi2], "verdict": v2},
        }

    out_path = "results/congestion_e3_heft_meanparam.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_test": n_test, "seed": seed,
            "heft_true_raw": heft_true_raw,
            "heft_mean_raw": heft_mean_raw,
            "ours_raw": ours_raw,
            "analysis": analysis,
        }, f, indent=2)
    print(f"\n结果已保存 → {out_path}")
    print("\n完成。")


if __name__ == "__main__":
    main()
