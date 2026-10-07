"""
paired_eval.py — 决定性实验：六个配置在报告集(seed=999)上的分层配对比较

这是整个 v3 异构多边缘改造要回答的核心问题：
修正了观测空间、探索坍缩、评估口径三个问题之后，
CCSM 和 ADC 在异构多边缘 MDP 下到底有没有稳健收益？

设计要点：

1. **报告集 seed=999**，与 checkpoint 选择用的 seed=777 完全分离，
   不存在"在报告集上挑模型"的泄漏。每个配置用的是它自己
   results/ckpt_selection_E3v3_*.json 里选出的最优 checkpoint。

2. **每个配置施加自己的 cfg**（通过 ablation_cfg.apply_ablation），
   内循环步数 K 也取该配置自己的 cfg.adc.inner_steps
   （no_meta 训练时 K=1，评估也用 K=1，与训练口径一致）。

3. **适配学习率统一用 3e-4**，不是 cfg.adc.inner_lr=0.003。
   依据是 adapt_diag.py 在 seed=777 上的结果：0.003 对纯推理期内循环偏大，
   k>=3 时改善被抵消甚至变负；3e-4 下 k=1~10 稳定获得约10%显著改善。

4. **按 DAG 配对**：每个 DAG 用固定的 numpy 种子(1000+i)和 torch 种子(20000+i)，
   所有配置、所有拥塞档位下完全相同。DAG 结构是 makespan 方差的主要来源，
   配对掉它之后，配置之间的差异才可能从噪声里分辨出来
   （非配对时 SE≈0.010，差异 0.001~0.009 完全淹没）。
   注意配对是配在 DAG 身份上，不是配在信道实现上（k>0 时适配 rollout 会
   额外消耗环境随机数），这会让检验略偏保守，不会制造假阳性。

5. **按回程拥塞分层**：CCSM 是拥塞机制，它的收益按设计就应该出现在
   高拥塞档位，而不是在"平均"上。只看总体均值可能把机制的收益稀释掉。
   拥塞档位沿用 eval_congestion.py 的定义（Q_ec 初始范围，单位MB）。

6. **homogeneous 不参与配对检验**：它跑在同构环境里（各边缘都是10GHz），
   makespan 不在同一个量纲上，和异构配置直接比没有意义。
   本脚本仍会报告它的绝对数值，但单列，不做差值检验。

用法（在 ~/zjy/iotj/adc_meta_rl 目录下）：
    python -u paired_eval.py
    python -u paired_eval.py --n_test 200          # 默认就是200
    python -u paired_eval.py --n_test 100 --levels default,high   # 想先快速看趋势
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import torch

from config import Config
from model import S2SPolicy
import evaluate as ev

try:
    from ablation_cfg import apply_ablation
except ImportError:
    print("✗ 找不到 ablation_cfg.py，请确认它和本脚本在同一目录，未做任何评估。")
    sys.exit(1)


ADAPT_EPISODES = 3
ADAPT_LR = 3e-4
PREFIX = "E3v3_"

# 沿用 eval_congestion.py 的拥塞档位定义（Q_ec 初始范围，MB）
LEVELS = {
    "default": None,
    "low": (0.0, 15.0),
    "medium": (15.0, 35.0),
    "high": (35.0, 50.0),
}

# 参与配对检验的配置（homogeneous 环境不同，单列报告不做检验）
BASE = "none"
COMPARED = ["ccsm_cloud_only", "no_ccsm", "fixed_eps", "no_meta"]
REPORT_ONLY = ["homogeneous"]


def paired_bootstrap_ci(d, conf=0.95, n_boot=10000, seed=12345):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo_q = (1.0 - conf) / 2.0 * 100
    hi_q = (1.0 + conf) / 2.0 * 100
    return float(np.percentile(means, lo_q)), float(np.percentile(means, hi_q))


def load_selection(tag):
    """读取该配置的 checkpoint 选择结果，并校验它是修复版脚本产出的"""
    path = f"results/ckpt_selection_{PREFIX}{tag}.json"
    if not os.path.isfile(path):
        print(f"✗ 找不到 {path}，请先用修复版 select_best_ckpt_v3.py 跑完该配置。")
        sys.exit(1)
    with open(path) as f:
        d = json.load(f)
    if "ablation" not in d:
        print(f"✗ {path} 里没有 'ablation' 字段，说明它是旧版选择脚本的产物"
              f"（旧版对所有配置统一用默认cfg，结果不可用）。请重跑该配置。")
        sys.exit(1)
    if d["ablation"] != tag:
        print(f"✗ {path} 记录的 ablation={d['ablation']} 与文件名标签 {tag} 不一致。")
        sys.exit(1)
    ckpt = d["best"]["path"]
    if not os.path.isfile(ckpt):
        print(f"✗ 找不到 checkpoint: {ckpt}")
        sys.exit(1)
    return d["ablation"], ckpt, d["best"]["iter"]


def load_model(ckpt_path, cfg):
    model = S2SPolicy(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def eval_one_dag(model, dag, cfg, K, q_range, i):
    """单DAG评估。种子只依赖DAG序号，保证跨配置、跨档位严格可配对且可复现。"""
    torch.manual_seed(20000 + i)
    ms = ev.eval_model_on_dags(
        model, [dag], cfg,
        adapt_steps=K,
        adapt_episodes=ADAPT_EPISODES,
        seed=1000 + i,
        adapt_lr=ADAPT_LR,
        q_ec_range=q_range,
    )
    return float(ms[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=999, help="报告集种子，默认999")
    p.add_argument("--levels", type=str, default="default,low,medium,high")
    p.add_argument("--out", type=str, default="results/paired_eval_E3v3.json")
    args = p.parse_args()

    level_names = [s.strip() for s in args.levels.split(",") if s.strip()]
    bad = [x for x in level_names if x not in LEVELS]
    if bad:
        print(f"✗ 未知拥塞档位 {bad}，可选：{list(LEVELS)}")
        sys.exit(1)

    sig = inspect.signature(ev.eval_model_on_dags)
    need = ["adapt_steps", "adapt_episodes", "seed", "adapt_lr", "q_ec_range"]
    missing = [x for x in need if x not in sig.parameters]
    if missing:
        print(f"✗ evaluate.eval_model_on_dags 缺少参数 {missing}，未做任何评估。")
        print(f"  实际签名: {sig}")
        sys.exit(1)

    all_tags = [BASE] + COMPARED + REPORT_ONLY

    # ── 先把六个配置的 cfg / K / checkpoint 都准备好并打印，便于核对 ──
    setups = {}
    print("=" * 92)
    print(f"配对评估  报告集 seed={args.seed}  n={args.n_test}  "
          f"adapt_lr={ADAPT_LR}  adapt_episodes={ADAPT_EPISODES}")
    print("=" * 92)
    for tag in all_tags:
        ablation, ckpt, it = load_selection(tag)
        cfg = Config()
        apply_ablation(cfg, ablation)
        K = cfg.adc.inner_steps
        setups[tag] = {"cfg": cfg, "K": K, "ckpt": ckpt, "iter": it}
        print(f"  {tag:<17} iter{it:<5} K={K}  "
              f"alpha={cfg.ccsm.alpha} alpha_edge={cfg.ccsm.alpha_edge} "
              f"delta={cfg.adc.delta_lo}/{cfg.adc.delta_hi} het={cfg.env.heterogeneous}")

    # DAG 只依赖 cfg.dag.*，任何消融都不改它，所以六个配置拿到的是同一批DAG
    dags = ev.generate_test_dags(setups[BASE]["cfg"], n=args.n_test, seed=args.seed)

    raw = {}
    for tag in all_tags:
        s = setups[tag]
        model = load_model(s["ckpt"], s["cfg"])
        raw[tag] = {}
        for lv in level_names:
            t0 = time.time()
            vals = [eval_one_dag(model, dag, s["cfg"], s["K"], LEVELS[lv], i)
                    for i, dag in enumerate(dags)]
            raw[tag][lv] = vals
            print(f"  {tag:<17} {lv:<8} mean={np.mean(vals):.4f}s  "
                  f"p50={np.percentile(vals, 50):.4f}s  ({time.time() - t0:.0f}s)", flush=True)

    # ── 绝对值总表 ────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("绝对 makespan 均值（秒）")
    print("=" * 92)
    print(f"  {'配置':<17} " + " ".join(f"{lv:>12}" for lv in level_names))
    for tag in all_tags:
        row = " ".join(f"{np.mean(raw[tag][lv]):>12.4f}" for lv in level_names)
        mark = "  (同构环境，不可与上面直接比)" if tag in REPORT_ONLY else ""
        print(f"  {tag:<17} {row}{mark}")

    # ── 配对差值检验 ──────────────────────────────────────────────
    n_tests = len(COMPARED) * len(level_names)
    bonf_conf = 1.0 - 0.05 / n_tests
    print("\n" + "=" * 92)
    print(f"相对 Full(none) 的配对差值（正数 = 该消融更差 = 被消融的机制有用）")
    print(f"共 {n_tests} 次比较，Bonferroni 校正后单次置信水平 {bonf_conf * 100:.2f}%")
    print("=" * 92)

    analysis = {}
    for lv in level_names:
        base = np.asarray(raw[BASE][lv], dtype=float)
        print(f"\n  [{lv}]  Full(none) = {base.mean():.4f}s")
        print(f"    {'消融':<17} {'差值':>9} {'95%CI':>21} {'Bonf.CI':>21} {'更差占比':>9}")
        for tag in COMPARED:
            arr = np.asarray(raw[tag][lv], dtype=float)
            d = arr - base
            lo95, hi95 = paired_bootstrap_ci(d, conf=0.95)
            lob, hib = paired_bootstrap_ci(d, conf=bonf_conf)
            worse = float((d > 0).mean())
            if lob > 0:
                verdict = "机制有用(Bonf显著)"
            elif hib < 0:
                verdict = "机制有害(Bonf显著)"
            elif lo95 > 0:
                verdict = "机制有用(仅95%)"
            elif hi95 < 0:
                verdict = "机制有害(仅95%)"
            else:
                verdict = "分辨不出"
            analysis[f"{lv}|{tag}"] = {
                "mean_diff": float(d.mean()),
                "ci95": [lo95, hi95], "ci_bonf": [lob, hib],
                "worse_rate": worse, "verdict": verdict,
                "mean_abs": float(arr.mean()), "base_abs": float(base.mean()),
            }
            print(f"    {tag:<17} {d.mean():>+9.4f} "
                  f"[{lo95:>+8.4f},{hi95:>+8.4f}] "
                  f"[{lob:>+8.4f},{hib:>+8.4f}] "
                  f"{worse * 100:>8.1f}%  {verdict}")

    print("\n  读法：差值为正 = 去掉该机制后变差 = 该机制确实有贡献。")
    print("        差值为负 = 去掉该机制反而更好 = 该机制在当前设定下没有帮助。")
    print("        置信区间跨过0 = 在本样本量下分辨不出差别。")

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "seed": args.seed, "n_test": args.n_test,
            "adapt_lr": ADAPT_LR, "adapt_episodes": ADAPT_EPISODES,
            "levels": {lv: LEVELS[lv] for lv in level_names},
            "setups": {t: {"ckpt": s["ckpt"], "iter": s["iter"], "K": s["K"]}
                       for t, s in setups.items()},
            "raw_makespans": raw,
            "paired_analysis": analysis,
            "n_tests": n_tests, "bonferroni_conf": bonf_conf,
        }, f, indent=2)
    print(f"\n结果已保存 → {args.out}")


if __name__ == "__main__":
    main()
