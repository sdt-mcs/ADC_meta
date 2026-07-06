"""
eval_by_n.py  ──  按不同任务规模 N 分别评估 ADC 和 HEFT
对应 MRLCO 论文 Fig.7 的对标实验

用法：
    python eval_by_n.py
    python eval_by_n.py --ckpt checkpoints/model_iter1750.pt --n_test 30
"""
import argparse
import json
import os
import numpy as np

from config import Config
from daggen_py import DaggenPy
from baseline import run_all_baselines
from evaluate import load_model, eval_model_on_dags


def eval_by_n(ckpt_path: str, n_test: int = 30):
    cfg = Config()

    # ── 配置安全检查（防止用错误的config跑）─────────────
    print("=" * 60)
    print("配置确认")
    print("=" * 60)
    print(f"  ccsm.alpha     = {cfg.ccsm.alpha}   (应为 3.0)")
    print(f"  adc.delta_lo   = {cfg.adc.delta_lo}  (应为 0.5)")
    print(f"  adc.delta_hi   = {cfg.adc.delta_hi}  (应为 2.0)")
    print(f"  adc.inner_steps= {cfg.adc.inner_steps}  (应为 5)")
    assert abs(cfg.ccsm.alpha - 3.0) < 0.01,  "错误：ccsm.alpha 不是3.0，请先修复config.py"
    assert abs(cfg.adc.delta_lo - 0.5) < 0.01, "错误：delta_lo 不是0.5，请先修复config.py"
    assert abs(cfg.adc.delta_hi - 2.0) < 0.01, "错误：delta_hi 不是2.0，请先修复config.py"
    print("  ✅ 配置正确\n")

    # ── 加载主模型 ───────────────────────────────────────
    model = load_model(ckpt_path, cfg)

    # ── 按 N 值逐一评估 ──────────────────────────────────
    n_values = [10, 20, 30, 40, 50]
    results  = {"n_values": n_values, "HEFT": [], "ADC_K0": [], "ADC_K2": []}

    print(f"\n{'N':>4}   {'HEFT':>8}   {'ADC K=0':>8}   "
          f"{'ADC K=2':>8}   {'K0 vs HEFT':>11}   {'K2 vs HEFT':>11}")
    print("-" * 68)

    for n in n_values:
        # 生成固定N的测试集（每个N用独立种子，保证可复现）
        gen      = DaggenPy(seed=999 + n)
        test_dags = gen.generate_batch(
            n_test,
            n_options       = (n,),                      # 关键：固定任务数为 n
            fat_options     = cfg.dag.fat_options,
            density_options = cfg.dag.density_options,
            ccr_options     = cfg.dag.ccr_options,
            regular         = cfg.dag.regular,
            mindata         = cfg.dag.mindata,
            maxdata         = cfg.dag.maxdata,
            jump            = cfg.dag.jump,
        )

        # HEFT 基线
        baseline_list = run_all_baselines(test_dags, cfg)
        heft_mean = next(r["mean"] for r in baseline_list if r["name"] == "HEFT")

        # ADC K=0（零适配直接推理）
        ms_k0  = eval_model_on_dags(model, test_dags, cfg,
                                     adapt_steps=0, adapt_episodes=3,
                                     seed=0, adapt_lr=0.001)
        mean_k0 = float(np.mean(ms_k0))

        # ADC K=2（2步适配后推理）
        ms_k2  = eval_model_on_dags(model, test_dags, cfg,
                                     adapt_steps=2, adapt_episodes=3,
                                     seed=0, adapt_lr=0.001)
        mean_k2 = float(np.mean(ms_k2))

        results["HEFT"].append(round(heft_mean, 4))
        results["ADC_K0"].append(round(mean_k0, 4))
        results["ADC_K2"].append(round(mean_k2, 4))

        print(f"N={n:2d}   {heft_mean:8.3f}s   {mean_k0:8.3f}s   "
              f"{mean_k2:8.3f}s   "
              f"{(mean_k0/heft_mean-1)*100:+10.1f}%   "
              f"{(mean_k2/heft_mean-1)*100:+10.1f}%")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    out = "results/eval_by_n.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存 → {out}")
    print("✅ 完成")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=str, default="checkpoints/model_iter1750.pt",
                        help="主模型checkpoint路径")
    parser.add_argument("--n_test", type=int, default=30,
                        help="每个N值测试的DAG数量（建议30）")
    args = parser.parse_args()
    eval_by_n(args.ckpt, args.n_test)
