"""
select_best_ckpt_v3.py — 固定测试集两阶段选最优 checkpoint（第二版，修复两个口径bug）

本版相对上一版的两处修复（都是我审出来的，之前那版结果不能用）：

  bug1（cfg不匹配）：上一版对全部六个配置统一用 cfg=Config() 默认值评估，
  但 no_ccsm/ccsm_cloud_only/homogeneous/no_meta 四个配置训练时是在改过的
  cfg 下练的，评估时不施加同样的改动，等于在错的环境/CCSM/内循环设置下
  评估——测出来的既不是Full也不是对应消融，是混合体。
  修复：新增必填 --ablation 参数，通过 ablation_cfg.apply_ablation()
  （与 main.py 训练时的分支逻辑保持唯一数据源）在构造 cfg 后立即施加。

  bug2（精选阶段学习率）：精选阶段 adapt_steps=5 时用的是
  cfg.adc.inner_lr=0.003（训练期外循环的学习率），但 adapt_diag.py 在固定
  测试集上验证过：这个学习率对纯推理期的内循环适配偏大，k>=3时改善会
  被抵消甚至变负（k=10时能到 +3.2%，即适配把策略变差了）；改用
  adapt_lr=3e-4 后 k=1~10 全程稳定获得约10%的显著改善（95%CI全部<0）。
  修复：精选阶段默认改用 --adapt_lr 3e-4（可覆盖），不再默认用
  cfg.adc.inner_lr。

为什么不用训练期的 [Eval] 数字：训练期评估复用同一个顺序推进的 RNG 流，
相同形状的训练配置会抽到相同的评估批次，存在"幸运批次"假象；且噪声极大
（相邻评估可差 75%）。这里用独立种子(seed=777，选择集；999留给最终报告)
的固定测试集重新评估。

两阶段设计（省时但不损失判据正确性）：
  阶段1 粗筛：全部 checkpoint，adapt_steps=0（不涉及学习率），n=n_coarse，
             只用于排序缩小候选集
  阶段2 精选：粗筛前 top_k 名，adapt_steps=K，adapt_lr=3e-4，n=n_fine，
             这是最终判据

复用 evaluate.py 里已有的 generate_test_dags / eval_model_on_dags，
不引入新的评估口径，避免与论文最终评估脚本不一致。

用法（在 ~/zjy/iotj/adc_meta_rl 目录下，ablation_cfg.py 需和本脚本放同一目录）：
    python -u select_best_ckpt_v3.py --ckpt_dir checkpoints/E3v3_none --ablation none
    python -u select_best_ckpt_v3.py --ckpt_dir checkpoints/E3v3_no_ccsm --ablation no_ccsm
    # 只筛后半段（早期 checkpoint 明显更差时可加，进一步省时）：
    python -u select_best_ckpt_v3.py --ckpt_dir checkpoints/E3v3_none --ablation none --min_iter 800
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np
import torch

from config import Config
from model import S2SPolicy
from evaluate import generate_test_dags, eval_model_on_dags

try:
    from ablation_cfg import apply_ablation, ABLATIONS
except ImportError:
    print("✗ 找不到 ablation_cfg.py，请确认它和本脚本放在同一目录下，未做任何评估。")
    sys.exit(1)


def iter_of(f):
    m = re.search(r"model_iter(\d+)\.pt", f)
    return int(m.group(1)) if m else -1


def load_model(ckpt_path, cfg):
    model = S2SPolicy(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--ablation", type=str, required=True, choices=ABLATIONS,
                   help="必须与训练该 checkpoint 时用的 --ablation 一致，"
                        "用于在评估前对 cfg 施加同样的改动（见 ablation_cfg.py）")
    p.add_argument("--n_coarse", type=int, default=50, help="阶段1粗筛的测试集大小")
    p.add_argument("--n_fine", type=int, default=100, help="阶段2精选的测试集大小")
    p.add_argument("--top_k", type=int, default=5, help="粗筛后进入精选的候选数")
    p.add_argument("--adapt_steps", type=int, default=-1,
                   help="精选阶段的内循环步数，-1 表示用（施加ablation后的）cfg.adc.inner_steps")
    p.add_argument("--adapt_lr", type=float, default=3e-4,
                   help="精选阶段内循环适配的学习率。默认3e-4，"
                        "已用 adapt_diag.py 在固定测试集上验证："
                        "cfg.adc.inner_lr=0.003 对纯推理期适配偏大，"
                        "k>=3时改善会被抵消甚至变负。传 0 则退回用 cfg.adc.inner_lr（不推荐）。")
    p.add_argument("--min_iter", type=int, default=0, help="只考虑 iter >= 此值的 checkpoint")
    p.add_argument("--seed", type=int, default=777,
                   help="测试集种子。777=选择集（默认，正确）；999留给论文最终报告，"
                        "不要在这里用999，否则构成选择集/报告集数据泄漏。")
    args = p.parse_args()

    if args.seed == 999:
        print("✗ --seed 999 是论文最终报告集，不能用于 checkpoint 选择（会造成数据泄漏），未做任何评估。")
        print("  如果确实要用999跑最终报告数字，请用 evaluate.py，不要用本脚本。")
        sys.exit(1)

    cfg = Config()
    apply_ablation(cfg, args.ablation)
    adapt_lr = None if args.adapt_lr == 0 else args.adapt_lr
    K = cfg.adc.inner_steps if args.adapt_steps < 0 else args.adapt_steps

    files = sorted(glob.glob(os.path.join(args.ckpt_dir, "model_iter*.pt")), key=iter_of)
    files = [f for f in files if iter_of(f) >= args.min_iter]
    if not files:
        print(f"✗ 未在 {args.ckpt_dir} 找到符合条件的 checkpoint")
        return

    print("=" * 72)
    print(f"ablation={args.ablation}  (ccsm.alpha={cfg.ccsm.alpha}, "
          f"ccsm.alpha_edge={cfg.ccsm.alpha_edge}, adc.delta_lo/hi="
          f"{cfg.adc.delta_lo}/{cfg.adc.delta_hi}, adc.inner_steps={cfg.adc.inner_steps}, "
          f"env.heterogeneous={cfg.env.heterogeneous})")
    print(f"精选阶段: adapt_steps={K}  adapt_lr={'cfg.adc.inner_lr=' + str(cfg.adc.inner_lr) if adapt_lr is None else adapt_lr}")
    print("=" * 72)

    print(f"阶段1 粗筛：{len(files)} 个 checkpoint，adapt_steps=0，n={args.n_coarse}")
    print("=" * 72)
    dags_coarse = generate_test_dags(cfg, n=args.n_coarse, seed=args.seed)

    coarse = []
    for f in files:
        model = load_model(f, cfg)
        ms = eval_model_on_dags(model, dags_coarse, cfg, adapt_steps=0)
        mean_ms = float(np.mean(ms))
        coarse.append((iter_of(f), f, mean_ms))
        print(f"  iter{iter_of(f):>5}: mean={mean_ms:.4f}s", flush=True)

    coarse.sort(key=lambda t: t[2])
    shortlist = coarse[:args.top_k]

    print("\n" + "=" * 72)
    print(f"阶段2 精选：粗筛前 {len(shortlist)} 名，adapt_steps={K}，"
          f"adapt_lr={'cfg默认' if adapt_lr is None else adapt_lr}，n={args.n_fine}")
    print("=" * 72)
    print(f"  候选：{[t[0] for t in shortlist]}")
    dags_fine = generate_test_dags(cfg, n=args.n_fine, seed=args.seed)

    fine = []
    for it, f, _ in shortlist:
        model = load_model(f, cfg)
        ms = eval_model_on_dags(model, dags_fine, cfg, adapt_steps=K, adapt_lr=adapt_lr)
        arr = np.array(ms)
        fine.append({
            "iter": it, "path": f,
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
        })
        print(f"  iter{it:>5}: mean={arr.mean():.4f}s  std={arr.std():.4f}s  "
              f"p50={np.percentile(arr, 50):.4f}s", flush=True)

    fine.sort(key=lambda d: d["mean"])
    best = fine[0]

    print("\n" + "=" * 72)
    print(f"最优 checkpoint: iter{best['iter']}  mean={best['mean']:.4f}s "
          f"(ablation={args.ablation}, adapt_steps={K}, "
          f"adapt_lr={'cfg默认' if adapt_lr is None else adapt_lr}, "
          f"n={args.n_fine}, seed={args.seed})")
    print(f"  路径: {best['path']}")
    print("=" * 72)

    os.makedirs("results", exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.ckpt_dir))
    out = f"results/ckpt_selection_{tag}.json"
    with open(out, "w") as fp:
        json.dump({
            "ckpt_dir": args.ckpt_dir, "ablation": args.ablation,
            "adapt_steps": K, "adapt_lr": adapt_lr if adapt_lr is not None else cfg.adc.inner_lr,
            "n_coarse": args.n_coarse, "n_fine": args.n_fine, "seed": args.seed,
            "coarse": [{"iter": t[0], "mean_adapt0": t[2]} for t in coarse],
            "fine": fine, "best": best,
        }, fp, indent=2)
    print(f"结果已保存 → {out}")


if __name__ == "__main__":
    main()
