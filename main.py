"""
main.py — 统一入口（多边缘版 / Multi-Edge）

用法：
  # 冒烟测试（含 3 轮真实 meta 训练）
  python main.py --mode smoke

  # 正式训练（E=3 主模型，2000 轮）
  python main.py --mode train --edges 3 --ablation none --iters 2000

  # 消融训练（各自独立 checkpoint 目录，互不覆盖）
  python main.py --mode train --edges 3 --ablation no_ccsm   --iters 2000
  python main.py --mode train --edges 3 --ablation fixed_eps --iters 2000
  python main.py --mode train --edges 3 --ablation no_meta   --iters 2000

  # E 扩展性（训练不同边缘数的主模型）
  python main.py --mode train --edges 2 --ablation none --iters 2000
  python main.py --mode train --edges 4 --ablation none --iters 2000
  python main.py --mode train --edges 5 --ablation none --iters 2000

每次训练存到独立目录：
  checkpoints/E{E}_{ablation}/model_iter{N}.pt
  logs/E{E}_{ablation}/training_log.json
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from daggen_py import DaggenPy
from env import ThreeTierEnv
from model import S2SPolicy
from meta_trainer import ADCMetaTrainer


def _loc_name(loc: int, E: int) -> str:
    if loc == 0:
        return "本地"
    if 1 <= loc <= E:
        return f"边缘{loc}"
    return "云端"


def smoke_test():
    print("=" * 55)
    print("冒烟测试：验证各组件（多边缘）")
    print("=" * 55)

    cfg = Config()
    rng = np.random.default_rng(42)
    gen = DaggenPy(seed=42)
    print(f"\n  num_edges E={cfg.env.num_edges}  n_locations L={cfg.model.n_locations}")

    # ── 1. DAG 生成 ───────────────────────────────────────
    print("\n[1] daggen_py DAG 生成")
    dag = gen.generate(n=20, fat=0.5, density=0.5, ccr=0.5)
    print(f"  {dag}")
    print(f"  入口:{[t.idx for t in dag.entry_tasks]}  "
          f"出口:{[t.idx for t in dag.exit_tasks]}")

    # ── 2. 环境 ──────────────────────────────────────────
    print("\n[2] 三层 MDP 环境")
    env = ThreeTierEnv(cfg, rng)
    obs = env.reset(dag)
    print(f"  task_features: {obs['task_features'].shape}  "
          f"global_features: {obs['global_features'].shape}（应=4E+3）")
    print(f"  Q_ec={np.round(env.state.Q_ec, 2)}MB  rate_ue={env.state.rate_ue}MB/s")

    done, total_r, steps = False, 0, 0
    model_tmp = S2SPolicy(cfg)
    while not done:
        a, lp, v = model_tmp.get_action(obs)
        obs, r, done, info = env.step(a)
        total_r += r; steps += 1
    print(f"  Episode: {steps}步  makespan={info['final_makespan']:.4f}s")

    # ── 3. CCSM 软掩码 ───────────────────────────────────
    print("\n[3] CCSM 软掩码")
    obs = env.reset(dag)
    w = obs["ccsm_weights"]
    L = env.L
    print(f"  Q_ec={np.round(env.state.Q_ec,2)}MB → 云端权重={w[env.LOC_CLOUD::L][0]:.4f}  "
          f"本地权重={w[0::L][0]:.4f}")
    print(f"  软掩码区间 [{w.min():.4f}, {w.max():.4f}]（应≤1.0 且>0）")

    # ── 4. 策略网络 ──────────────────────────────────────
    print("\n[4] S2S 策略网络")
    model = S2SPolicy(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,}")
    log_probs, value = model.forward(obs)
    assert not log_probs.isnan().any(), "前向传播产生 NaN！"
    print(f"  forward OK  value={value.item():.4f}")
    a, lp, v = model.get_action(obs)
    ti, loc = divmod(a, env.L)
    print(f"  采样动作: Task {ti} → {_loc_name(loc, env.E)}")

    # ── 5. ADC 自适应截断映射 ────────────────────────────
    print("\n[5] ADC ε_in 自适应映射")
    adc = cfg.adc
    g_ema = 1.0
    print(f"  ε_base={adc.eps_base}  ε_out={adc.eps_out}  δ=[{adc.delta_lo},{adc.delta_hi}]")
    for g_norm in [0.1, 0.5, 1.0, 2.0, 5.0]:
        g_ema = adc.beta_g * g_ema + (1 - adc.beta_g) * g_norm
        rho = g_norm / (g_ema + adc.eps_stab)
        eps_in = adc.eps_base * float(np.clip(rho, adc.delta_lo, adc.delta_hi))
        print(f"    g_norm={g_norm:.1f} → ε_in={eps_in:.4f}")

    # ── 6. 迷你训练（3轮外循环，真实跑一遍训练闭环）──────
    print("\n[6] ADC Meta-RL 迷你训练（3轮外循环）")
    cfg_mini = Config()
    cfg_mini.meta.meta_iterations = 3
    cfg_mini.meta.meta_batch_size = 2
    cfg_mini.meta.eval_interval = 999
    cfg_mini.adc.inner_steps = 2
    cfg_mini.adc.inner_episodes = 2
    cfg_mini.log_dir = "logs/smoke/"
    cfg_mini.ckpt_dir = "checkpoints/smoke/"
    trainer = ADCMetaTrainer(cfg_mini)
    log = trainer.meta_train()
    assert all(np.isfinite(v) for v in log["mean_makespan"]), "训练产生 NaN makespan！"
    print(f"  ε_in 轨迹: {[f'{v:.3f}' for v in log['eps_in']]}")

    print("\n" + "=" * 55)
    print("✅  所有冒烟测试通过！")
    print("=" * 55)


def full_train(args):
    import json
    cfg = Config()

    # ── 边缘数 ──
    cfg.env.num_edges = args.edges
    cfg.__post_init__()                      # 重新派生 n_locations = E+2

    # ── 消融配置 ──
    if args.ablation == "no_ccsm":
        cfg.ccsm.alpha = 0.0                 # 软掩码恒为 1
        cfg.ccsm.alpha_edge = 0.0
    elif args.ablation == "fixed_eps":
        cfg.adc.delta_lo = 1.0               # ε_in 固定 = ε_base
        cfg.adc.delta_hi = 1.0
    elif args.ablation == "no_meta":
        cfg.adc.inner_steps = 1              # 内循环单步 = 联合(pooled)PPO
    elif args.ablation == "ccsm_cloud_only":
        cfg.ccsm.alpha_edge = 0.0            # 仅云端一维先验（v2 行为）
    elif args.ablation == "homogeneous":
        cfg.env.heterogeneous = False        # 各边缘同构

    # ── CCSM alpha 覆盖（no_ccsm 分支已置 0，不覆盖它）──
    if getattr(args, "ccsm_alpha", None) is not None and args.ablation != "no_ccsm":
        cfg.ccsm.alpha = args.ccsm_alpha

    # ── 轮数 ──
    cfg.meta.meta_iterations = args.iters

    # ── 独立输出目录（防止不同配置互相覆盖 checkpoint）──
    tag = args.tag if getattr(args, "tag", None) else f"E{args.edges}_{args.ablation}"
    cfg.ckpt_dir = f"checkpoints/{tag}/"
    cfg.log_dir = f"logs/{tag}/"

    if torch.cuda.is_available():
        cfg.device = args.device

    print("=" * 55)
    print(f"训练配置: E={cfg.env.num_edges}  ablation={args.ablation}  "
          f"iters={cfg.meta.meta_iterations}")
    print(f"  n_locations={cfg.model.n_locations}  device={cfg.device}")
    print(f"  ckpt → {cfg.ckpt_dir}")
    print(f"  ccsm.alpha={cfg.ccsm.alpha}")
    print("=" * 55)

    if args.entropy_coeff is not None:
        cfg.ppo.entropy_coeff = args.entropy_coeff
    print(f"  entropy_coeff = {cfg.ppo.entropy_coeff}")
    trainer = ADCMetaTrainer(cfg)
    log = trainer.meta_train()

    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in log.items()}, f, indent=2)
    print(f"\n训练日志已保存 → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--edges", type=int, default=3,
                        help="边缘服务器数 E（默认 3）")
    parser.add_argument("--ablation",
                        choices=["none", "ccsm_cloud_only", "no_ccsm", "fixed_eps", "no_meta", "homogeneous"],
                        default="none", help="消融配置")
    parser.add_argument("--iters", type=int, default=2000,
                        help="外循环轮数（冒烟用小值，如 60）")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="训练设备")
    parser.add_argument("--ccsm_alpha", type=float, default=None,
                        help="覆盖 CCSM 拥塞敏感度 alpha（默认用 config.py 的 3.0）")
    parser.add_argument("--tag", type=str, default=None,
                        help="自定义输出目录标签，默认 E{edges}_{ablation}")
    parser.add_argument("--entropy_coeff", type=float, default=None,
                        help="覆盖 cfg.ppo.entropy_coeff（探索强度试点用，不传则保持config.py默认值）")
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    else:
        full_train(args)
