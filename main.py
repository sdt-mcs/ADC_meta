"""
main.py — 统一入口

用法（PyCharm 中 Parameters 栏填写）：
  --mode smoke   冒烟测试，约30秒，验证所有模块正常
  --mode train   正式训练，约30-60分钟(CPU)
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


def smoke_test():
    print("=" * 55)
    print("冒烟测试：验证各组件")
    print("=" * 55)

    cfg = Config()
    rng = np.random.default_rng(42)
    gen = DaggenPy(seed=42)

    # ── 1. DAG 生成 ───────────────────────────────────────
    print("\n[1] daggen_py DAG 生成")
    dag = gen.generate(n=20, fat=0.5, density=0.5, ccr=0.5)
    print(f"  {dag}")
    print(f"  gen_params: fat={dag.gen_params['fat']} "
          f"density={dag.gen_params['density']} ccr={dag.gen_params['ccr']}")
    print(f"  入口:{[t.idx for t in dag.entry_tasks]}  "
          f"出口:{[t.idx for t in dag.exit_tasks]}")
    print(f"  cpu范围: [{min(t.cpu for t in dag.tasks):.2e}, "
          f"{max(t.cpu for t in dag.tasks):.2e}] cycles")
    print(f"  data_in范围: [{min(t.data_in for t in dag.tasks)*1000:.2f}, "
          f"{max(t.data_in for t in dag.tasks)*1000:.2f}] KB")

    # ── 2. 环境 ──────────────────────────────────────────
    print("\n[2] 三层 MDP 环境")
    env = ThreeTierEnv(cfg, rng)
    obs = env.reset(dag)
    print(f"  task_features: {obs['task_features'].shape}  "
          f"（归一化范围正常: max={obs['task_features'].max():.3f}）")
    print(f"  Q_ec={env.state.Q_ec:.2f}MB  rate_ue={env.state.rate_ue}MB/s")

    done, total_r, steps = False, 0, 0
    while not done:
        valid = np.where(obs["valid_mask"])[0]
        a, lp, v = S2SPolicy(cfg).get_action(obs)
        obs, r, done, info = env.step(a)
        total_r += r; steps += 1
    print(f"  Episode: {steps}步  makespan={info['final_makespan']:.4f}s")

    # ── 3. CCSM 软掩码 ───────────────────────────────────
    print("\n[3] CCSM 软掩码")
    obs = env.reset(dag)
    w = obs["ccsm_weights"]
    print(f"  Q_ec={env.state.Q_ec:.2f}MB → 云端权重={w[2::3][0]:.4f}  本地权重={w[0::3][0]:.4f}")
    print(f"  软掩码区间 [{w.min():.4f}, {w.max():.4f}]（应< 1.0 且> 0）")

    # ── 4. 策略网络 ──────────────────────────────────────
    print("\n[4] S2S 策略网络")
    model = S2SPolicy(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,}")
    log_probs, value = model.forward(obs)
    valid_lp = log_probs[obs["valid_mask"]]
    assert not log_probs.isnan().any(), "前向传播产生 NaN！"
    print(f"  forward OK  value={value.item():.4f}  合法动作 log_prob 范围: "
          f"[{valid_lp.min().item():.3f}, {valid_lp.max().item():.3f}]")
    a, lp, v = model.get_action(obs)
    ti, loc = divmod(a, 3)
    print(f"  采样动作: Task {ti} → {'本地/边缘/云端'.split('/')[loc]}")

    # ── 5. ADC 自适应截断映射 ────────────────────────────
    print("\n[5] ADC ε_in 自适应映射")
    adc = cfg.adc
    g_ema = 1.0
    print(f"  ε_base={adc.eps_base}  ε_out={adc.eps_out}  "
          f"δ=[{adc.delta_lo},{adc.delta_hi}]")
    for g_norm in [0.1, 0.5, 1.0, 2.0, 5.0]:
        g_ema = adc.beta_g * g_ema + (1 - adc.beta_g) * g_norm
        rho   = g_norm / (g_ema + adc.eps_stab)
        eps_in = adc.eps_base * float(np.clip(rho, adc.delta_lo, adc.delta_hi))
        print(f"    g_norm={g_norm:.1f} → ε_in={eps_in:.4f}")

    # ── 6. 批量生成（论文标准参数）──────────────────────
    print("\n[6] 批量生成（论文标准参数）")
    dags = gen.generate_batch(20)
    ns   = [d.N for d in dags]
    fats = [d.gen_params["fat"] for d in dags]
    ccrs = [d.gen_params["ccr"] for d in dags]
    print(f"  N: min={min(ns)} max={max(ns)} mean={np.mean(ns):.1f}")
    print(f"  fat 分布: {sorted(set(fats))}")
    print(f"  ccr 分布: {sorted(set(ccrs))}")

    # ── 7. 迷你训练（3轮外循环）─────────────────────────
    print("\n[7] ADC Meta-RL 迷你训练（3轮外循环）")
    cfg_mini = Config()
    cfg_mini.meta.meta_iterations = 3
    cfg_mini.meta.meta_batch_size = 2
    cfg_mini.meta.eval_interval   = 999
    cfg_mini.adc.inner_steps      = 2
    cfg_mini.adc.inner_episodes   = 2
    trainer = ADCMetaTrainer(cfg_mini)
    log = trainer.meta_train()
    assert all(np.isfinite(v) for v in log["mean_makespan"]), "训练产生 NaN makespan！"
    print(f"  ε_in 轨迹: {[f'{v:.3f}' for v in log['eps_in']]}")

    print("\n" + "=" * 55)
    print("✅  所有冒烟测试通过！")
    print("=" * 55)


def full_train():
    import json
    cfg = Config()
    # 如需调整，在此修改 cfg 字段
    # cfg.meta.meta_iterations = 500
    # cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = ADCMetaTrainer(cfg)
    log = trainer.meta_train()

    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in log.items()}, f, indent=2)
    print(f"\n训练日志已保存 → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke",
                        help="smoke=冒烟测试  train=正式训练")
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    else:
        full_train()
