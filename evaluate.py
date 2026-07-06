"""
evaluate.py — 评估脚本（消融实验 + 对比基线）

用法（PyCharm 中直接运行此文件）：
  python evaluate.py --mode ablation   # 消融实验
  python evaluate.py --mode compare    # 与基线对比
  python evaluate.py --mode adapt      # 快速适应曲线

结果自动保存到 results/ 目录，供 plot_results.py 读取。
"""
import argparse
import json
import os
import copy
import numpy as np
import torch
from typing import List, Dict

from config import Config
from daggen_py import DaggenPy
from env import ThreeTierEnv
from model import S2SPolicy
from meta_trainer import ADCMetaTrainer
from baseline import run_all_baselines
from ppo import PPOUpdater, RolloutBuffer


# ─────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg: Config) -> S2SPolicy:
    """从 checkpoint 加载模型"""
    model = S2SPolicy(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  已加载模型: {ckpt_path}（iter={ckpt.get('iteration', '?')}）")
    return model


def generate_test_dags(cfg: Config, n: int = 100, seed: int = 999) -> list:
    """生成固定测试集（与训练集不重叠，seed=999 保证每次一致）"""
    gen = DaggenPy(seed=seed)
    return gen.generate_batch(
        n,
        n_options=cfg.dag.n_options,
        fat_options=cfg.dag.fat_options,
        density_options=cfg.dag.density_options,
        ccr_options=cfg.dag.ccr_options,
        regular=cfg.dag.regular,
        mindata=cfg.dag.mindata,
        maxdata=cfg.dag.maxdata,
        jump=cfg.dag.jump,
    )


def eval_model_on_dags(
    model: S2SPolicy,
    dags: list,
    cfg: Config,
    adapt_steps: int = 0,
    adapt_episodes: int = 3,
    seed: int = 0,
    adapt_lr: float = None,
    q_ec_range: tuple = None,
) -> List[float]:
    """
    用模型在一批 DAG 上评估，返回每个 DAG 的 makespan 列表
    adapt_steps=0 表示直接推理（不做内循环适配）
    adapt_steps>0 表示先做 adapt_steps 步内循环适配再推理
    """
    rng = np.random.default_rng(seed)
    env = ThreeTierEnv(cfg, rng)
    device = torch.device("cpu")
    updater = PPOUpdater(model, cfg, device)
    makespans = []

    theta_meta = {k: v.clone() for k, v in model.state_dict().items()}

    for dag in dags:
        theta = copy.deepcopy(theta_meta)

        # 内循环适配（若 adapt_steps > 0）
        if adapt_steps > 0:
            buf = RolloutBuffer()
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(theta)
            model.eval()
            for _ in range(adapt_episodes):
                obs = env.reset(dag, q_ec_range=q_ec_range)
                done = False
                while not done:
                    a, lp, v = model.get_action(obs, device)
                    nobs, r, done, _ = env.step(a)
                    buf.add(obs, a, lp, r, v, done)
                    obs = nobs
            model.load_state_dict(saved)
            _lr = adapt_lr if adapt_lr is not None else cfg.adc.inner_lr
            theta, _ = updater.inner_loop_update(
                buf, theta,
                eps_in=cfg.adc.eps_base,
                inner_steps=adapt_steps,
                inner_lr=_lr,
            )

        # 贪心推理
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(theta)
        model.eval()

        obs = env.reset(dag, q_ec_range=q_ec_range)
        done = False
        while not done:
            a, _, _ = model.get_action(obs, device, deterministic=True)
            obs, _, done, info = env.step(a)
        makespans.append(info.get("final_makespan", 0.0))

        model.load_state_dict(saved)

    return makespans


def stats(values: List[float]) -> Dict:
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# ─────────────────────────────────────────────────────────
#  消融实验
# ─────────────────────────────────────────────────────────

def run_ablation(cfg: Config, ckpt_path: str, n_test: int = 100):
    print("\n" + "=" * 60)
    print("消融实验")
    print("=" * 60)

    test_dags = generate_test_dags(cfg, n=n_test)
    os.makedirs("results", exist_ok=True)
    results = {}

    print("\n[基准] Full ADC Meta-RL")
    model_full = load_model(ckpt_path, cfg)
    ms_full = eval_model_on_dags(model_full, test_dags, cfg,
                                  adapt_steps=cfg.adc.inner_steps)
    results["Full_ADC_MetaRL"] = stats(ms_full)
    print(f"  mean={results['Full_ADC_MetaRL']['mean']:.3f}s")

    print("\n[消融A] w/o CCSM（软掩码权重恒为1.0）")
    cfg_no_ccsm = copy.deepcopy(cfg)
    cfg_no_ccsm.ccsm.alpha = 0.0
    model_no_ccsm = load_model(ckpt_path, cfg_no_ccsm)
    ms_no_ccsm = eval_model_on_dags(model_no_ccsm, test_dags, cfg_no_ccsm,
                                     adapt_steps=cfg.adc.inner_steps)
    results["w/o_CCSM"] = stats(ms_no_ccsm)
    print(f"  mean={results['w/o_CCSM']['mean']:.3f}s  "
          f"退化: +{results['w/o_CCSM']['mean']-results['Full_ADC_MetaRL']['mean']:.3f}s")

    print("\n[消融B1] Fixed eps_in（δ_lo=δ_hi=1.0，eps_in 恒等于 eps_base）")
    cfg_fixed = copy.deepcopy(cfg)
    cfg_fixed.adc.delta_lo = 1.0
    cfg_fixed.adc.delta_hi = 1.0
    model_fixed = load_model(ckpt_path, cfg_fixed)
    ms_fixed = eval_model_on_dags(model_fixed, test_dags, cfg_fixed,
                                   adapt_steps=cfg.adc.inner_steps)
    results["Fixed_eps_in"] = stats(ms_fixed)
    print(f"  mean={results['Fixed_eps_in']['mean']:.3f}s  "
          f"退化: +{results['Fixed_eps_in']['mean']-results['Full_ADC_MetaRL']['mean']:.3f}s")

    print("\n[消融B2] Single eps（内外循环共享 eps_out=0.1）")
    cfg_single = copy.deepcopy(cfg)
    cfg_single.adc.eps_base = cfg.adc.eps_out
    cfg_single.adc.delta_lo = 1.0
    cfg_single.adc.delta_hi = 1.0
    model_single = load_model(ckpt_path, cfg_single)
    ms_single = eval_model_on_dags(model_single, test_dags, cfg_single,
                                    adapt_steps=cfg.adc.inner_steps)
    results["Single_eps"] = stats(ms_single)
    print(f"  mean={results['Single_eps']['mean']:.3f}s  "
          f"退化: +{results['Single_eps']['mean']-results['Full_ADC_MetaRL']['mean']:.3f}s")

    print("\n[消融C] No Adaptation（adapt_steps=0，直接推理）")
    model_no_adapt = load_model(ckpt_path, cfg)
    ms_no_adapt = eval_model_on_dags(model_no_adapt, test_dags, cfg, adapt_steps=0)
    results["No_Adaptation"] = stats(ms_no_adapt)
    print(f"  mean={results['No_Adaptation']['mean']:.3f}s  "
          f"退化: +{results['No_Adaptation']['mean']-results['Full_ADC_MetaRL']['mean']:.3f}s")

    out_path = "results/ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n消融结果已保存 → {out_path}")
    return results


# ─────────────────────────────────────────────────────────
#  基线对比
# ─────────────────────────────────────────────────────────

def run_comparison(cfg: Config, ckpt_path: str, n_test: int = 100):
    print("\n" + "=" * 60)
    print("基线对比实验")
    print("=" * 60)

    test_dags = generate_test_dags(cfg, n=n_test)
    os.makedirs("results", exist_ok=True)
    results = {}

    print("\n[基线算法]")
    baseline_results = run_all_baselines(test_dags, cfg)
    for r in baseline_results:
        results[r["name"]] = {k: v for k, v in r.items() if k != "makespans"}

    print("\n[本工作] ADC Meta-RL")
    model = load_model(ckpt_path, cfg)
    #ms = eval_model_on_dags(model, test_dags, cfg, adapt_steps=cfg.adc.inner_steps)
    ms = eval_model_on_dags(model, test_dags, cfg, adapt_steps=0)
    results["ADC_MetaRL"] = stats(ms)
    print(f"  mean={results['ADC_MetaRL']['mean']:.3f}s")

    print(f"\n{'方法':<16} {'Mean(s)':<10} {'Std(s)':<10} {'Min(s)':<10} Max(s)")
    print("-" * 55)
    for name, r in results.items():
        print(f"  {name:<14} {r['mean']:<10.3f} {r['std']:<10.3f} "
              f"{r['min']:<10.3f} {r['max']:.3f}")

    out_path = "results/comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n对比结果已保存 → {out_path}")
    return results


# ─────────────────────────────────────────────────────────
#  快速适应曲线
# ─────────────────────────────────────────────────────────

def run_adaptation_curve(cfg: Config, ckpt_path: str, n_test: int = 50):
    print("\n" + "=" * 60)
    print("快速适应曲线（adapt_steps vs makespan）")
    print("=" * 60)

    test_dags = generate_test_dags(cfg, n=n_test)
    os.makedirs("results", exist_ok=True)
    results = {"adapt_steps": [], "ADC_MetaRL": [], "RandomInit": []}

    model_meta = load_model(ckpt_path, cfg)
    torch.manual_seed(42)
    model_random = S2SPolicy(cfg)
    model_random.eval()

    for k in [0, 1, 2, 3, 5, 10, 20]:
        print(f"  adapt_steps={k} ...", end=" ", flush=True)
        ms_meta = eval_model_on_dags(model_meta, test_dags, cfg,
                                      adapt_steps=k, adapt_episodes=3,
                                      adapt_lr=0.001)
        ms_rand = eval_model_on_dags(model_random, test_dags, cfg,
                                      adapt_steps=k, adapt_episodes=3)
        results["adapt_steps"].append(k)
        results["ADC_MetaRL"].append(float(np.mean(ms_meta)))
        results["RandomInit"].append(float(np.mean(ms_rand)))
        print(f"ADC={np.mean(ms_meta):.3f}s  Random={np.mean(ms_rand):.3f}s")

    out_path = "results/adaptation_curve.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n适应曲线已保存 → {out_path}")
    return results


# ─────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADC Meta-RL 评估")
    parser.add_argument("--mode", choices=["ablation", "compare", "adapt", "all"],
                        default="all", help="评估模式")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model_iter500.pt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--n_test", type=int, default=100,
                        help="测试 DAG 数量")
    args = parser.parse_args()

    cfg = Config()

    if not os.path.exists(args.ckpt):
        print(f"⚠️  未找到 checkpoint: {args.ckpt}")
        print("请先运行 main.py --mode train 完成训练")
        os.makedirs("checkpoints", exist_ok=True)
        model = S2SPolicy(cfg)
        torch.save({"model_state": model.state_dict(), "iteration": 0}, args.ckpt)
        print(f"[演示] 已用随机初始化模型保存到 {args.ckpt}，继续运行流程验证...")

    if args.mode in ("ablation", "all"):
        run_ablation(cfg, args.ckpt, args.n_test)

    if args.mode in ("compare", "all"):
        run_comparison(cfg, args.ckpt, args.n_test)

    if args.mode in ("adapt", "all"):
        run_adaptation_curve(cfg, args.ckpt, args.n_test)

    print("\n✅ 评估完成，结果在 results/ 目录")
