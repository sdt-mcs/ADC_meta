"""
meta_trainer.py — ADC Meta-RL 训练主循环
对应数学文档 Section 5.5-5.6（Algorithm 完整实现）

修改记录：
  v2.0  增加详细计时统计（每轮耗时、剩余时间预测、各阶段占比）
        修复 generate_task_distribution → generate_batch
        修复 Makespan 单位显示（ms → s）
"""
import time
import torch
import numpy as np
import copy
import os
from typing import List, Dict, Tuple

from config import Config
from daggen_py import DaggenPy, DaggenDAG as DAG
from env import ThreeTierEnv
from model import S2SPolicy
from ppo import PPOUpdater, RolloutBuffer


class ADCMetaTrainer:
    """
    自适应双截断元强化学习训练器

    核心循环：
      外循环：采样 B 个任务场景 → 对每个场景运行内循环 → 聚合元梯度更新 θ
      内循环：从元参数出发，在当前任务上做 K 步自适应截断 PPO 更新

    ADC 自适应截断：
      eps_in(t) = eps_base · clip(||∇L|| / ḡ, δ_lo, δ_hi)
      其中 ḡ 为梯度范数的 EMA
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.rng = np.random.default_rng(cfg.seed)

        # 初始化组件
        self.dag_gen = DaggenPy(seed=cfg.seed)
        self.env     = ThreeTierEnv(cfg, self.rng)
        self.model   = S2SPolicy(cfg).to(self.device)
        self.updater = PPOUpdater(self.model, cfg, self.device)

        # ADC 自适应状态：梯度范数的 EMA ḡ
        self.g_ema = 1.0

        # 训练日志
        self.log: Dict[str, List] = {
            "outer_iter"    : [],
            "mean_makespan" : [],
            "eps_in"        : [],
            "g_ema"         : [],
            "meta_grad_norm": [],
        }

        os.makedirs(cfg.log_dir,  exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    #  外循环（Meta-Training）
    # ──────────────────────────────────────────────

    def meta_train(self):
        """外循环主训练流程，含详细计时统计"""

        print("=" * 65)
        print("ADC Meta-RL 训练开始")
        print(f"  外循环迭代次数: {self.cfg.meta.meta_iterations}")
        print(f"  每轮任务批大小: {self.cfg.meta.meta_batch_size}")
        print(f"  内循环步数 K:   {self.cfg.adc.inner_steps}")
        print(f"  ε_out (外循环): {self.cfg.adc.eps_out}")
        print(f"  ε_base (内循环基础): {self.cfg.adc.eps_base}")
        print(f"  计算设备:       {self.cfg.device}")
        print("=" * 65)

        # ── 计时器初始化 ──────────────────────────────────────
        time_total_start = time.time()
        iter_times       = []          # 每轮外循环耗时（秒）
        phase_times      = {           # 各阶段累计耗时
            "collect"   : 0.0,         # 轨迹采集
            "grad_norm" : 0.0,         # 梯度范数估算
            "inner_loop": 0.0,         # 内循环 K 步更新
            "outer_loop": 0.0,         # 外循环元梯度聚合
        }

        for outer_iter in range(1, self.cfg.meta.meta_iterations + 1):
            # 每1000轮衰减学习率
            if outer_iter % 1000 == 0 and outer_iter > 0:
                for pg in self.updater.optimizer.param_groups:
                    pg['lr'] *= 0.7
                print(f"  [LR Decay] outer_lr → {self.updater.optimizer.param_groups[0]['lr']:.2e}")
            iter_start = time.time()

            # ① 采样一批任务场景
            task_dags = self.dag_gen.generate_batch(
                self.cfg.meta.meta_batch_size,
                n_options       = self.cfg.dag.n_options,
                fat_options     = self.cfg.dag.fat_options,
                density_options = self.cfg.dag.density_options,
                ccr_options     = self.cfg.dag.ccr_options,
                regular         = self.cfg.dag.regular,
                mindata         = self.cfg.dag.mindata,
                maxdata         = self.cfg.dag.maxdata,
                jump            = self.cfg.dag.jump,
            )

            meta_grads     = []
            task_makespans = []
            eps_in_values  = []
            theta = {k: v.clone() for k, v in self.model.state_dict().items()}

            for dag in task_dags:
                # ② 内循环（含计时）
                theta_prime, eps_in, makespan = self._inner_loop(
                    dag, theta, phase_times
                )

                # ③ 计算元梯度（FOMAML：参数差）
                meta_grad = self.updater.compute_meta_gradient(
                    buffer=None,
                    theta_prime=theta_prime,
                    theta=theta,
                )
                meta_grads.append(meta_grad)
                task_makespans.append(makespan)
                eps_in_values.append(eps_in)

            # ④ 外循环：聚合梯度更新元参数（计时）
            t0 = time.time()
            self.updater.outer_loop_update(meta_grads, self.cfg.adc.eps_out)
            phase_times["outer_loop"] += time.time() - t0

            # ── 本轮耗时与剩余时间预测 ───────────────────────
            iter_elapsed  = time.time() - iter_start
            iter_times.append(iter_elapsed)
            total_elapsed = time.time() - time_total_start
            recent_avg    = np.mean(iter_times[-10:])
            remaining     = recent_avg * (self.cfg.meta.meta_iterations - outer_iter)

            # ── 记录日志 ─────────────────────────────────────
            mean_ms  = np.mean(task_makespans)
            mean_eps = np.mean(eps_in_values)
            meta_grad_norm = np.mean([
                sum(v.norm().item() ** 2 for v in g.values()) ** 0.5
                for g in meta_grads
            ])

            self.log["outer_iter"]    .append(outer_iter)
            self.log["mean_makespan"] .append(mean_ms)
            self.log["eps_in"]        .append(mean_eps)
            self.log["g_ema"]         .append(self.g_ema)
            self.log["meta_grad_norm"].append(meta_grad_norm)

            # ── 每10轮打印进度 ────────────────────────────────
            if outer_iter % 10 == 0:
                print(
                    f"Iter {outer_iter:4d} | "
                    f"Makespan: {mean_ms:.3f}s | "
                    f"ε_in: {mean_eps:.3f} | "
                    f"ḡ: {self.g_ema:.4f} | "
                    f"本轮: {iter_elapsed:.1f}s | "
                    f"已用: {self._fmt_time(total_elapsed)} | "
                    f"剩余: {self._fmt_time(remaining)}"
                )

            # ── 定期评估 ─────────────────────────────────────
            if outer_iter % self.cfg.meta.eval_interval == 0:
                t0 = time.time()
                eval_ms   = self.evaluate(n_tasks=self.cfg.meta.eval_tasks)
                eval_time = time.time() - t0
                print(
                    f"  [Eval] Iter {outer_iter} | "
                    f"Avg Makespan: {eval_ms:.3f}s | "
                    f"评估耗时: {eval_time:.1f}s"
                )
                self._save_checkpoint(outer_iter)

        # ── 训练完成汇总 ──────────────────────────────────────
        total_time = time.time() - time_total_start
        print("\n" + "=" * 65)
        print("训练完成！耗时汇总")
        print("=" * 65)
        print(f"  总训练时间:     {self._fmt_time(total_time)}")
        print(f"  平均每轮耗时:   {np.mean(iter_times):.2f}s")
        print(f"  最快轮次耗时:   {min(iter_times):.2f}s")
        print(f"  最慢轮次耗时:   {max(iter_times):.2f}s")
        print(f"\n  各阶段累计耗时占比：")
        for phase, t in phase_times.items():
            pct = t / total_time * 100 if total_time > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {phase:<12}: {self._fmt_time(t):>12}  ({pct:5.1f}%)  {bar}")
        print("=" * 65)

        return self.log

    # ──────────────────────────────────────────────
    #  内循环（Task-Specific Adaptation）
    # ──────────────────────────────────────────────

    def _inner_loop(
        self,
        dag        : DAG,
        theta      : Dict,
        phase_times: Dict,
    ) -> Tuple[Dict, float, float]:
        """
        内循环：在单个任务 DAG 上进行快速适配（含各阶段计时）

        Returns:
          theta_prime : 适配后的参数
          eps_in      : 本次使用的内循环截断参数
          makespan    : 本轮平均 makespan
        """
        adc = self.cfg.adc

        # ① 轨迹采集（计时）
        t0 = time.time()
        buffer = self._collect_rollout(dag, theta, n_episodes=adc.inner_episodes)
        phase_times["collect"] += time.time() - t0

        # ② 梯度范数估算（计时）
        t0 = time.time()
        g_norm = self._estimate_grad_norm(buffer, theta)
        phase_times["grad_norm"] += time.time() - t0

        # ③ 更新 EMA：ḡ ← β_g·ḡ + (1-β_g)·g_norm
        self.g_ema = adc.beta_g * self.g_ema + (1 - adc.beta_g) * g_norm
        self.g_ema = min(self.g_ema, 0.8)
        # ④ 计算自适应截断 ε_in
        rho         = g_norm / (self.g_ema + adc.eps_stab)
        rho_clipped = float(np.clip(rho, adc.delta_lo, adc.delta_hi))
        eps_in      = adc.eps_base * rho_clipped

        # ⑤ 内循环 K 步 PPO 更新（计时）
        t0 = time.time()
        theta_prime, _ = self.updater.inner_loop_update(
            buffer     = buffer,
            theta_init = theta,
            eps_in     = eps_in,
            inner_steps= adc.inner_steps,
            inner_lr   = adc.inner_lr,
        )
        phase_times["inner_loop"] += time.time() - t0

        makespan = -float(np.sum(buffer.rewards)) / max(1, adc.inner_episodes)
        return theta_prime, eps_in, makespan

    def _estimate_grad_norm(self, buffer: RolloutBuffer, theta: Dict) -> float:
        """估算当前任务的梯度范数（只做前向+反向，不更新参数）"""
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(theta)
        local_model.train()

        returns, advantages = buffer.compute_returns_and_advantages(
            self.cfg.ppo.gamma, self.cfg.ppo.gae_lambda
        )
        returns_t    = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        actions_t    = torch.LongTensor(buffer.actions).to(self.device)
        old_lp_t     = torch.FloatTensor(buffer.log_probs_old).to(self.device)
        obs_batch    = buffer.to_tensors(self.device)

        finite_mask = torch.isfinite(old_lp_t)
        if not finite_mask.all():
            old_lp_t = torch.where(
                finite_mask, old_lp_t, torch.full_like(old_lp_t, -5.0)
            )

        losses = self.updater.compute_ppo_loss(
            local_model, obs_batch, actions_t,
            old_lp_t, advantages_t, returns_t,
            eps=self.cfg.adc.eps_base,
        )
        if not torch.isfinite(losses["total"]):
            return 1.0

        losses["total"].backward()

        g_norm = sum(
            p.grad.data.norm().item() ** 2
            for p in local_model.parameters()
            if p.grad is not None and torch.isfinite(p.grad).all()
        ) ** 0.5

        return float(g_norm) if np.isfinite(g_norm) else 1.0

    # ──────────────────────────────────────────────
    #  轨迹采集
    # ──────────────────────────────────────────────

    def _collect_rollout(
        self,
        dag       : DAG,
        theta     : Dict,
        n_episodes: int = 10,
    ) -> RolloutBuffer:
        """用参数 theta 在 dag 上采集 n_episodes 条轨迹"""
        saved_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(theta)
        self.model.eval()

        buffer = RolloutBuffer()
        for _ in range(n_episodes):
            obs  = self.env.reset(dag)
            done = False
            while not done:
                action, log_prob, value = self.model.get_action(obs, self.device)
                next_obs, reward, done, _ = self.env.step(action)
                buffer.add(obs, action, log_prob, reward, value, done)
                obs = next_obs

        self.model.load_state_dict(saved_state)
        return buffer

    # ──────────────────────────────────────────────
    #  评估 & 存档
    # ──────────────────────────────────────────────

    def evaluate(self, n_tasks: int = 20, adaptation_steps: int = None) -> float:
        """评估元策略：内循环 K 步适配 → 贪心推理，返回平均 makespan"""
        if adaptation_steps is None:
            adaptation_steps = self.cfg.adc.inner_steps

        test_dags = self.dag_gen.generate_batch(
            n_tasks,
            n_options       = self.cfg.dag.n_options,
            fat_options     = self.cfg.dag.fat_options,
            density_options = self.cfg.dag.density_options,
            ccr_options     = self.cfg.dag.ccr_options,
            regular         = self.cfg.dag.regular,
            mindata         = self.cfg.dag.mindata,
            maxdata         = self.cfg.dag.maxdata,
            jump            = self.cfg.dag.jump,
        )
        theta     = {k: v.clone() for k, v in self.model.state_dict().items()}
        makespans = []

        for dag in test_dags:
            buffer = self._collect_rollout(dag, theta, n_episodes=3)
            theta_prime, _ = self.updater.inner_loop_update(
                buffer, theta,
                eps_in     = self.cfg.adc.eps_base,
                inner_steps= adaptation_steps,
                inner_lr   = self.cfg.adc.inner_lr,
            )

            saved = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(theta_prime)
            self.model.eval()

            obs  = self.env.reset(dag)
            done = False
            while not done:
                action, _, _ = self.model.get_action(
                    obs, self.device, deterministic=True
                )
                obs, _, done, info = self.env.step(action)

            makespans.append(info.get("final_makespan", 0.0))
            self.model.load_state_dict(saved)

        return float(np.mean(makespans))

    def _save_checkpoint(self, iteration: int):
        path = os.path.join(self.cfg.ckpt_dir, f"model_iter{iteration}.pt")
        torch.save({
            "iteration"      : iteration,
            "model_state"    : self.model.state_dict(),
            "optimizer_state": self.updater.optimizer.state_dict(),
            "g_ema"          : self.g_ema,
            "log"            : self.log,
        }, path)
        print(f"  [Checkpoint] 已保存 → {path}")

    # ──────────────────────────────────────────────
    #  工具函数
    # ──────────────────────────────────────────────

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """将秒数格式化为 Xh XXm XXs"""
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"
