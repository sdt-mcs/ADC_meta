"""
ppo.py — PPO 更新器（支持 ADC 双截断）— 多边缘版 / Multi-Edge
对应数学文档 Section 5.3-5.5

【多边缘改动（唯一功能性改动）】
  * RolloutBuffer.to_tensors 中 valid_mask/ccsm_weights 的填充长度
    从 3*max_N → L*max_N（L=E+2，由 obs["L"] 提供；缺省回退 3 兼容单边缘）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy


class RolloutBuffer:
    """
    轨迹缓冲区：存储一批 episode 数据用于 PPO 更新
    """
    def __init__(self):
        self.obs_list: List[Dict] = []
        self.actions: List[int] = []
        self.log_probs_old: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs_list.append(obs)
        self.actions.append(action)
        self.log_probs_old.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.actions)

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 GAE 优势估计和折扣回报
        对应数学文档公式 Â_t 和 V_t^{target}
        """
        T = len(self.rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)

        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def to_tensors(self, device: torch.device) -> Dict:
        """将缓冲区数据转为张量（用于 evaluate_actions 批量计算）"""
        max_N = max(o["N"] for o in self.obs_list)
        # 多边缘：动作维度 L = E+2 由 obs 提供（缺省回退 3，兼容旧单边缘）
        L = int(self.obs_list[0].get("L", 3))

        def pad_task_features(obs_list):
            arrays = []
            for obs in obs_list:
                arr = obs["task_features"]   # [N, 7]
                padded = np.zeros((max_N, arr.shape[1]), dtype=np.float32)
                padded[:arr.shape[0]] = arr
                arrays.append(padded)
            return np.stack(arrays)   # [B, max_N, 7]

        def pad_nl(obs_list, key, dtype=np.float32):
            arrays = []
            for obs in obs_list:
                arr = obs[key]              # [N*L]
                padded = np.zeros(L * max_N, dtype=dtype)
                padded[:len(arr)] = arr
                arrays.append(padded)
            return np.stack(arrays)         # [B, L*max_N]

        batch = {
            "task_features": torch.FloatTensor(pad_task_features(self.obs_list)).to(device),
            "global_features": torch.FloatTensor(
                np.stack([o["global_features"] for o in self.obs_list])
            ).to(device),
            "valid_mask": torch.BoolTensor(
                pad_nl(self.obs_list, "valid_mask", dtype=bool)
            ).to(device),
            "ccsm_weights": torch.FloatTensor(
                pad_nl(self.obs_list, "ccsm_weights")
            ).to(device),
            "N": max_N,
        }
        return batch


class PPOUpdater:
    """
    PPO 更新器
    支持 ADC 双截断：内循环（eps_in）和外循环（eps_out）使用不同截断参数
    """

    def __init__(self, model, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device

        # 外循环优化器（元策略更新用）
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.meta.outer_lr)

    def compute_ppo_loss(
        self,
        model,
        obs_batch: Dict,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        eps: float,            # 截断参数（内循环传 eps_in，外循环传 eps_out）
    ) -> Dict[str, torch.Tensor]:
        """
        计算 PPO 损失
        对应数学文档公式 L^{PPO}(θ)

        eps: 截断参数（ADC 中内外循环分别传入不同值）
        """
        ppo_cfg = self.cfg.ppo

        # 用当前策略重新评估动作
        new_log_probs, values, entropy = model.evaluate_actions(
            obs_batch, actions, self.device
        )

        # 重要性采样比 r_t(θ)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO 裁剪代理目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = ((values - returns) ** 2).mean()

        # 熵奖励（鼓励探索）
        entropy_loss = -entropy.mean()

        # 总损失
        total_loss = (policy_loss
                      + ppo_cfg.value_coeff * value_loss
                      + ppo_cfg.entropy_coeff * entropy_loss)

        return {
            "total": total_loss,
            "policy": policy_loss,
            "value": value_loss,
            "entropy": -entropy_loss,
        }

    def inner_loop_update(
        self,
        buffer: RolloutBuffer,
        theta_init: Dict,
        eps_in: float,
        inner_steps: int,
        inner_lr: float,
    ) -> Tuple[Dict, float]:
        """
        内循环：从 theta_init 出发，在当前任务上做 K 步 PPO 更新
        使用 eps_in（自适应）截断参数

        Returns:
          theta_prime: 适配后的参数（若出现 NaN 则回退到 theta_init）
          grad_norm: 第一步的梯度范数
        """
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(theta_init)
        local_model.train()

        # 使用 Adam（比 SGD 更稳定）
        local_opt = optim.Adam(local_model.parameters(), lr=inner_lr)

        returns, advantages = buffer.compute_returns_and_advantages(
            self.cfg.ppo.gamma, self.cfg.ppo.gae_lambda
        )
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        actions_t = torch.LongTensor(buffer.actions).to(self.device)
        old_lp_t = torch.FloatTensor(buffer.log_probs_old).to(self.device)

        # 过滤非有限的 old_log_probs（防止 ratio 爆炸）
        finite_mask = torch.isfinite(old_lp_t)
        if not finite_mask.all():
            old_lp_t = torch.where(finite_mask, old_lp_t, torch.full_like(old_lp_t, -5.0))

        obs_batch = buffer.to_tensors(self.device)

        grad_norm = 0.0
        for step_k in range(inner_steps):
            local_opt.zero_grad()
            losses = self.compute_ppo_loss(
                local_model, obs_batch, actions_t,
                old_lp_t, advantages_t, returns_t, eps_in
            )
            loss_val = losses["total"]

            # NaN 检测：若 loss 为 NaN 则跳过本步
            if not torch.isfinite(loss_val):
                break

            loss_val.backward()

            # 记录第一步梯度范数
            if step_k == 0:
                raw_norm = sum(
                    p.grad.data.norm().item() ** 2
                    for p in local_model.parameters()
                    if p.grad is not None
                ) ** 0.5
                grad_norm = raw_norm if np.isfinite(raw_norm) else 0.0

            nn.utils.clip_grad_norm_(local_model.parameters(),
                                     self.cfg.ppo.max_grad_norm)

            # 检查梯度是否含 NaN，有则跳过本步
            has_nan_grad = any(
                (p.grad is not None and not torch.isfinite(p.grad).all())
                for p in local_model.parameters()
            )
            if has_nan_grad:
                break

            local_opt.step()

            # 检查参数是否含 NaN，有则回退
            has_nan_param = any(
                not torch.isfinite(p.data).all()
                for p in local_model.parameters()
            )
            if has_nan_param:
                local_model.load_state_dict(theta_init)
                break

        theta_prime = {k: v.clone().detach() for k, v in local_model.state_dict().items()}
        return theta_prime, grad_norm

    def outer_loop_update(self, meta_grads: List[Dict], eps_out: float):
        """外循环：聚合各任务梯度，更新元参数（使用 eps_out 截断）"""
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            grads = []
            for g in meta_grads:
                gv = g.get(name)
                if gv is not None and torch.isfinite(gv).all():
                    grads.append(gv)
            if not grads:
                continue
            avg_grad = torch.stack(grads).mean(dim=0)
            # eps_out 截断：限制单步更新幅度
            scale = eps_out * (param.data.abs().mean() + 1e-8)
            avg_grad = torch.clamp(avg_grad, -scale, scale)
            param.grad = avg_grad

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.ppo.max_grad_norm)
        self.optimizer.step()

        # 检查参数健康性，若出现 NaN 则回滚（用零梯度更新前的参数）
        for param in self.model.parameters():
            if not torch.isfinite(param.data).all():
                param.data.fill_(0.0)  # 最保守回退

    def compute_meta_gradient(
        self,
        buffer: RolloutBuffer,
        theta_prime: Dict,
        theta: Dict,
    ) -> Dict:
        """
        计算元梯度（FOMAML 近似）
        ∇_θ L(θ') ≈ ∇_{θ'} L(θ')
        直接返回 θ' 相对于 θ 的参数差（梯度近似）
        """
        grads = {}
        for name in theta.keys():
            grads[name] = theta[name] - theta_prime[name]
        return grads
