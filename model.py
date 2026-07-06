"""
model.py — S2S 策略网络（含 CCSM 软掩码）
对应数学文档 Section 3-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class TaskEncoder(nn.Module):
    """
    子任务嵌入层：将原始特征映射到高维嵌入空间
    输入：[N, feat_dim] → 输出：[N, emb_dim]
    """
    def __init__(self, feat_dim: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # [N, emb_dim]


class S2SEncoder(nn.Module):
    """
    LSTM 编码器：对子任务嵌入序列编码
    对应数学文档 Section 4.1
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, emb_seq: torch.Tensor
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        emb_seq: [B, N, emb_dim]
        Returns:
          enc_outputs: [B, N, hidden_dim]  — 每步隐状态
          (h_n, c_n): 最终隐状态，用于初始化解码器
        """
        enc_outputs, (h_n, c_n) = self.lstm(emb_seq)
        return enc_outputs, (h_n, c_n)


class AttentionDecoder(nn.Module):
    """
    LSTM 解码器 + 注意力机制
    对应数学文档 Section 4.2
    """
    def __init__(self, emb_dim: int, hidden_dim: int, n_locations: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_locations = n_locations

        # 动作嵌入（用于解码器输入）
        self.action_emb = nn.Embedding(n_locations + 1, emb_dim)  # +1 for start token

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        # 注意力投影（对应公式中的 W_q, W_k）
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

        # 输出层：logits（用于 1D 联合索引动作）
        # 每个子任务 × n_locations 个动作
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, n_locations)  # 输出每个位置的 logit
        self.critic_head = nn.Linear(hidden_dim * 2, 1)       # 状态值函数

    def forward(
        self,
        enc_outputs: torch.Tensor,       # [B, N, H]
        h_state: Tuple,                  # LSTM 隐状态
        prev_action_emb: torch.Tensor,   # [B, 1, emb_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        单步解码
        Returns:
          task_logits: [B, N, n_locations]  — 每个子任务×每个位置的 logit
          value: [B, 1]                      — 状态值
          h_state: 更新后的 LSTM 状态
        """
        # LSTM 解码步
        dec_out, h_state = self.lstm(prev_action_emb, h_state)  # dec_out: [B, 1, H]

        # 注意力：解码器当前状态 对齐 编码器所有输出
        q = self.W_q(dec_out)              # [B, 1, H]
        k = self.W_k(enc_outputs)          # [B, N, H]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 1, N]
        attn_weights = F.softmax(scores, dim=-1)                # [B, 1, N]
        context = torch.bmm(attn_weights, enc_outputs)          # [B, 1, H]

        # 融合上下文
        fused = torch.cat([dec_out, context], dim=-1)           # [B, 1, 2H]
        feat = F.relu(self.W_o(fused))                          # [B, 1, H]

        # 每个子任务的位置 logits：扩展到 [B, N, n_locations]
        # 方式：用 feat 与每个子任务的 enc_output 再次计算位置得分
        # 简化版：将 feat 广播后与 enc_outputs 拼接，分别计算每个子任务的 logits
        feat_expanded = feat.expand(-1, enc_outputs.shape[1], -1)  # [B, N, H]
        task_feat = enc_outputs + feat_expanded                     # [B, N, H]（残差融合）
        task_logits = self.actor_head(task_feat)                    # [B, N, n_locs]

        # 状态值
        value = self.critic_head(fused.squeeze(1))                  # [B, 1]

        return task_logits, value, h_state


class S2SPolicy(nn.Module):
    """
    完整的 S2S 策略网络
    输入：环境观测字典
    输出：动作概率分布（经 CCSM 掩码后）+ 状态值
    """

    def __init__(self, cfg):
        super().__init__()
        feat_dim = 7 + 2    # 子任务特征(7) + 全局特征(2) 拼入每个子任务
        self.emb_dim = cfg.model.emb_dim
        self.hidden_dim = cfg.model.hidden_dim
        self.n_locations = cfg.model.n_locations

        self.task_encoder = TaskEncoder(feat_dim, self.emb_dim)
        self.s2s_encoder = S2SEncoder(self.emb_dim, self.hidden_dim)
        self.decoder = AttentionDecoder(self.emb_dim, self.hidden_dim, self.n_locations)

        # 开始 token 的动作嵌入（step=0 时使用）
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))

    def forward(
        self,
        obs: Dict,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: 单步观测字典（来自 env._get_obs()）
        Returns:
          log_probs: [3N]  — 对数概率（已掩码）
          value: scalar    — 状态值
        """
        task_feats = torch.FloatTensor(obs["task_features"]).to(device)  # [N, 7]
        global_feats = torch.FloatTensor(obs["global_features"]).to(device)  # [2]
        valid_mask = torch.BoolTensor(obs["valid_mask"]).to(device)      # [3N]
        ccsm_w = torch.FloatTensor(obs["ccsm_weights"]).to(device)       # [3N]
        N = obs["N"]

        # 将全局特征拼接到每个子任务特征
        global_expanded = global_feats.unsqueeze(0).expand(N, -1)       # [N, 2]
        full_feats = torch.cat([task_feats, global_expanded], dim=-1)   # [N, 9]

        # 编码
        emb = self.task_encoder(full_feats.unsqueeze(0))                 # [1, N, emb]
        enc_out, (h_n, c_n) = self.s2s_encoder(emb)                    # [1, N, H]

        # 解码（单步：当前调度决策）
        prev_emb = self.start_token                                       # [1, 1, emb]
        task_logits, value, _ = self.decoder(enc_out, (h_n, c_n), prev_emb)
        # task_logits: [1, N, n_locations]

        # 展平为 1D 联合索引 [3N]
        flat_logits = task_logits.squeeze(0).view(-1)                    # [3N]

        # ── CCSM 掩码（Section 3.2）──────────────────────────────────
        # 1. 软掩码：log(w_a) 加到 logits
        flat_logits = flat_logits + torch.log(ccsm_w + 1e-10)

        # 2. 硬掩码：非法动作设为 -inf
        flat_logits = flat_logits.masked_fill(~valid_mask, float("-inf"))

        # 3. Softmax → log_probs
        log_probs = F.log_softmax(flat_logits, dim=-1)                  # [3N]

        return log_probs, value.squeeze()

    def get_action(
        self,
        obs: Dict,
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        采样一个动作
        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            log_probs, value = self.forward(obs, device)
            valid_mask = torch.BoolTensor(obs["valid_mask"]).to(device)

            # 仅在合法动作上采样，保持与 forward 计算一致
            # 将 -inf 位置替换为极小值以便 softmax 数值稳定
            sample_logits = log_probs.clone()
            sample_logits[~valid_mask] = -1e9
            probs = torch.softmax(sample_logits, dim=-1)
            probs = probs * valid_mask.float()  # 确保非法动作概率为 0
            probs = probs / (probs.sum() + 1e-10)  # 重新归一化

            if deterministic:
                action = probs.argmax().item()
            else:
                action = torch.multinomial(probs, 1).item()

            # log_prob 与 forward 保持一致（从 log_probs 直接取）
            log_prob = log_probs[action].item()
            if not np.isfinite(log_prob):
                log_prob = -10.0  # fallback

        return action, log_prob, value.item()

    def evaluate_actions(
        self,
        obs_batch: Dict,
        actions: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量评估已采集动作的 log_prob、value、entropy
        用于 PPO 更新

        obs_batch: 批量观测（已预处理为张量）
        actions: [B]
        Returns: log_probs [B], values [B], entropy [B]
        """
        task_feats = obs_batch["task_features"].to(device)      # [B, N, 7]
        global_feats = obs_batch["global_features"].to(device)  # [B, 2]
        valid_masks = obs_batch["valid_mask"].to(device)        # [B, 3N]
        ccsm_ws = obs_batch["ccsm_weights"].to(device)          # [B, 3N]
        B, N, _ = task_feats.shape

        # 拼接全局特征
        global_exp = global_feats.unsqueeze(1).expand(-1, N, -1)         # [B, N, 2]
        full_feats = torch.cat([task_feats, global_exp], dim=-1)         # [B, N, 9]

        # 批量编码
        emb = self.task_encoder(full_feats)                               # [B, N, emb]
        enc_out, (h_n, c_n) = self.s2s_encoder(emb)

        # 批量解码
        start = self.start_token.expand(B, -1, -1)
        task_logits, values, _ = self.decoder(enc_out, (h_n, c_n), start)
        flat_logits = task_logits.view(B, -1)                            # [B, 3N]

        # CCSM 掩码
        flat_logits = flat_logits + torch.log(ccsm_ws + 1e-10)
        flat_logits = flat_logits.masked_fill(~valid_masks, float("-inf"))

        log_probs_all = F.log_softmax(flat_logits, dim=-1)               # [B, 3N]

        # 取对应动作的 log_prob
        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        # 策略熵：避免 0 * (-inf) = NaN 的 IEEE 陷阱
        # 方法：仅在合法动作上计算熵，屏蔽位置贡献为 0
        probs = log_probs_all.exp()                                      # -inf → 0
        # nan_to_num 处理残余 NaN（-inf * 0 可能产生 NaN）
        # 改后：用 where 彻底切断 masked 位置的梯度路径
        log_p = torch.where(valid_masks, log_probs_all, torch.zeros_like(log_probs_all))
        p = torch.where(valid_masks, probs, torch.zeros_like(probs))
        entropy = -(p * log_p).sum(dim=-1)                          # [B]

        return log_probs, values.squeeze(-1), entropy
