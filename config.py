"""
config.py — 超参数配置（异构多边缘版 / Heterogeneous Multi-Edge）
对应数学文档中所有符号的具体取值

【本版改动：v3 异构多边缘】
  * EnvConfig 新增 heterogeneous 开关与 f_edge_options / rate_ec_max_options
      - 异构时每个"元任务"独立采样 E 个边缘的算力与回程带宽
      - 取值集合的均值等于原同构值（f_edge=10e9, rate_ec_max=100.0），
        因此原同构设定是本设定的退化特例，数字可比
  * CCSMConfig 新增 alpha_edge：逐位置软掩码中"边缘间拥塞"的敏感度
      - alpha       → 云端权重（回程拥塞），沿用原语义
      - alpha_edge  → 边缘权重（各边缘计算排队相对差额），新增
      - alpha_edge=0 即退化为原"仅云端"单维先验；两者皆 0 即无 CCSM
  * ModelConfig 的特征维度改为由 Config.__post_init__ 统一派生（单一数据源）
      - task_feat_dim   = 7 + L        （7 基础特征 + 放置位置 one-hot）
      - global_feat_dim = 4E + 3
      - feat_dim        = task_feat_dim + global_feat_dim
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    # ── 多边缘设置 ────────────────────────────────
    num_edges: int = 3            # 边缘服务器数量 E（=1 时退化回原单边缘）

    # ── 异构设置 ──────────────────────────────────
    heterogeneous: bool = True    # False → 各边缘同构（回到 v2 行为）
    # 每个元任务从下列集合中为每个边缘独立采样（均值 = 原同构值）
    f_edge_options:      Tuple = (5e9, 10e9, 15e9)        # 均值 10e9
    rate_ec_max_options: Tuple = (50.0, 100.0, 150.0)     # 均值 100.0

    # ── 计算资源 ──────────────────────────────────
    f_local: float = 1e9          # 终端本地计算频率 (cycles/s)
    f_edge: float = 10e9          # 边缘参考算力（同构模式取值 / 归一化基准）
    f_cloud: float = 50e9         # 云服务器计算频率 (cycles/s)

    # ── 通信速率 (MB/s) ───────────────────────────
    rate_ue_options: Tuple = (3.0, 7.0, 11.0, 15.0, 19.0)
    rate_eu: float = 5.0          # 边缘→终端下行（固定，假设对称）

    rate_ec_max: float = 100.0    # 回程参考带宽（同构模式取值 / 归一化基准）
    rate_ce: float = 100.0        # 云→边缘下行 (MB/s)
    Q_ec_max: float = 50.0        # 单条回程链路最大队列容量 (MB)

    # ── 任务属性范围（归一化用，对齐 daggen_py 实际输出）──
    cpu_low: float = 1e7
    cpu_high: float = 2e8
    data_in_low: float = 0.005
    data_in_high: float = 0.05
    data_out_ratio: float = 0.5

    # ── 派生的归一化基准（__post_init__ 填充）──────
    f_edge_ref: float = 15e9          # = max(f_edge_options)
    rate_ec_ref: float = 150.0        # = max(rate_ec_max_options)


@dataclass
class DAGConfig:
    """daggen 参数体系（对齐 DRLTO / MRLCO 论文实验设置）"""
    n_options: Tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50)
    fat_options: Tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    density_options: Tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ccr_options: Tuple = (0.3, 0.4, 0.5)
    regular: float = 0.9
    mindata: float = 5000.0
    maxdata: float = 50000.0
    jump: int = 1


@dataclass
class ModelConfig:
    emb_dim: int = 64
    hidden_dim: int = 128
    n_locations: int = 3          # 由 Config.__post_init__ 派生为 E+2
    dropout: float = 0.1
    # ↓ 以下三项由 Config.__post_init__ 派生，勿手动设置
    task_feat_dim: int = 0        # 7 + L
    global_feat_dim: int = 0      # 4E + 3
    feat_dim: int = 0             # task_feat_dim + global_feat_dim


@dataclass
class CCSMConfig:
    """
    逐位置拥塞感知软掩码 (Congestion-Aware Soft Mask)

      w_local  = 1
      w_edge_e = exp( -alpha_edge · (avail_e − min_j avail_j) / T_ref )
      w_cloud  = exp( -alpha      · min_e Q_ec[e] / Q_ec_max )

    设计边界：软掩码仅使用与具体子任务无关的拥塞状态（排队相对差额、
    回程占用），不引入子任务大小做逐任务完成时刻估计——保持其"先验"
    定位，避免退化为启发式调度器。
    """
    alpha: float = 3.0            # 云端：回程拥塞敏感度（原语义不变）
    alpha_edge: float = 3.0       # 边缘：边缘间排队差额敏感度（新增）


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    batch_size: int = 64
    value_coeff: float = 0.5
    entropy_coeff: float = 0.05
    max_grad_norm: float = 0.5


@dataclass
class ADCConfig:
    """自适应双截断 (ADC) 超参数，对应数学文档 Section 5.5"""
    eps_out: float = 0.1
    eps_base: float = 0.2
    delta_lo: float = 0.5
    delta_hi: float = 2
    beta_g: float = 0.97
    eps_stab: float = 1e-8
    inner_lr: float = 0.003
    inner_steps: int = 5
    inner_episodes: int = 10


@dataclass
class MetaConfig:
    outer_lr: float = 3e-4
    meta_batch_size: int = 8
    meta_iterations: int = 2000
    eval_interval: int = 50
    eval_tasks: int = 20


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ccsm: CCSMConfig = field(default_factory=CCSMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    adc: ADCConfig = field(default_factory=ADCConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)

    seed: int = 42
    device: str = "cuda:1"
    log_dir: str = "logs/"
    ckpt_dir: str = "checkpoints/"

    def __post_init__(self):
        E = self.env.num_edges
        L = E + 2

        # 动作位置数 = 本地(1) + 边缘(E) + 云(1)
        self.model.n_locations = L

        # 归一化基准
        self.env.f_edge_ref  = float(max(self.env.f_edge_options))
        self.env.rate_ec_ref = float(max(self.env.rate_ec_max_options))

        # 观测维度（单一数据源，env 与 model 必须一致）
        #   task_features : 7 基础 + L 放置位置 one-hot
        #   global_features:
        #     Q_ec[e]/Q_max            → E
        #     avail_edge[e]/T_ref      → E
        #     f_edge[e]/f_edge_ref     → E
        #     rate_ec_max[e]/rate_ref  → E
        #     avail_local/T_ref        → 1
        #     avail_cloud/T_ref        → 1
        #     rate_ue 归一化           → 1
        self.model.task_feat_dim   = 7 + L
        self.model.global_feat_dim = 4 * E + 3
        self.model.feat_dim        = self.model.task_feat_dim + self.model.global_feat_dim
