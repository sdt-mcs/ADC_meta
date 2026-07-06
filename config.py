"""
config.py — 超参数配置
对应数学文档中所有符号的具体取值
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    # ── 计算资源 ──────────────────────────────────
    f_local: float = 1e9          # 终端本地计算频率 (cycles/s)
    f_edge: float = 10e9          # 边缘服务器计算频率 (cycles/s)
    f_cloud: float = 50e9         # 云服务器计算频率 (cycles/s)

    # ── 通信速率 (MB/s) ───────────────────────────
    # 终端-边缘上行/下行（随信道质量变化）
    rate_ue_options: Tuple = (3.0, 7.0, 11.0, 15.0, 19.0)
    rate_eu: float = 5.0          # 边缘→终端下行（固定，假设对称）

    # 边缘-云回程链路
    rate_ec_max: float = 100.0    # 最大边-云速率 (MB/s)
    rate_ce: float = 100.0        # 云→边缘下行 (MB/s)
    Q_ec_max: float = 50.0        # 回程链路最大队列容量 (MB)

    # ── 任务属性范围 ──────────────────────────────
    # ── 归一化用范围（对齐 daggen_py 实际输出）────────────
    # daggen 参数：mindata=5000B, maxdata=50000B, ccr∈{0.3,0.4,0.5}
    # 实测：cpu ∈ [1e7, 1.67e8]，data_in ∈ [0.005, 0.05] MB
    cpu_low: float = 1e7           # 子任务 CPU 需求下界 (cycles)
    cpu_high: float = 2e8          # 子任务 CPU 需求上界 (cycles)
    data_in_low: float = 0.005     # 输入数据下界 (MB) = 5 KB
    data_in_high: float = 0.05     # 输入数据上界 (MB) = 50 KB
    data_out_ratio: float = 0.5   # 输出数据 = 输入数据 × ratio


@dataclass
class DAGConfig:
    """
    daggen 参数体系（对齐 DRLTO / MRLCO 论文实验设置）
    参数含义详见 daggen_py.py 文件头注释
    """
    n_options: Tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50)     # 任务数选项
    fat_options: Tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)         # DAG宽度选项
    density_options: Tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)     # 依赖密度选项
    ccr_options: Tuple = (0.3, 0.4, 0.5)                         # 通信-计算比选项
    regular: float = 0.9        # 各层任务数均匀度（论文默认值）
    mindata: float = 5000.0     # 数据量下界（字节）= 5 KB
    maxdata: float = 50000.0    # 数据量上界（字节）= 50 KB
    jump: int = 1               # 仅连接相邻层


@dataclass
class ModelConfig:
    emb_dim: int = 64              # 子任务嵌入维度
    hidden_dim: int = 128          # LSTM 隐状态维度
    n_locations: int = 3           # 卸载位置数（本地/边缘/云端）
    dropout: float = 0.1


@dataclass
class CCSMConfig:
    alpha: float = 3.0             # 拥塞敏感度 α
    # 软掩码权重 w = exp(-α · Q_ec / Q_max)
    # α=3 时：Q_ec=0 → w=1.0；Q_ec=Q_max → w≈0.05


@dataclass
class PPOConfig:
    gamma: float = 0.99            # 折扣因子 γ
    gae_lambda: float = 0.95       # GAE 参数 λ
    ppo_epochs: int = 4            # 每次更新的 epoch 数
    batch_size: int = 64           # mini-batch 大小
    value_coeff: float = 0.5       # 价值损失系数 c1
    entropy_coeff: float = 0.01    # 熵正则系数 c2
    max_grad_norm: float = 0.5     # 梯度裁剪范数


@dataclass
class ADCConfig:
    """
    自适应双截断 (ADC) 超参数
    对应数学文档 Section 5.5
    """
    # 外循环（元策略更新，固定保守值）
    eps_out: float = 0.1           # ε_out

    # 内循环（场景适配，自适应）
    eps_base: float = 0.2          # ε_base（内循环基础截断）
    delta_lo: float = 0.5          # δ_lo（自适应下界）
    delta_hi: float = 2          # δ_hi（自适应上界）
    beta_g: float = 0.97           # EMA 衰减系数 β_g
    eps_stab: float = 1e-8         # 数值稳定项

    # 内循环优化
    inner_lr: float = 0.003        # α（内循环学习率，Adam 用低 LR）
    inner_steps: int = 5           # K（内循环梯度步数）
    inner_episodes: int = 10       # 内循环每任务采集轨迹数


@dataclass
class MetaConfig:
    # 外循环优化
    outer_lr: float = 3e-4         # β（外循环学习率）
    meta_batch_size: int = 8       # B（每次外循环采样任务数）
    meta_iterations: int = 2000      # 外循环总迭代次数
    eval_interval: int = 50        # 每隔多少次外循环评估一次
    eval_tasks: int = 20           # 评估时使用的任务数


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
