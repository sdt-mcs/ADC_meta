"""
ablation_cfg.py — 六个消融配置到 cfg 改动的唯一数据源

内容逐行照抄服务器 main.py 的分支逻辑（2026-09-11 用
`grep -n -A 6 "ablation" main.py` 核对过，对应第142-153行）：

    if args.ablation == "no_ccsm":
        cfg.ccsm.alpha = 0.0
        cfg.ccsm.alpha_edge = 0.0
    elif args.ablation == "fixed_eps":
        cfg.adc.delta_lo = 1.0
        cfg.adc.delta_hi = 1.0
    elif args.ablation == "no_meta":
        cfg.adc.inner_steps = 1
    elif args.ablation == "ccsm_cloud_only":
        cfg.ccsm.alpha_edge = 0.0
    elif args.ablation == "homogeneous":
        cfg.env.heterogeneous = False

任何评估/选择脚本都必须 import 这里的 apply_ablation()，不要各自维护一份
if/elif —— seed=999 第一轮 checkpoint 选择正是因为评估脚本对全部六个配置
统一用了 cfg=Config() 默认值（没做消融覆盖），导致 no_ccsm/ccsm_cloud_only/
homogeneous/no_meta 四个配置实际是在错的环境/CCSM/内循环设置下被评估的，
只有 none 和 fixed_eps 是干净的（fixed_eps 只影响训练期，评估期内循环本来
就用固定 eps_base，不受影响）。

⚠️ 如果以后 main.py 里这段分支逻辑改了，必须同步改这里，
否则本文件就变成新的不一致来源。
"""

ABLATIONS = ["none", "ccsm_cloud_only", "no_ccsm", "fixed_eps", "no_meta", "homogeneous"]


def apply_ablation(cfg, ablation: str):
    """原地修改 cfg，与 main.py 分支逻辑逐条对应。返回 cfg 本身，便于链式调用。"""
    if ablation not in ABLATIONS:
        raise ValueError(f"未知的 ablation='{ablation}'，必须是 {ABLATIONS} 之一")

    if ablation == "no_ccsm":
        cfg.ccsm.alpha = 0.0
        cfg.ccsm.alpha_edge = 0.0
    elif ablation == "fixed_eps":
        cfg.adc.delta_lo = 1.0
        cfg.adc.delta_hi = 1.0
    elif ablation == "no_meta":
        cfg.adc.inner_steps = 1
    elif ablation == "ccsm_cloud_only":
        cfg.ccsm.alpha_edge = 0.0
    elif ablation == "homogeneous":
        cfg.env.heterogeneous = False
    # ablation == "none": 不做任何修改，保持 Config() 默认值

    return cfg
