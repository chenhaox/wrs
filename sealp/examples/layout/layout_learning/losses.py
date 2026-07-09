"""复合损失。

L = L_cls + alpha * L_score + beta * L_xy + gamma * L_station  (+ kl_weight * KL)

- L_cls    : feasibility 分类 (BCEWithLogitsLoss)
- L_score  : layout_score 回归 (SmoothL1), 只在 feasible 样本上算
- L_xy     : xy proposal (SmoothL1), 只在"高分 feasible" 样本 & valid 零件上算
- L_station: 连续装配站 xy 回归 (SmoothL1), 只在高分 feasible 样本上算
             (若模型仍输出离散 region_logits, 则回退为 region 分类 CrossEntropy)
- KL       : CVAE 专用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as Fn


@dataclass
class LossWeights:
    alpha: float = 1.0     # score
    beta: float = 1.0      # xy proposal
    gamma: float = 0.5     # region
    kl: float = 0.01       # cvae kl
    pos_weight: float = 3.0  # 可行样本通常偏少, 给正类加权
    score_threshold: float = 0.0  # 只有 score >= 该阈值的 feasible 样本参与 proposal 监督


def compute_loss(out: Dict[str, torch.Tensor],
                 batch: Dict[str, torch.Tensor],
                 weights: LossWeights,
                 is_generator: bool) -> Dict[str, torch.Tensor]:
    device = out["feas_logit"].device
    feas = batch["feas"]
    score = batch["score"]

    # ---- feasibility ----
    pw = torch.tensor(weights.pos_weight, device=device)
    l_cls = Fn.binary_cross_entropy_with_logits(out["feas_logit"], feas, pos_weight=pw)

    # ---- score (仅 feasible) ----
    feas_mask = feas > 0.5
    if feas_mask.any():
        l_score = Fn.smooth_l1_loss(out["score_pred"][feas_mask], score[feas_mask])
    else:
        l_score = torch.zeros((), device=device)

    total = l_cls + weights.alpha * l_score
    logs = {"l_cls": l_cls.detach(), "l_score": l_score.detach()}

    # ---- proposal 相关 (仅生成式模型) ----
    if is_generator and "xy_pred" in out:
        # 高分 feasible 样本作为 proposal target
        good = feas_mask & (score >= weights.score_threshold)
        xy_pred = out["xy_pred"]
        xy_tgt = batch["xy_target"]
        valid = batch["xy_valid"] * good.float().unsqueeze(1)   # [B,N]
        denom = valid.sum().clamp_min(1.0)
        l_xy = (Fn.smooth_l1_loss(xy_pred, xy_tgt, reduction="none").sum(-1) * valid).sum() / denom
        total = total + weights.beta * l_xy
        logs["l_xy"] = l_xy.detach()

        # 连续装配站回归 (取代固定网格 region 分类)
        if "station_pred" in out and good.any():
            st_valid = batch.get("station_valid")
            gmask = good.float()
            if st_valid is not None:
                gmask = gmask * st_valid
            denom_s = gmask.sum().clamp_min(1.0)
            l_station = (Fn.smooth_l1_loss(out["station_pred"], batch["station_target"],
                                           reduction="none").sum(-1) * gmask).sum() / denom_s
            total = total + weights.gamma * l_station
            logs["l_station"] = l_station.detach()
        elif "region_logits" in out and good.any():
            # 兼容: 仍输出离散 region 的模型
            l_region = Fn.cross_entropy(out["region_logits"][good], batch["region_target"][good])
            total = total + weights.gamma * l_region
            logs["l_region"] = l_region.detach()

        if "kl" in out:
            total = total + weights.kl * out["kl"]
            logs["kl"] = out["kl"].detach()

    logs["total"] = total.detach()
    return {"loss": total, "logs": logs}
