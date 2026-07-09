"""统一推理接口。

把训练好的 checkpoint 包装成 ``LayoutModelRunner``, 提供两条推理路径:

    - scorer 模型 : ``score_layouts(candidates)`` 对一批候选 layout 打分/排序,
                    供搜索器"采样 -> NN 预筛 -> evaluate_layout"。
    - generator   : ``propose_layouts(cond, k)`` 直接生成 k 个候选 layout xy,
                    供搜索器"NN 生成 -> evaluate_layout"。

关键: 本模块只输出**候选 xy / 排序**, 绝不判定最终可行性。最终仍由调用方的
evaluate_layout + motion-level validation 决定。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from . import features as F
from .dataset import sample_to_item, collate_items, move_batch
from .models import build_model, is_generator


class LayoutModelRunner:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = ckpt["model_name"]
        self.max_parts = int(ckpt.get("max_parts", F.MAX_PARTS_DEFAULT))
        self.feature_version = str(ckpt.get("feature_version", "v1"))
        self.is_generator = bool(ckpt.get("is_generator", is_generator(self.model_name)))
        self.model = build_model(self.model_name, flat_dim=ckpt["flat_dim"],
                                 **ckpt.get("model_kwargs", {})).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # ---------- 单样本 -> batch ----------
    def _batch_from_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        item = sample_to_item(sample, self.max_parts, self.feature_version)
        return move_batch(collate_items([item]), self.device)

    # ---------- scorer 路径 ----------
    @torch.no_grad()
    def score_layouts(self, samples: List[Dict]) -> Dict[str, np.ndarray]:
        """对一批含 staging_xy 的 sample 打分。

        返回 {"feas_prob":[M], "score":[M]}。
        """
        items = [sample_to_item(s, self.max_parts, self.feature_version) for s in samples]
        batch = move_batch(collate_items(items), self.device)
        out = self.model(batch)
        prob = torch.sigmoid(out["feas_logit"]).cpu().numpy()
        score = out["score_pred"].cpu().numpy()
        return {"feas_prob": prob, "score": score}

    # ---------- generator 路径 ----------
    @torch.no_grad()
    def propose_layouts(self, cond: Dict, k: int,
                        part_ids: Optional[List[str]] = None,
                        station_xy: Optional[np.ndarray] = None
                        ) -> List[Dict[str, np.ndarray]]:
        """给定条件 sample, 生成 k 个候选 layout。

        模型输出的是"相对装配站的偏移", 绝对 staging = 装配站 + 偏移:
            - station_xy 显式给定时直接使用 (scorer 采样站位 / CVAE / Diffusion);
            - 否则若模型能回归站位 (SAGPN) 则用 predict_station;
            - 再否则退回 cond 里的 assembly_station_pos。

        返回 [{pid: np.array([x,y]) (table 绝对坐标), ...}, ...]
        """
        batch = self._batch_from_sample(cond)
        off_norm = self.model.propose(batch, k).cpu().numpy()   # [k, N, 2] 偏移(归一化)
        bounds = F._table_bounds(cond)

        if station_xy is None:
            station_xy = self.predict_station(cond)
        if station_xy is None:
            station_xy = np.asarray(cond.get("assembly_station_pos", [0.0, 0.0, 0.0]),
                                    dtype=np.float32)[:2]
        station_xy = np.asarray(station_xy, dtype=np.float32)[:2]

        parts = F._ordered_parts(cond)
        pids = part_ids or [p["part_id"] for p in parts]
        out: List[Dict[str, np.ndarray]] = []
        for kk in range(off_norm.shape[0]):
            layout: Dict[str, np.ndarray] = {}
            for i, pid in enumerate(pids):
                if i >= off_norm.shape[1]:
                    break
                layout[pid] = station_xy + F.denormalize_offset(off_norm[kk, i], bounds)
            out.append(layout)
        return out

    @torch.no_grad()
    def predict_station(self, cond: Dict) -> Optional[np.ndarray]:
        """SAGPN: 回归连续装配站 xy (桌面绝对坐标 [x, y]); 其它模型返回 None。"""
        if not hasattr(self.model, "predict_station"):
            return None
        batch = self._batch_from_sample(cond)
        st_norm = self.model.predict_station(batch).cpu().numpy()[0]
        return F.denormalize_xy(st_norm, F._table_bounds(cond))
