"""Inference: fixed goal → top-k init positions per bbox (variable N parts).

Input JSON ``parts[]``: each with extent, goal_pos, goal_rotmat, pose_candidates.
Optional ``init_candidates`` per part or ``table_x_range`` for auto grid.

Output: ``per_part_ranking[part_id]`` = top-k (init_pos, init_rotmat, pose_tag, score).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning import features as F
from layout_learning.models.part_placement_ranker import PartPlacementRankerNet
from layout_learning.part_placement_dataset import (
    MAX_CANDIDATES,
    POSE_FEAT_DIM,
    _candidate_feature,
    _global_vector,
    _part_static_vector,
    _table_bounds,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Part placement ranker inference (fixed goal → top-k init)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-json", required=True,
                   help='{"num_parts":N,"parts":[{part_id,extent,goal_pos,goal_rotmat,pose_candidates,...}],'
                          '"table_x_range":[...],"table_y_range":[...]}')
    p.add_argument("--candidates-json", default="",
                   help="Optional {part_id: [candidate dicts with xy, pose_tag, ...]}")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-json", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _default_candidates(sample: Dict, part: Dict) -> List[Dict]:
    pid = str(part["part_id"])
    blocks = sample.get("ranking_targets", []) or []
    for b in blocks:
        if str(b.get("part_id")) == pid:
            return list(b.get("candidates", []) or [])
    ranked = sample.get("init_candidates_ranked")
    if ranked and len(sample.get("parts", [])) == 1:
        return list(ranked)
    return []


def _grid_candidates(sample: Dict, spacing: float = 0.06) -> List[List[float]]:
    bounds = _table_bounds(sample)
    xlo, xhi, ylo, yhi = bounds
    step = max(float(spacing), 0.04)
    xs = np.arange(xlo, xhi + 1e-9, step)
    ys = np.arange(ylo, yhi + 1e-9, step)
    grid = [[float(x), float(y)] for y in ys for x in xs]
    return grid[:MAX_CANDIDATES]


def _rank_part(model, sample, part, candidates, device, top_k, init_hint=None):
    if not candidates:
        return []
    bounds = _table_bounds(sample)
    pid = str(part["part_id"])
    if init_hint is None:
        init_hint = part.get("init_pos_balanced") or (sample.get("init_pos_balanced") or {}).get(pid)

    n_c = min(len(candidates), MAX_CANDIDATES)
    cand_feat = np.zeros((1, MAX_CANDIDATES, POSE_FEAT_DIM), dtype=np.float32)
    cand_mask = np.zeros((1, MAX_CANDIDATES), dtype=np.float32)
    for i, c in enumerate(candidates[:MAX_CANDIDATES]):
        xy = c.get("xy") or (c.get("init_pos", [0, 0])[:2])
        cand_feat[0, i] = _candidate_feature(xy, init_hint, i, float(c.get("score", 0)), bounds)
        cand_mask[0, i] = 1.0

    num_parts = int(sample.get("num_parts", len(sample.get("parts", [part]))))
    part_static = _part_static_vector(part, sample, max(num_parts, 1))
    global_feat = _global_vector(sample)

    with torch.no_grad():
        out = model(
            torch.from_numpy(part_static).unsqueeze(0).to(device),
            torch.from_numpy(global_feat).unsqueeze(0).to(device),
            torch.from_numpy(cand_feat).to(device),
            torch.from_numpy(cand_mask).to(device),
        )
    scores = out["cand_scores"].squeeze(0).cpu().numpy()
    order = np.argsort(-scores[:n_c])
    k = min(int(top_k), n_c)

    ranked = []
    for rank, idx in enumerate(order[:k], start=1):
        c = dict(candidates[int(idx)])
        c["nn_score"] = float(scores[int(idx)])
        c["rank"] = rank
        if "init_pos" not in c and "xy" in c:
            z = float(sample.get("table_top_z", 0.0))
            c["init_pos"] = [c["xy"][0], c["xy"][1], z]
        ranked.append(c)
    return ranked


def main():
    args = _parse_args()
    with open(args.input_json, "r", encoding="utf-8") as stream:
        sample = json.load(stream)

    cands_map: Dict[str, List] = {}
    if args.candidates_json.strip():
        with open(args.candidates_json, "r", encoding="utf-8") as stream:
            cands_map = json.load(stream)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = PartPlacementRankerNet(
        hidden=int(ckpt.get("hidden", 128)),
        dropout=float(ckpt.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    if "table_x_range" not in sample:
        sample.setdefault("table_x_range", [0.06, 0.47])
        sample.setdefault("table_y_range", [-0.24, 0.24])
        sample.setdefault("table_top_z", 0.0)
    sample.setdefault("num_parts", len(sample.get("parts", [])))
    sample.setdefault("assembly_station_pos", sample.get("parts", [{}])[0].get("goal_pos", [0, 0, 0]))

    per_part: Dict[str, List[Dict]] = {}
    for part in sample.get("parts", []):
        pid = str(part["part_id"])
        if pid in cands_map:
            cands = cands_map[pid]
        else:
            cands = _default_candidates(sample, part)
        if not cands:
            grid = _grid_candidates(sample)
            pose_tags = part.get("pose_candidates", []) or [{"pose_tag": "identity"}]
            cands = []
            for xy in grid:
                for pc in pose_tags[:4]:
                    cands.append({
                        "xy": xy,
                        "pose_tag": pc.get("pose_tag", "identity"),
                        "rot_name": pc.get("rot_name", "identity"),
                        "score": 0.0,
                    })
        per_part[pid] = _rank_part(model, sample, part, cands, args.device, args.top_k)
        print(f"[rank] {pid}: top-{len(per_part[pid])} (goal fixed)")

    result = {
        "num_parts": len(sample.get("parts", [])),
        "top_k": int(args.top_k),
        "per_part_ranking": per_part,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
