#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shelf-unit STL 局部坐标与 fixture 基座标下的装配 goal 位姿。

``side_panel.stl`` 局部 AABB（底面 z=0）：
  x ±0.045, y ±0.009, z ∈ [0, 0.300]

``shelf.stl`` 局部 AABB（**平放板**，几何中心在原点，与 ``gen_meshes`` 一致）：
  x ±0.045, y ±0.075, z ±0.0075  （薄轴 ≈ Z）

Staging：层板用 ``shelf_staging_rotmat()`` → 平放板 Rx(+90°) **竖立**贴桌取料。
Goal：层板 ``SHELF_GOAL_ROTMAT`` 水平插入（local 薄轴 → world +Z）。
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
SHELF_STL = os.path.join(_ASSET_DIR, "shelf.stl")
SIDE_STL = os.path.join(_ASSET_DIR, "side_panel.stl")

TABLE_TOP_Z = 0.0
FIXTURE_POS = np.array([0.35, -0.31, 0.0], dtype=float)
FIXTURE_ROTMAT = np.eye(3)

PART_IDS = ("side_l", "shelf_m", "shelf_t", "side_r")

# 层板 goal：local +X(薄) → world +Z，local +Z(长) → world +X，local +Y → world +Y
SHELF_GOAL_ROTMAT = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=float)

SIDE_GOAL_ROTMAT = np.eye(3)
GOAL_ROTMAT = SIDE_GOAL_ROTMAT  # 兼容旧 import；层板请用 GOAL_REL_ROTMAT


@dataclass(frozen=True)
class MeshFrame:
    path: str
    lo: np.ndarray
    hi: np.ndarray

    @property
    def extent(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def centroid(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @classmethod
    def from_stl(cls, path: str) -> "MeshFrame":
        with open(path, "rb") as f:
            f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            xs, ys, zs = [], [], []
            for _ in range(n_tri):
                f.read(12)
                for __ in range(3):
                    x, y, z = struct.unpack("<fff", f.read(12))
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                f.read(2)
        verts = np.column_stack([xs, ys, zs])
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        return cls(path=os.path.abspath(path), lo=lo, hi=hi)


def _shelf_is_vertical_at_identity(shelf: MeshFrame) -> bool:
    ext = shelf.extent
    return float(ext[2]) > float(ext[0]) * 3.0 and float(shelf.lo[2]) >= -1e-6


def shelf_staging_rotmat(shelf: MeshFrame | None = None) -> np.ndarray:
    """Staging 竖立 rotmat：新 mesh 为 I；旧 mesh（中心原点、I 为平放）为 Rx(+90°)。"""
    frame = shelf if shelf is not None else SHELF_FRAME
    if _shelf_is_vertical_at_identity(frame):
        return np.eye(3)
    return np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0],
    ], dtype=float)


def shelf_staging_z_offset(shelf: MeshFrame, rotmat: np.ndarray) -> float:
    """使旋转后 mesh 底面贴桌面 z=0 的 pos.z。"""
    lo, hi = shelf.lo, shelf.hi
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
    ], dtype=float)
    R = np.asarray(rotmat, dtype=float)
    rotated = (R @ corners.T).T
    return float(-rotated[:, 2].min())


def _side_y_center(side: MeshFrame, shelf: MeshFrame) -> float:
    """水平层板 world Y 半宽 + 侧板半厚。"""
    shelf_y_half = float(shelf.extent[1] / 2.0)
    return shelf_y_half + float(side.extent[1] / 2.0)


def _origin_for_centroid(frame: MeshFrame, rotmat: np.ndarray,
                         target_centroid: np.ndarray) -> np.ndarray:
    return np.asarray(target_centroid, dtype=float) - rotmat @ frame.centroid


def compute_goal_poses(
    side: MeshFrame, shelf: MeshFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    side_y = _side_y_center(side, shelf)
    side_bottom_z = float(side.lo[2])
    mid_z = float(side.lo[2] + side.extent[2] / 2.0)
    thin_half = float(shelf.extent[0] / 2.0)
    top_z = float(side.hi[2] + thin_half)

    Rg = SHELF_GOAL_ROTMAT
    rel_pos = {
        "side_l": np.array([0.0, +side_y, side_bottom_z], dtype=float),
        "side_r": np.array([0.0, -side_y, side_bottom_z], dtype=float),
        "shelf_m": _origin_for_centroid(
            shelf, Rg, np.array([0.0, 0.0, mid_z], dtype=float)),
        "shelf_t": _origin_for_centroid(
            shelf, Rg, np.array([0.0, 0.0, top_z], dtype=float)),
    }
    rel_rot = {
        "side_l": SIDE_GOAL_ROTMAT.copy(),
        "side_r": SIDE_GOAL_ROTMAT.copy(),
        "shelf_m": Rg.copy(),
        "shelf_t": Rg.copy(),
    }
    return rel_pos, rel_rot


def fixture_to_world(rel_pos: np.ndarray, rel_rotmat: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    R_fix = FIXTURE_ROTMAT
    t_fix = FIXTURE_POS
    R_rel = np.eye(3) if rel_rotmat is None else np.asarray(rel_rotmat, dtype=float)
    world_pos = t_fix + R_fix @ np.asarray(rel_pos, dtype=float)
    world_rot = R_fix @ R_rel
    return world_pos, world_rot


SIDE_FRAME = MeshFrame.from_stl(SIDE_STL)
SHELF_FRAME = MeshFrame.from_stl(SHELF_STL)
GOAL_REL_POS, GOAL_REL_ROTMAT = compute_goal_poses(SIDE_FRAME, SHELF_FRAME)

SHELF_STAGING_ROTMAT = shelf_staging_rotmat(SHELF_FRAME)
SHELF_STAGING_Z_OFFSET = shelf_staging_z_offset(SHELF_FRAME, SHELF_STAGING_ROTMAT)
# 竖立 I：底面 lo[2]=0 贴桌 → pos.z=0（勿用 lo[2] 当 pos.z，旧 center-origin mesh 会错）
SIDE_UPRIGHT_Z_OFFSET = shelf_staging_z_offset(SIDE_FRAME, np.eye(3))
