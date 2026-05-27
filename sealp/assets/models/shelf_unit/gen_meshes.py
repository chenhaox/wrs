#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Shelf-Unit Mesh Assets
=================================

Creates two STL files for the ``shelf_unit`` assembly task:

    side_panel.stl   — vertical side board (0.180 × 0.018 × 0.300 m)
    shelf.stl        — horizontal shelf (0.180 × 0.300 × 0.015 m)

(``shelf_b.stl`` 带四角孔的底板已不再用于当前 4 件装配，如需可自行调用
``_export_shelf_b_stl`` 生成。)

``side_panel.stl`` 底面在 local z=0（相对几何中心抬升半高 0.15 m），竖立 staging/goal
不穿透桌面；``shelf.stl`` 仍以几何中心为原点（WRS 抓取惯例）。

Run::

    python -m sealp.assets.models.shelf_unit.gen_meshes
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import wrs.basis.trimesh.creation as trm_creation
import wrs.basis.trimesh.util as trm_util


# ── Geometry constants (single source of truth) ───────────────────────────
# 坐标：x=深度 0.18, y=长边 0.30, z=厚度 0.015；侧板 y 方向厚 0.018。
SHELF_XYZ      = np.array([0.090, 0.150, 0.015], dtype=float)
SIDE_PANEL_XYZ = np.array([0.090, 0.018, 0.30], dtype=float)

# 底板四角孔（在 ±x, ±y 四个角附近各一个圆孔）
SHELF_B_HOLE_RADIUS      = 0.012   # 孔半径 12 mm
SHELF_B_HOLE_EDGE_INSET  = 0.035   # 孔心距外边缘 35 mm


def _export_box_stl(xyz_lengths: np.ndarray, out_path: str) -> None:
    mesh = trm_creation.box(extents=np.asarray(xyz_lengths, dtype=float))
    mesh.export(out_path)
    print(f"[OK] {os.path.relpath(out_path, _PROJ_ROOT)}  "
          f"xyz={xyz_lengths.tolist()}  "
          f"n_verts={len(mesh.vertices)}  n_faces={len(mesh.faces)}")


def _export_side_panel_stl(xyz_lengths: np.ndarray, out_path: str) -> None:
    """侧板：底面贴 local z=0（在 centroid box 基础上沿 +Z 抬升 lz/2）。"""
    lx, ly, lz = np.asarray(xyz_lengths, dtype=float).tolist()
    mesh = trm_creation.box(extents=np.array([lx, ly, lz], dtype=float))
    mesh.vertices[:, 2] += lz / 2.0
    mesh.export(out_path)
    print(f"[OK] {os.path.relpath(out_path, _PROJ_ROOT)}  "
          f"xyz={xyz_lengths.tolist()}  bottom@z=0  "
          f"n_verts={len(mesh.vertices)}  n_faces={len(mesh.faces)}")


def _box_mesh(lx: float, ly: float, lz: float,
              cx: float = 0.0, cy: float = 0.0, cz: float = 0.0):
    """Axis-aligned box centered at (cx, cy, cz) with edge lengths (lx, ly, lz)."""
    m = trm_creation.box(extents=np.array([lx, ly, lz], dtype=float))
    m.vertices += np.array([cx, cy, cz], dtype=float)
    return m


def _gen_shelf_b_corner_holes_mesh():
    """Bottom board = outer box minus four corner circular holes.

    Without meshpy/shapely boolean, we approximate the solid as a **frame**
    of five axis-aligned bars whose inner corner cut-outs match the four
    hole bounding boxes (each hole center inset from both adjacent outer
    edges by ``SHELF_B_HOLE_EDGE_INSET``, radius ``SHELF_B_HOLE_RADIUS``).
    """
    lx, ly, lz = SHELF_XYZ.tolist()
    hw, hd, hh = lx / 2.0, ly / 2.0, lz / 2.0
    inset = SHELF_B_HOLE_EDGE_INSET
    r = SHELF_B_HOLE_RADIUS

    # 孔心坐标（四角）
    hx = hw - inset          # 0.055
    hy = hd - inset          # 0.115
    # 孔区在内侧边界：孔心 ± r
    ix = hx + r              # 0.067  — 侧翼内缘 x
    iy = hy + r              # 0.127  — 上下翼内缘 y

    parts = [
        # 中央横条（贯穿左右，y 方向去掉四角孔区）
        _box_mesh(lx, 2.0 * iy, lz, cx=0.0, cy=0.0, cz=0.0),
        # 下缘条（两角孔之间）
        _box_mesh(2.0 * ix, 2.0 * (hd - iy), lz,
                  cx=0.0, cy=-(hd + iy) / 2.0, cz=0.0),
        # 上缘条
        _box_mesh(2.0 * ix, 2.0 * (hd - iy), lz,
                  cx=0.0, cy=(hd + iy) / 2.0, cz=0.0),
        # 左缘条
        _box_mesh(2.0 * (hw - ix), 2.0 * iy, lz,
                  cx=-(hw + ix) / 2.0, cy=0.0, cz=0.0),
        # 右缘条
        _box_mesh(2.0 * (hw - ix), 2.0 * iy, lz,
                  cx=(hw + ix) / 2.0, cy=0.0, cz=0.0),
    ]
    return trm_util.concatenate(parts)


def _export_shelf_b_stl(out_path: str) -> None:
    mesh = _gen_shelf_b_corner_holes_mesh()
    mesh.export(out_path)
    print(f"[OK] {os.path.relpath(out_path, _PROJ_ROOT)}  "
          f"(bottom board, 4 corner holes)  "
          f"outer={SHELF_XYZ.tolist()}  "
          f"hole_r={SHELF_B_HOLE_RADIUS}  inset={SHELF_B_HOLE_EDGE_INSET}  "
          f"n_verts={len(mesh.vertices)}  n_faces={len(mesh.faces)}")


def main() -> None:
    side_path = os.path.join(_HERE, "side_panel.stl")
    shelf_path = os.path.join(_HERE, "shelf.stl")
    _export_side_panel_stl(SIDE_PANEL_XYZ, side_path)
    _export_box_stl(SHELF_XYZ, shelf_path)
    print("\nShelf-Unit STL assets generated.")
    print(f"  side_panel: {side_path}")
    print(f"  shelf     : {shelf_path}")


if __name__ == "__main__":
    main()
