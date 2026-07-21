#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 fixture 基座标下展示 shelf_unit 目标装配位姿（与 asmdef / mesh_frames 一致）。"""
from __future__ import annotations

import os

import numpy as np

from wrs import wd, mgm, mcm

from sealp.assets.models.shelf_unit.mesh_frames import (
    FIXTURE_POS,
    FIXTURE_ROTMAT,
    GOAL_REL_POS,
    GOAL_REL_ROTMAT,
    PART_IDS,
    SHELF_FRAME,
    SHELF_STL,
    SIDE_FRAME,
    SIDE_STL,
    TABLE_TOP_Z,
    fixture_to_world,
)


def _print_mesh_info() -> None:
    print("=== Mesh local AABB ===")
    for label, frame in [("side_panel", SIDE_FRAME), ("shelf", SHELF_FRAME)]:
        print(f"  {label:10s} lo={frame.lo.round(4).tolist()}  hi={frame.hi.round(4).tolist()}")
    print(f"\nfixture origin (world) = {FIXTURE_POS.tolist()}")
    print(f"table top z = {TABLE_TOP_Z}\n")


def _add_table(base, size=(1.4, 1.4), thickness=0.01, rgb=(0.85, 0.85, 0.85), alpha=0.35):
    top = TABLE_TOP_Z
    center = np.array([0.0, 0.0, top - thickness / 2.0])
    mgm.gen_box(
        xyz_lengths=np.array([size[0], size[1], thickness], dtype=float),
        pos=center,
        rgb=np.asarray(rgb, dtype=float),
        alpha=alpha,
    ).attach_to(base)


def _add_part(base, part_id: str, stl_path: str, rel_pos, rel_rotmat, rgba):
    world_pos, world_rot = fixture_to_world(rel_pos, rel_rotmat)
    model = mcm.CollisionModel(stl_path)
    model.pos = world_pos
    model.rotmat = world_rot
    model.rgba = np.asarray(rgba, dtype=float)
    model.attach_to(base)
    print(
        f"{part_id:8s}  rel={np.asarray(rel_pos).round(4).tolist()}  "
        f"world={world_pos.round(4).tolist()}"
    )
    return model


def main() -> None:
    _print_mesh_info()

    look_at = FIXTURE_POS + np.array([0.0, 0.0, SIDE_FRAME.extent[2] / 2.0])
    base = wd.World(cam_pos=[0.75, -0.45, 0.55], lookat_pos=look_at)

    # 世界基座标 + fixture 装配基座标
    mgm.gen_frame(ax_length=0.06).attach_to(base)
    mgm.gen_frame(pos=FIXTURE_POS, rotmat=FIXTURE_ROTMAT, ax_length=0.05).attach_to(base)
    _add_table(base)

    colors = {
        "side_l": [0.75, 0.55, 0.35, 1.0],
        "side_r": [0.65, 0.45, 0.28, 1.0],
        "shelf_m": [0.45, 0.70, 0.90, 1.0],
        "shelf_t": [0.35, 0.60, 0.85, 1.0],
    }

    print("=== Goal poses (fixture base → world) ===")
    for pid in PART_IDS:
        rel = GOAL_REL_POS[pid]
        stl = SIDE_STL if pid.startswith("side_") else SHELF_STL
        _add_part(base, pid, stl, rel, GOAL_REL_ROTMAT[pid], colors[pid])

    base.run()


if __name__ == "__main__":
    main()
