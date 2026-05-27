#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View Initial Layout Only
===========================

只用于查看已经生成的 .layout 初始布局，不做抓取规划、不做 RRT、不播放装配动画。

功能：
1. 读取 .layout 文件；
2. 读取 .asmdef 文件；
3. 显示 work_table / 环境障碍物；
4. 显示机器人 home 姿态，方便检查零件是否和手臂/夹爪穿模；
5. 显示每个零件的初始 staging 位置；
6. 如果某个零件是 preassembled，例如 base_plate，则以实心模型显示在装配区；
7. 可选显示最终目标装配位姿 ghost。

推荐放置路径：
    D:/Project/wrs-sealp/sealp/examples/layout/show_initial_layout_from_file.py

运行：
    python -m sealp.examples.layout.show_initial_layout_from_file

指定 layout：
    python -m sealp.examples.layout.show_initial_layout_from_file ^
      --layout D:/Project/wrs-sealp/sealp/examples/layout/_output/tower_optimal_initial.layout ^
      --asmdef D:/Project/wrs-sealp/sealp/assembly_sequence/_demo_output/topdown_tower.asmdef
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

from wrs import wd, mgm, mcm

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef
from sealp.layout import WorkspaceLayout
from sealp.config import load_config
from sealp.colliders import StaticEnvironment


# 兼容旧版/搜索脚本生成的 .layout：
# 有些 .layout 文件里会把 tuple 保存成 YAML 的 !!python/tuple 标签。
# WorkspaceLayout.load 内部使用 yaml.safe_load，默认不认识这个标签，
# 所以这里只给 SafeLoader 补一个 tuple 构造器，不改成 unsafe_load。
def _enable_yaml_tuple_safe_load():
    tag = "tag:yaml.org,2002:python/tuple"

    def _construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor(tag, _construct_python_tuple)


_enable_yaml_tuple_safe_load()

try:
    import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
except Exception:
    pda = None


DEFAULT_ASMDEF = os.path.join(
    SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef"
)
DEFAULT_LAYOUT = os.path.join(
    SEALP_ROOT, "examples", "layout", "_output", "tower_optimal_initial.layout"
)
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")

DUAL_ARM_Y_OFFSET = 0.62
HOME_JV = np.zeros(6)


def make_model(mesh_path: str, rgba=None):
    cm = mcm.CollisionModel(mesh_path)
    if rgba is not None:
        cm.rgba = np.asarray(rgba, dtype=float)
    return cm


def _load_env_obstacles(config_path: str, base) -> List:
    if not config_path or not os.path.isfile(config_path):
        print(f"[WARN] config 不存在，跳过环境障碍物: {config_path}")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs_list = list(env.obstacle_list)

    for obs in obs_list:
        try:
            obs.rgba = np.array([0.55, 0.55, 0.55, 0.35])
        except Exception:
            pass
        obs.attach_to(base)

    print(f"[ENV] loaded obstacles: {len(obs_list)}")
    return obs_list


def _part_order_from_layout_or_asm(asm: AssemblyDef, layout: WorkspaceLayout) -> List[str]:
    meta = getattr(layout, "metadata", {}) or {}
    order = meta.get("part_order")
    if order:
        return [p for p in order if p in asm.part_ids]

    return [
        step.part_id
        for step in asm.steps
        if step.part_id in asm.part_ids
    ]


def _is_preassembled(pid: str, layout: WorkspaceLayout, part_order: List[str]) -> bool:
    meta = getattr(layout, "metadata", {}) or {}
    arm_choice = meta.get("arm_choice", {}) or {}
    pose_tag = meta.get("pose_tag", {}) or {}

    if arm_choice.get(pid) == "preassembled":
        return True

    if str(pose_tag.get(pid, "")).startswith("preassembled"):
        return True

    if meta.get("preassemble_first_part", False) and part_order and pid == part_order[0]:
        return True

    return False


def _metadata_map(layout: WorkspaceLayout, key: str) -> Dict:
    meta = getattr(layout, "metadata", {}) or {}
    v = meta.get(key, {})
    return v if isinstance(v, dict) else {}


def attach_robot_home(base, layout: WorkspaceLayout, show_robot: bool = True):
    if not show_robot:
        return None

    if pda is None:
        print("[WARN] 无法导入 DualPantheraHTNoBody，跳过机器人显示。")
        return None

    robot_base_pos = np.asarray(getattr(layout, "robot_base_pos", np.zeros(3)), dtype=float)
    robot_base_rotmat = np.asarray(getattr(layout, "robot_base_rotmat", np.eye(3)), dtype=float)

    robot = pda.DualPantheraHTNoBody(
        pos=robot_base_pos,
        rotmat=robot_base_rotmat,
        arm_y_offset=DUAL_ARM_Y_OFFSET,
        enable_cc=True,
    )

    try:
        robot.lft_arm.goto_given_conf(HOME_JV)
        robot.rgt_arm.goto_given_conf(HOME_JV)
    except Exception:
        pass

    try:
        robot.gen_meshmodel(alpha=0.22).attach_to(base)
        mgm.gen_frame(pos=robot_base_pos, rotmat=robot_base_rotmat, ax_length=0.12).attach_to(base)
        print(f"[ROBOT] home shown at {np.round(robot_base_pos, 4).tolist()}")
    except Exception as e:
        print(f"[WARN] robot mesh 显示失败: {type(e).__name__}: {e}")

    return robot


def attach_goal_ghosts(base, asm: AssemblyDef, world_poses: Dict, show_goal_ghosts: bool):
    if not show_goal_ghosts:
        return

    print("\n========== Goal Ghosts ==========")
    for pid, pose in world_poses.items():
        if pid not in asm.part_ids:
            continue

        mesh_path = asm.model_path(pid)
        if not os.path.isfile(mesh_path):
            print(f"[WARN] missing mesh for goal ghost: {pid}, {mesh_path}")
            continue

        gp, gr = pose
        cm = make_model(mesh_path, rgba=[0.70, 0.70, 0.70, 0.18])
        cm.pos = np.asarray(gp, dtype=float)
        cm.rotmat = np.asarray(gr, dtype=float)
        cm.attach_to(base)

        print(f"ghost {pid:14s}: pos={np.round(gp, 4).tolist()}")


def attach_initial_layout(base, asm: AssemblyDef, layout: WorkspaceLayout, part_order: List[str]):
    pose_tag = _metadata_map(layout, "pose_tag")
    rot_name = _metadata_map(layout, "rot_name")
    arm_choice = _metadata_map(layout, "arm_choice")
    grasp_counts = _metadata_map(layout, "grasp_counts")
    topdown_counts = _metadata_map(layout, "topdown_counts_identity")

    colors = [
        np.array([0.90, 0.45, 0.35, 0.88]),
        np.array([0.20, 0.60, 0.95, 0.88]),
        np.array([0.25, 0.80, 0.45, 0.88]),
        np.array([0.95, 0.60, 0.20, 0.88]),
        np.array([0.75, 0.35, 0.85, 0.88]),
        np.array([0.20, 0.85, 0.85, 0.88]),
        np.array([0.85, 0.85, 0.30, 0.88]),
        np.array([0.65, 0.65, 0.95, 0.88]),
    ]

    print("\n========== Initial Layout / Staging ==========")
    shown = []

    for i, pid in enumerate(part_order):
        if pid not in asm.part_ids:
            continue

        st = layout.staging_positions.get(pid)
        if st is None:
            print(f"[WARN] no staging pose for {pid}")
            continue

        mesh_path = asm.model_path(pid)
        if not os.path.isfile(mesh_path):
            print(f"[WARN] missing mesh for {pid}: {mesh_path}")
            continue

        pos, rot = st
        pos = np.asarray(pos, dtype=float)
        rot = np.asarray(rot, dtype=float)

        preassembled = _is_preassembled(pid, layout, part_order)

        if preassembled:
            rgba = np.array([0.40, 0.78, 0.42, 0.95])
        else:
            rgba = colors[i % len(colors)]

        cm = make_model(mesh_path, rgba=rgba)
        cm.pos = pos
        cm.rotmat = rot
        cm.attach_to(base)

        try:
            mgm.gen_frame(pos=pos, rotmat=rot, ax_length=0.04).attach_to(base)
        except Exception:
            pass

        shown.append(pid)

        print(
            f"{pid:14s}: "
            f"pos={np.round(pos, 4).tolist()} "
            f"preassembled={preassembled} "
            f"arm={arm_choice.get(pid, '-')} "
            f"pose={pose_tag.get(pid, '-')} "
            f"rot={rot_name.get(pid, '-')} "
            f"grasp={grasp_counts.get(pid, '-')} "
            f"topdown={topdown_counts.get(pid, '-')}"
        )

    print(f"\n[OK] shown parts: {shown}")


def main():
    parser = argparse.ArgumentParser(
        description="Only view initial staging layout from a .layout file."
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help=".layout 文件路径")
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help=".asmdef 文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml 路径")
    parser.add_argument("--hide-goal-ghosts", action="store_true", help="不显示最终目标 ghost")
    parser.add_argument("--hide-robot", action="store_true", help="不显示机器人 home 姿态")
    parser.add_argument("--hide-env", action="store_true", help="不显示 work_table 等环境")
    parser.add_argument(
        "--cam-pos",
        default="1.05,-1.25,0.85",
        help="相机位置，格式 x,y,z",
    )
    args = parser.parse_args()

    layout_path = os.path.abspath(args.layout)
    asmdef_path = os.path.abspath(args.asmdef)
    config_path = os.path.abspath(args.config)

    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"layout 不存在: {layout_path}")
    if not os.path.isfile(asmdef_path):
        raise FileNotFoundError(f"asmdef 不存在: {asmdef_path}")

    asm = AssemblyDef.load(asmdef_path)
    layout = WorkspaceLayout.load(layout_path)

    part_order = _part_order_from_layout_or_asm(asm, layout)

    assembly_pos = np.asarray(layout.assembly_station_pos, dtype=float)
    assembly_rot = np.asarray(layout.assembly_station_rotmat, dtype=float)

    cam_pos = [float(x.strip()) for x in args.cam_pos.split(",")]

    print("=" * 78)
    print("View Initial Layout Only")
    print(f"asmdef      = {asmdef_path}")
    print(f"layout      = {layout_path}")
    print(f"config      = {config_path}")
    print(f"assembly    = {np.round(assembly_pos, 4).tolist()}")
    print(f"part_order  = {part_order}")
    print(f"metadata    = {list((layout.metadata or {}).keys())}")
    print("=" * 78)

    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=assembly_pos + np.array([0.0, 0.0, 0.10]),
    )

    mgm.gen_frame(pos=assembly_pos, rotmat=assembly_rot, ax_length=0.12).attach_to(base)

    if not args.hide_env:
        _load_env_obstacles(config_path, base)

    attach_robot_home(base, layout, show_robot=not args.hide_robot)

    world_poses = asm.compute_world_poses(
        fixture_pos=assembly_pos,
        fixture_rotmat=assembly_rot,
    )

    attach_goal_ghosts(
        base,
        asm,
        world_poses,
        show_goal_ghosts=not args.hide_goal_ghosts,
    )

    attach_initial_layout(base, asm, layout, part_order)

    print("\n提示：")
    print("  绿色实心模型通常表示 preassembled，例如 base_plate。")
    print("  彩色模型表示 layout 中的初始 staging 零件。")
    print("  半透明灰色模型表示最终目标装配 ghost。")
    print("  每个零件中心的小坐标系表示该零件初始姿态。")

    base.run()


if __name__ == "__main__":
    main()
