#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute + Visualize Assembly From .layout
================================================

这个脚本用于读取已经搜索得到的 WorkspaceLayout（.layout），然后按照 asmdef
顺序执行装配路径规划，并把能成功规划到的步骤动画显示出来。

特点：
1. 通用性：
   - 不写死 tower 的具体坐标；
   - 读取 .asmdef 决定零件模型和目标装配位姿；
   - 读取 .layout 决定 assembly_station 和 staging 初始摆放；
   - 读取 .layout.metadata["arm_choice"] 作为优先手臂，但失败后会自动尝试另一只手。
2. 动态障碍物：
   - 已装好的零件：使用 goal pose，作为后续步骤障碍物；
   - 未装零件：使用 staging pose，作为后续步骤障碍物；
   - 当前搬运零件：从静态障碍列表中移除，由 TransportPrimitive 作为被抓物体处理。
3. 尽力执行：
   - 如果某一步失败，不会崩溃，也不会停止；
   - 会继续尝试后续零件；
   - 最后播放所有成功规划出来的步骤动画；
   - 动画中会补充显示当前搬运零件，避免零件隐身或跑到原点；
   - 同时显示初始布局、目标 ghost、已成功装配的零件。
4. middle_plate **直接走双臂换手**（跳过单臂 pick-place，需预生成 hopg）：
   - ``tower_handover/middle_plate_hopg.pickle``
   - 生成命令：``python -m sealp.examples.grasp.gen_middle_plate_regrasp_data``
5. 默认使用 mesh/triangles 碰撞，不用 box。
   - 想加速可以用 --cdprim-type box。

推荐放置路径：
    sealp/examples/motion/execute_layout_sequence_visual.py

默认运行：
    python -m sealp.examples.motion.execute_layout_sequence_visual

指定 tower layout：
    python -m sealp.examples.motion.execute_layout_sequence_visual ^
      --layout D:/Project/wrs-sealp/sealp/examples/layout/_output/tower_optimal_initial.layout ^
      --asmdef D:/Project/wrs-sealp/sealp/assembly_sequence/_demo_output/topdown_tower.asmdef ^
      --grasp-dir D:/Project/wrs-sealp/sealp/examples/grasp/tower_grasp

说明：
    默认会优先使用 .layout 里的 assembly_station_pos。
    如果 layout 里 base_plate 被标记为 preassembled，则 step=0 会跳过抓取，
    直接把 base_plate 作为已经装好的动态障碍物加入后续规划。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import wrs.basis.robot_math as rm
import wrs.manipulation.handover_regrasp as horeg
from wrs import wd, mgm, mcm
from direct.task.TaskManagerGlobal import taskMgr

# 允许直接右键运行
_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # 假设脚本位于 sealp/examples/motion 下
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.layout import WorkspaceLayout
from sealp.primitives.transport import TransportPrimitive
from wrs.grasping.grasp import GraspCollection

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda


# ============================================================
# 默认路径
# ============================================================

DEFAULT_ASMDEF = os.path.join(
    SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef"
)
DEFAULT_LAYOUT = os.path.join(
    SEALP_ROOT, "examples", "layout", "_output", "tower_optimal_initial.layout"
)
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")
DEFAULT_GRASP_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")
DEFAULT_HANDOVER_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_handover")

# middle_plate 直接走换手，不走单臂 pick-place
HANDOVER_PART_IDS = frozenset({"middle_plate"})


# ============================================================
# 规划参数：默认 mesh / triangles
# ============================================================

DUAL_ARM_Y_OFFSET = 0.62
HOME_JV = np.zeros(6)

APPROACH_DIST = 0.0
PICK_DEPART_DIST = 0.02
PLACE_APPROACH_DIST = 0.02
PLACE_DEPART_DIST = 0.05
LINEAR_GRANULARITY = 0.04

# 多方向候选的水平倾斜量
MOTION_TILT = 0.35

# RRT 稀疏化，不降低障碍物要求，只降低搜索开销
RRT_EXT_DIST = 0.30
RRT_SMOOTHING_N_ITER = 150
RRT_MAX_TIME = 10.0

# 默认使用 mesh/triangles
DEFAULT_CDPRIM_TYPE = "triangles"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class StepMotion:
    step_id: int
    part_id: str
    arm_tag: str
    motion_tag: str
    mot_data: object


@dataclass
class FailedStep:
    step_id: int
    part_id: str
    reason: str


@dataclass
class ExecutionSummary:
    success_steps: List[StepMotion]
    failed_steps: List[FailedStep]
    failed_step_id: Optional[int] = None
    failed_part_id: Optional[str] = None
    failed_reason: str = ""


@dataclass
class _AnimMotionData:
    """Handover 多段 MotionData 合并后的动画载体。"""

    mesh_list: list


class _QuietHandoverPlanner(horeg.HandoverPlanner):
    def show_graph(self):
        return


def _duplicate_grasp_collection(gc: GraspCollection) -> GraspCollection:
    try:
        return gc.copy()
    except Exception:
        return copy.deepcopy(gc)


def _merge_motion_mesh_list(motion_list: List) -> _AnimMotionData:
    """合并 MotionData.mesh_list。

    注意：
        这里不再做“左臂蓝色 / 右臂红色 / middle_plate 黄色跟随当前臂”的二次重绘。
        直接使用规划器原始生成的 mesh_list，避免因为手动 attach 零件而把模型放到原点。
    """
    mesh_list = []
    for md in motion_list:
        mesh_list.extend(getattr(md, "mesh_list", []) or [])
    return _AnimMotionData(mesh_list=mesh_list)


def _interp_rotmat(r0, r1, t: float) -> np.ndarray:
    """在两个旋转矩阵之间做一个稳定的近似插值。"""
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    m = (1.0 - t) * r0 + t * r1
    try:
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vt
        return r
    except Exception:
        return r0 if t < 0.5 else r1


def _pose_from_any(raw):
    """尽量从 MotionData 里可能存在的 obj pose 表达中取出 (pos, rotmat)。"""
    if raw is None:
        return None

    # 常见形式：((x,y,z), rotmat) 或 [pos, rotmat]
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        pos, rot = raw
        try:
            pos = np.asarray(pos, dtype=float).reshape(3)
            rot = np.asarray(rot, dtype=float).reshape(3, 3)
            return pos, rot
        except Exception:
            return None

    # 常见形式：4x4 齐次矩阵
    try:
        arr = np.asarray(raw, dtype=float)
        if arr.shape == (4, 4):
            return arr[:3, 3].copy(), arr[:3, :3].copy()
    except Exception:
        pass

    return None


def _extract_obj_pose_list(md) -> List[Tuple[np.ndarray, np.ndarray]]:
    """兼容不同 WRS MotionData 版本，尝试提取物体逐帧位姿。"""
    candidate_attrs = (
        "obj_pose_list",
        "objpose_list",
        "object_pose_list",
        "obj_pose_seq",
        "objpose_seq",
        "obj_pose_list_list",
    )
    for attr in candidate_attrs:
        raw_list = getattr(md, attr, None)
        if not raw_list:
            continue
        out = []
        for raw in raw_list:
            pose = _pose_from_any(raw)
            if pose is not None:
                out.append(pose)
        if out:
            return out
    return []


def _concat_obj_pose_lists(motion_list: List) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    for md in motion_list:
        out.extend(_extract_obj_pose_list(md))
    return out


def _attach_moving_object_overlay_to_mesh_list(
    mesh_list: List,
    asm: AssemblyDef,
    part_id: str,
    start_pose: Tuple[np.ndarray, np.ndarray],
    goal_pose: Tuple[np.ndarray, np.ndarray],
    cdprim_type: str = DEFAULT_CDPRIM_TYPE,
    obj_pose_list: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    """给每一帧补一个移动零件模型，防止动画里零件“隐身”或跑到原点。

    优先使用 MotionData 自带的 obj_pose_list；如果当前 WRS 版本没有保存逐帧物体位姿，
    则在 layout 的真实 start_pose 和 asmdef 的 goal_pose 之间做插值显示。
    这至少保证：零件从真实 layout 初始位置出发，最后到真实装配位置，不会出现在世界原点。
    """
    if not mesh_list:
        return

    mesh_path = asm.model_path(part_id)
    if not mesh_path or not os.path.isfile(mesh_path):
        return

    sp, sr = start_pose
    gp, gr = goal_pose
    sp = np.asarray(sp, dtype=float)
    sr = np.asarray(sr, dtype=float)
    gp = np.asarray(gp, dtype=float)
    gr = np.asarray(gr, dtype=float)

    pose_list = obj_pose_list or []
    n = len(mesh_list)

    for i, frame in enumerate(mesh_list):
        # 防止重复补同一个物体
        if getattr(frame, "_sealp_moving_overlay_part_id", None) == part_id:
            continue

        if i < len(pose_list):
            pos, rot = pose_list[i]
        else:
            t = 0.0 if n <= 1 else i / float(n - 1)
            pos = (1.0 - t) * sp + t * gp
            rot = _interp_rotmat(sr, gr, t)

        try:
            obj = make_collision_model(mesh_path, cdprim_type=cdprim_type)
            obj.pos = np.asarray(pos, dtype=float)
            obj.rotmat = np.asarray(rot, dtype=float)
            obj._sealp_part_id = part_id
            obj._sealp_role = "moving_object_overlay"
            # 不强制改成黄色，保留模型自身颜色/材质，避免 middle_plate 被特殊染色。
            obj.attach_to(frame)
            setattr(frame, "_sealp_moving_overlay_part_id", part_id)
        except Exception as e:
            print(f"[WARN] moving object overlay failed for {part_id}: {type(e).__name__}: {e}")
            return


def _apply_motion_end_states(motion_list: List) -> None:
    for md in motion_list:
        robot = getattr(md, "robot", None)
        jv_list = getattr(md, "jv_list", None)
        if robot is None or not jv_list:
            continue
        try:
            robot.goto_given_conf(jv_list[-1])
        except Exception:
            pass


# ============================================================
# 基础工具
# ============================================================

def _patch_rrt():
    """让 RRT 稍微稀疏一点，避免太慢。"""
    try:
        from wrs.motion.probabilistic.rrt_connect import RRTConnect
    except Exception:
        return

    if getattr(RRTConnect.plan, "_layout_visual_patched", False):
        return

    _orig_plan = RRTConnect.plan

    def _patched_plan(self, *args, **kwargs):
        kwargs["ext_dist"] = RRT_EXT_DIST
        kwargs["smoothing_n_iter"] = RRT_SMOOTHING_N_ITER
        kwargs["max_time"] = RRT_MAX_TIME
        return _orig_plan(self, *args, **kwargs)

    _patched_plan._layout_visual_patched = True
    RRTConnect.plan = _patched_plan
    print(
        f"[RRT patch] ext_dist={RRT_EXT_DIST}, "
        f"smoothing={RRT_SMOOTHING_N_ITER}, max_time={RRT_MAX_TIME}s"
    )


def _unit_vec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def make_collision_model(mesh_path: str, cdprim_type: str = DEFAULT_CDPRIM_TYPE):
    """创建 CollisionModel。

    默认 cdprim_type="triangles"，也就是 mesh 碰撞。
    如果当前 WRS 版本不支持 triangles，则自动回退默认构造。
    """
    cdprim_type = str(cdprim_type or "triangles")

    trials = []
    if cdprim_type not in ("default", "none", ""):
        trials.append({"initor": mesh_path, "cdprim_type": cdprim_type})
    if cdprim_type != "triangles":
        trials.append({"initor": mesh_path, "cdprim_type": "triangles"})
    trials.append({"initor": mesh_path})
    trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})

    last_err = None
    for kw in trials:
        try:
            cm = mcm.CollisionModel(**kw)
            cm._sealp_cdprim_type = kw.get("cdprim_type", "default")
            return cm
        except TypeError as e:
            last_err = e
            try:
                if "cdprim_type" in kw:
                    cm = mcm.CollisionModel(mesh_path, cdprim_type=kw["cdprim_type"])
                    cm._sealp_cdprim_type = kw["cdprim_type"]
                    return cm
                else:
                    cm = mcm.CollisionModel(mesh_path)
                    cm._sealp_cdprim_type = "default"
                    return cm
            except Exception as ee:
                last_err = ee
        except Exception as e:
            last_err = e

    raise RuntimeError(f"无法创建 CollisionModel: {mesh_path}, last_err={last_err!r}")


def load_env_obstacles(config_path: str, base=None) -> List:
    """从 sample_config.yaml 加载静态环境障碍物。"""
    if not config_path or not os.path.isfile(config_path):
        print(f"[WARN] config 不存在，不加载环境障碍物: {config_path}")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs_list = list(env.obstacle_list)

    for obs in obs_list:
        obs._sealp_role = "environment_obstacle"
        if base is not None:
            obs.attach_to(base)

    print(f"[环境] 已加载 {len(obs_list)} 个静态障碍物。")
    return obs_list


def _model_id_for_part(asm: AssemblyDef, part_id: str) -> Optional[str]:
    """从 AssemblyDef 中尽量取出 part 对应的 model id。"""
    for attr in ("parts", "part_defs"):
        parts = getattr(asm, attr, None)
        if parts is None:
            continue

        if isinstance(parts, dict):
            p = parts.get(part_id)
            if p is not None:
                return getattr(p, "model", None) or getattr(p, "model_id", None)
        else:
            for p in parts:
                if getattr(p, "part_id", None) == part_id:
                    return getattr(p, "model", None) or getattr(p, "model_id", None)

    return None


def _find_grasp_pickle(
    asm: AssemblyDef,
    part_id: str,
    grasp_dir: str,
    explicit_map: Optional[Dict[str, str]] = None,
) -> str:
    """为 part_id 找 grasp pickle。

    兼容 tower:
        post_bl/post_fl/post_br/post_fr 共用 tower_post_grasps.pickle
    """
    explicit_map = explicit_map or {}

    mesh_path = asm.model_path(part_id)
    mesh_base = os.path.splitext(os.path.basename(mesh_path))[0]
    model_id = _model_id_for_part(asm, part_id)

    keys = [part_id, mesh_base]
    if model_id:
        keys.append(model_id)
    if part_id.startswith("post_"):
        keys.append("post")

    # 显式映射优先
    for k in keys:
        if k in explicit_map:
            p = explicit_map[k]
            if not os.path.isabs(p):
                p = os.path.join(grasp_dir, p)
            if os.path.isfile(p):
                return p

    names = []
    for k in keys:
        names += [
            f"{k}_grasps.pickle",
            f"{k}_grasps_topdown.pickle",
            f"{k}.pickle",
            f"tower_{k}_grasps.pickle",
            f"tower_{k}_grasps_topdown.pickle",
        ]

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(grasp_dir, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"找不到 {part_id} 的 grasp pickle。尝试 keys={keys}, grasp_dir={grasp_dir}"
    )


def load_grasp_cache(
    asm: AssemblyDef,
    part_ids: List[str],
    grasp_dir: str,
    explicit_map: Optional[Dict[str, str]] = None,
) -> Dict[str, GraspCollection]:
    out = {}
    file_cache = {}

    print("\n========== Grasp 文件加载 ==========")
    for pid in part_ids:
        pkl = _find_grasp_pickle(asm, pid, grasp_dir, explicit_map)
        if pkl not in file_cache:
            file_cache[pkl] = GraspCollection.load_from_disk(file_name=pkl)
        out[pid] = file_cache[pkl]
        print(f"{pid:16s}: {pkl}  n={len(out[pid])}")

    return out


def _load_json_map(text: str) -> Dict[str, str]:
    if not text:
        return {}
    return dict(json.loads(text))


def _motion_candidate_kwargs(pid: str, arm_tag: str):
    """多套 pick/place 接近撤离方向候选。

    先试纯 Z，再试带一点水平偏置的方向。
    """
    t = float(MOTION_TILT)

    specs = [
        ("z",       [0, 0, 1],      [0, 0, -1],      [0, 0, 1]),
        ("x_plus",  [t, 0, 1],      [-t, 0, -1],     [t, 0, 1]),
        ("x_minus", [-t, 0, 1],     [t, 0, -1],      [-t, 0, 1]),
        ("y_plus",  [0, t, 1],      [0, -t, -1],     [0, t, 1]),
        ("y_minus", [0, -t, 1],     [0, t, -1],      [0, -t, 1]),
    ]

    if arm_tag == "rgt":
        order = ["z", "y_minus", "x_plus", "x_minus", "y_plus"]
    else:
        order = ["z", "y_plus", "x_minus", "x_plus", "y_minus"]

    spec_map = {name: (pd, pa, pld) for name, pd, pa, pld in specs}

    out = []
    for name in order:
        pd, pa, pld = spec_map[name]
        out.append((
            name,
            dict(
                pick_depart_direction=_unit_vec(pd),
                pick_depart_distance=PICK_DEPART_DIST,
                place_approach_direction_list=[_unit_vec(pa)],
                place_approach_distance_list=[PLACE_APPROACH_DIST],
                place_depart_direction_list=[_unit_vec(pld)],
                place_depart_distance_list=[PLACE_DEPART_DIST],
            )
        ))
    return out


def _arm_try_order(pid: str, preferred: Optional[str]) -> List[str]:
    """优先使用 layout 里给出的 arm_choice，失败后尝试另一只手。"""
    if preferred in ("lft", "rgt"):
        other = "rgt" if preferred == "lft" else "lft"
        return [preferred, other]

    # 没有 preferred 时按零件名给个直觉顺序，不是强制
    if pid.endswith("_r") or pid.endswith("br") or pid.endswith("fr"):
        return ["rgt", "lft"]
    return ["lft", "rgt"]


def _is_preassembled(pid: str, layout: WorkspaceLayout) -> bool:
    meta = getattr(layout, "metadata", {}) or {}
    arm_choice = meta.get("arm_choice", {})
    pose_tag = meta.get("pose_tag", {})

    if arm_choice.get(pid) == "preassembled":
        return True
    if str(pose_tag.get(pid, "")).startswith("preassembled"):
        return True

    # 兼容 metadata 里的 preassemble_first_part
    part_order = meta.get("part_order", [])
    if meta.get("preassemble_first_part", False) and part_order and pid == part_order[0]:
        return True

    return False


def _step_for_part(asm: AssemblyDef, part_id: str):
    """根据 part_id 找对应 StepDef。"""
    for s in asm.steps:
        if s.part_id == part_id:
            return s
    return None


def _default_contact_exclusion_map(asm: AssemblyDef) -> Dict[str, List[str]]:
    """默认接触/插接豁免表。

    通用规则：
        当前零件的 direct parent 会自动在 _contact_exclusion_set 中加入；
        这里主要放一些 asmdef parent 无法表达但几何上明显插接/承托的关系。

    对当前 tower：
        - top_cross 竖着插入 middle_plate 顶面方孔，所以规划 top_cross 时要临时排除 middle_plate；
        - middle_plate 放到四根 post 顶部，最终接触面附近可能被 mesh 判交，所以规划 middle_plate 时可临时排除四根 post。
    """
    part_ids = set(getattr(asm, "part_ids", []))
    out: Dict[str, List[str]] = {}

    if "top_cross" in part_ids and "middle_plate" in part_ids:
        out.setdefault("top_cross", []).append("middle_plate")

    post_ids = [p for p in ("post_bl", "post_fl", "post_br", "post_fr") if p in part_ids]
    if "middle_plate" in part_ids and post_ids:
        out.setdefault("middle_plate", []).extend(post_ids)

    return out


def _part_order_from_asm_or_layout(asm: AssemblyDef, layout: WorkspaceLayout) -> List[str]:
    meta = getattr(layout, "metadata", {}) or {}
    order = meta.get("part_order")
    if order:
        return [p for p in order if p in asm.part_ids]

    return [
        s.part_id for s in asm.steps
        if s.part_id in asm.part_ids
    ]


# ============================================================
# 可视化
# ============================================================

def attach_goal_ghosts(base, asm, world_poses, cdprim_type: str):
    print("\n========== 目标 ghost ==========")
    for pid, (gp, gr) in world_poses.items():
        if pid not in asm.part_ids:
            continue
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue

        ghost = make_collision_model(mp, cdprim_type=cdprim_type)
        ghost.pos = np.asarray(gp, dtype=float)
        ghost.rotmat = np.asarray(gr, dtype=float)
        ghost.rgba = np.array([0.7, 0.7, 0.7, 0.18])
        ghost.attach_to(base)
        print(f"ghost {pid:14s}: pos={np.round(gp, 4).tolist()}")


def attach_staging_visuals(base, asm, layout, part_order, cdprim_type: str):
    colors = [
        np.array([0.90, 0.45, 0.35, 0.85]),
        np.array([0.25, 0.60, 0.95, 0.85]),
        np.array([0.25, 0.80, 0.45, 0.85]),
        np.array([0.85, 0.55, 0.20, 0.85]),
        np.array([0.75, 0.35, 0.85, 0.85]),
        np.array([0.35, 0.85, 0.85, 0.85]),
        np.array([0.85, 0.85, 0.35, 0.85]),
    ]

    vis = {}
    print("\n========== 初始 staging 彩色显示 ==========")
    for i, pid in enumerate(part_order):
        st = layout.staging_positions.get(pid)
        if st is None:
            continue
        pos, rot = st
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue

        cm = make_collision_model(mp, cdprim_type=cdprim_type)
        cm.pos = np.asarray(pos, dtype=float)
        cm.rotmat = np.asarray(rot, dtype=float)
        cm.rgba = colors[i % len(colors)]
        cm.attach_to(base)
        mgm.gen_frame(pos=pos, rotmat=rot, ax_length=0.035).attach_to(base)
        vis[pid] = cm
        print(f"{pid:14s}: pos={np.round(pos, 4).tolist()}")

    return vis


def animate_success_steps(base, step_motions: List[StepMotion], interval: float = 0.02, auto_play: bool = True):
    """播放已成功步骤的 MotionData 动画。"""
    if not step_motions:
        print("\n无成功轨迹可播放：仅显示初始布局、目标 ghost 和机械臂。")
        return

    motion_items = []
    for sm in step_motions:
        md = sm.mot_data
        mesh_list = getattr(md, "mesh_list", None)
        if mesh_list is None or len(mesh_list) == 0:
            continue
        motion_items.append(sm)

    if not motion_items:
        print("\n成功步骤没有 mesh_list，无法播放动画。")
        return

    class _State:
        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.paused = not auto_play

    state = _State()

    print("\n动画控制：")
    print("  默认自动播放；按 SPACE 可以逐帧推进/观察。")
    print(f"  可播放成功步骤数：{len(motion_items)}")

    def _update(st, task):
        if st.step_idx >= len(motion_items):
            st.step_idx = 0
            st.frame_idx = 0

        sm = motion_items[st.step_idx]
        mesh_list = sm.mot_data.mesh_list

        # detach 上一帧
        if st.frame_idx > 0 and st.frame_idx - 1 < len(mesh_list):
            mesh_list[st.frame_idx - 1].detach()
        elif st.frame_idx == 0 and st.step_idx > 0:
            prev = motion_items[st.step_idx - 1].mot_data.mesh_list
            if prev:
                prev[-1].detach()

        if st.frame_idx >= len(mesh_list):
            # 清掉本 step 所有帧
            for m in mesh_list:
                m.detach()
            print(f"[动画] step={sm.step_id} {sm.part_id} 播放结束")
            st.step_idx += 1
            st.frame_idx = 0
            return task.again

        mesh_list[st.frame_idx].attach_to(base)

        # 自动播放，或者按 SPACE 逐帧
        if auto_play or base.inputmgr.keymap.get("space", False):
            st.frame_idx += 1

        return task.again

    taskMgr.doMethodLater(interval, _update, "layout_sequence_animation", extraArgs=[state], appendTask=True)


# ============================================================
# 动态执行器
# ============================================================

class LayoutSequenceVisualizer:
    def __init__(
        self,
        asm: AssemblyDef,
        layout: WorkspaceLayout,
        config_path: str,
        grasp_dir: str,
        base,
        cdprim_type: str = DEFAULT_CDPRIM_TYPE,
        grasp_map: Optional[Dict[str, str]] = None,
        contact_exclusion_map: Optional[Dict[str, List[str]]] = None,
        enable_middle_plate_regrasp: bool = True,
        handover_dir: str = DEFAULT_HANDOVER_DIR,
    ):
        self.asm = asm
        self.layout = layout
        self.config_path = config_path
        self.grasp_dir = grasp_dir
        self.base = base
        self.cdprim_type = cdprim_type
        self.grasp_map = grasp_map or {}
        self.enable_middle_plate_regrasp = bool(enable_middle_plate_regrasp)
        self._middle_plate_hopg = os.path.join(
            handover_dir or DEFAULT_HANDOVER_DIR, "middle_plate_hopg.pickle"
        )

        # 接触/插接豁免表：每个 step 规划时，从动态障碍物中临时排除这些已装件。
        # direct parent 会自动加入；这里叠加默认 tower 规则和用户传入规则。
        self.contact_exclusion_map = _default_contact_exclusion_map(self.asm)
        if contact_exclusion_map:
            for k, v in contact_exclusion_map.items():
                self.contact_exclusion_map.setdefault(k, [])
                for item in v:
                    if item not in self.contact_exclusion_map[k]:
                        self.contact_exclusion_map[k].append(item)

        self.part_order = _part_order_from_asm_or_layout(self.asm, self.layout)

        self.fixture_pos = np.asarray(self.layout.assembly_station_pos, dtype=float)
        self.fixture_rotmat = np.asarray(self.layout.assembly_station_rotmat, dtype=float)

        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        self.robot_base_pos = np.asarray(
            getattr(self.layout, "robot_base_pos", np.zeros(3)), dtype=float
        )
        self.robot_base_rotmat = np.asarray(
            getattr(self.layout, "robot_base_rotmat", np.eye(3)), dtype=float
        )

        self.robot = pda.DualPantheraHTNoBody(
            pos=self.robot_base_pos,
            rotmat=self.robot_base_rotmat,
            arm_y_offset=DUAL_ARM_Y_OFFSET,
            enable_cc=True,
        )
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)

        self.env_obstacles = load_env_obstacles(config_path, base)
        self.grasps = load_grasp_cache(self.asm, self.part_order, grasp_dir, self.grasp_map)

        self.staging_models: Dict[str, object] = {}
        self.goal_models: Dict[str, object] = {}
        self.staging_visuals: Dict[str, object] = {}

        self._build_collision_models()

    def _build_collision_models(self):
        print("\n========== 构建 staging / goal 动态障碍模型 ==========")

        for pid in self.part_order:
            if pid not in self.asm.part_ids:
                continue

            mp = self.asm.model_path(pid)

            # staging
            st = self.layout.staging_positions.get(pid)
            if st is not None and os.path.isfile(mp):
                pos, rot = st
                cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
                cm.pos = np.asarray(pos, dtype=float)
                cm.rotmat = np.asarray(rot, dtype=float)
                cm._sealp_part_id = pid
                cm._sealp_role = "staging_on_table"
                self.staging_models[pid] = cm

            # goal
            if pid in self.world_poses and os.path.isfile(mp):
                gp, gr = self.world_poses[pid]
                gm = make_collision_model(mp, cdprim_type=self.cdprim_type)
                gm.pos = np.asarray(gp, dtype=float)
                gm.rotmat = np.asarray(gr, dtype=float)
                gm._sealp_part_id = pid
                gm._sealp_role = "assembled_at_goal"
                self.goal_models[pid] = gm

        print(f"staging_models = {list(self.staging_models.keys())}")
        print(f"goal_models    = {list(self.goal_models.keys())}")

    def _contact_exclusion_set(self, current_pid: str, placed: set) -> set:
        """当前 step 的接触/插接豁免集合。

        这些零件不会加入 obstacle_list：
        1. 当前零件的 direct parent；
        2. contact_exclusion_map 中声明的接触件。

        典型例子：
            post_bl 插入 base_plate 的孔时，base_plate 是父件，不能作为普通障碍；
            top_cross 插入 middle_plate 的方孔时，middle_plate 也应临时排除。
        """
        excl = set()

        step = _step_for_part(self.asm, current_pid)
        parent_id = getattr(step, "parent_id", None) if step is not None else None
        if parent_id and parent_id != "fixture":
            excl.add(parent_id)

        for p in self.contact_exclusion_map.get(current_pid, []):
            excl.add(p)

        # 只排除已经装好的接触件；还没装的零件仍然应该作为 staging 障碍。
        return {p for p in excl if p in placed}

    def _current_obstacles(self, current_pid: str, placed: set) -> List:
        """动态障碍物。

        已装件：goal pose，作为后续障碍物；
        未装件：staging pose，作为后续障碍物；
        当前件：不作为静态障碍，交给 TransportPrimitive 作为 moving object；
        当前件的父件/插接接触件：临时排除，避免“插入孔/放到支撑件上”被误判为碰撞。
        """
        obs = list(self.env_obstacles)
        excluded = self._contact_exclusion_set(current_pid, placed)

        if excluded:
            print(f"    [接触豁免] 当前 {current_pid!r} 规划时临时排除已装件: {sorted(excluded)}")

        for pid in placed:
            if pid in excluded:
                continue
            if pid in self.goal_models:
                obs.append(self.goal_models[pid])

        for pid in self.part_order:
            if pid == current_pid or pid in placed:
                continue
            if pid in self.staging_models:
                obs.append(self.staging_models[pid])

        return obs

    def _uses_handover_direct(self, pid: str) -> bool:
        return self.enable_middle_plate_regrasp and pid in HANDOVER_PART_IDS

    def _handover_arm_pairs(self, pid: str):
        meta = getattr(self.layout, "metadata", {}) or {}
        preferred = (meta.get("arm_choice", {}) or {}).get(pid)
        pairs = [
            ("lft", "rgt", self.robot.lft_arm, self.robot.rgt_arm),
            ("rgt", "lft", self.robot.rgt_arm, self.robot.lft_arm),
        ]
        if preferred == "rgt":
            pairs.reverse()
        elif preferred == "lft":
            pass
        return pairs

    def _try_middle_plate_handover(
        self,
        pid: str,
        start_pose: Tuple[np.ndarray, np.ndarray],
        goal_pose: Tuple[np.ndarray, np.ndarray],
        gc: GraspCollection,
        obstacle_list: List,
    ) -> Tuple[Optional[StepMotion], str]:
        if not os.path.isfile(self._middle_plate_hopg):
            return None, f"{pid}: hopg missing: {self._middle_plate_hopg}"

        obj_cm = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
        # 不能只写 obj_cm.pose = start_pose；部分 WRS 版本不会同步 pos/rotmat，
        # 会导致模型在动画/碰撞里跑到世界原点。这里显式写入真实 layout 起始位姿。
        obj_cm.pos = np.asarray(start_pose[0], dtype=float)
        obj_cm.rotmat = np.asarray(start_pose[1], dtype=float)
        last_err = ""

        for sender_tag, receiver_tag, sender_arm, receiver_arm in self._handover_arm_pairs(pid):
            planner = _QuietHandoverPlanner(
                obj_cmodel=obj_cm,
                sender_robot=sender_arm,
                receiver_robot=receiver_arm,
                sender_reference_gc=gc,
                receiver_reference_gc=_duplicate_grasp_collection(gc),
            )
            try:
                planner.add_hopg_collection_from_disk(self._middle_plate_hopg)
            except Exception as e:
                last_err = f"{pid} load hopg failed: {e!r}"
                print(f"  [NO] {last_err}")
                continue

            try:
                motion_list = planner.plan_by_obj_poses(
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    obstacle_list=obstacle_list,
                    toggle_dbg=False,
                )
            except Exception as e:
                last_err = (
                    f"{pid} handover {sender_tag}->{receiver_tag}: "
                    f"exception {type(e).__name__}: {e!r}"
                )
                print(f"  [NO] {last_err}")
                continue

            if motion_list is None:
                last_err = f"{pid} handover {sender_tag}->{receiver_tag}: no path"
                print(f"  [NO] {last_err}")
                continue

            _apply_motion_end_states(motion_list)
            anim_md = _merge_motion_mesh_list(motion_list)
            _attach_moving_object_overlay_to_mesh_list(
                anim_md.mesh_list,
                self.asm,
                pid,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cdprim_type=self.cdprim_type,
                obj_pose_list=_concat_obj_pose_lists(motion_list),
            )
            print(
                f"  [OK/handover] pid={pid:14s} {sender_tag}->{receiver_tag} "
                f"frames={len(anim_md.mesh_list)}"
            )
            return StepMotion(
                step_id=-1,
                part_id=pid,
                arm_tag=f"{sender_tag}+{receiver_tag}",
                motion_tag=f"handover_{sender_tag}_to_{receiver_tag}",
                mot_data=anim_md,
            ), ""

        return None, last_err or f"{pid}: handover failed"

    def _try_middle_plate_handover_step(
        self,
        pid: str,
        placed: set,
    ) -> Tuple[Optional[StepMotion], str]:
        if pid not in self.staging_models or pid not in self.world_poses:
            return None, f"{pid}: missing staging/goal for handover"

        gc = self.grasps.get(pid)
        if gc is None or len(gc) == 0:
            return None, f"{pid}: grasp collection missing for handover"

        st = self.staging_models[pid]
        start_pose = (np.asarray(st.pos, dtype=float), np.asarray(st.rotmat, dtype=float))
        gp, gr = self.world_poses[pid]
        goal_pose = (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))
        obstacle_list = self._current_obstacles(pid, placed)

        meta = getattr(self.layout, "metadata", {}) or {}
        preferred_arm = (meta.get("arm_choice", {}) or {}).get(pid)
        print(f"\n[HANDOVER] pid={pid} 直接换手（跳过单臂） preferred_sender={preferred_arm}")
        print(f"  hopg = {self._middle_plate_hopg}")

        return self._try_middle_plate_handover(
            pid, start_pose, goal_pose, gc, obstacle_list
        )

    def _try_plan_step(self, pid: str, placed: set):
        if pid not in self.staging_models:
            return None, f"{pid}: staging model missing"

        if pid not in self.world_poses:
            return None, f"{pid}: goal world pose missing"

        gc = self.grasps.get(pid)
        if gc is None or len(gc) == 0:
            return None, f"{pid}: grasp collection missing or empty"

        if self._uses_handover_direct(pid):
            return self._try_middle_plate_handover_step(pid, placed)

        obj_cm = self.staging_models[pid]
        gp, gr = self.world_poses[pid]

        meta = getattr(self.layout, "metadata", {}) or {}
        preferred_arm = (meta.get("arm_choice", {}) or {}).get(pid)
        arm_order = _arm_try_order(pid, preferred_arm)

        last_err = ""
        print(f"\n[PLAN] pid={pid}, preferred_arm={preferred_arm}, try_order={arm_order}")

        for arm_tag in arm_order:
            arm = self.robot.rgt_arm if arm_tag == "rgt" else self.robot.lft_arm
            transport = TransportPrimitive(arm)

            for motion_tag, motion_kwargs in _motion_candidate_kwargs(pid, arm_tag):
                # 每次尝试复制一个 moving object，避免失败污染 staging model
                moving = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
                moving.pos = obj_cm.pos.copy()
                moving.rotmat = obj_cm.rotmat.copy()
                moving._sealp_part_id = pid
                moving._sealp_role = "moving_object"

                obs = self._current_obstacles(pid, placed)

                try:
                    res = transport.plan(
                        obj_cmodel=moving,
                        grasp_collection=gc,
                        goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                        obstacle_list=obs,
                        approach_distance=APPROACH_DIST,
                        depart_distance=PICK_DEPART_DIST,
                        linear_granularity=LINEAR_GRANULARITY,
                        **motion_kwargs,
                    )
                except Exception as e:
                    last_err = (
                        f"{pid} {arm_tag} motion={motion_tag}: "
                        f"exception {type(e).__name__}: {e!r}"
                    )
                    print(f"  [NO] {last_err}")
                    continue

                if not bool(getattr(res, "success", False)):
                    msg = getattr(res, "error_msg", "") or "no valid plan"
                    last_err = f"{pid} {arm_tag} motion={motion_tag}: {msg}"
                    print(f"  [NO] {last_err}")
                    continue

                print(
                    f"  [OK] pid={pid:14s} arm={arm_tag} motion={motion_tag} "
                    f"obs={len(obs)} frames={len(res.mot_data.mesh_list) if getattr(res, 'mot_data', None) else 0}"
                )

                # 给每一帧补当前零件的真实移动显示：
                # 起点来自 layout.staging_positions，终点来自 asmdef/world_poses。
                # 这样动画里物体不会隐身，也不会莫名其妙出现在原点。
                if getattr(res, "mot_data", None) is not None:
                    _attach_moving_object_overlay_to_mesh_list(
                        getattr(res.mot_data, "mesh_list", []) or [],
                        self.asm,
                        pid,
                        start_pose=(obj_cm.pos.copy(), obj_cm.rotmat.copy()),
                        goal_pose=(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float)),
                        cdprim_type=self.cdprim_type,
                        obj_pose_list=_extract_obj_pose_list(res.mot_data),
                    )

                # 更新该臂末端关节状态
                try:
                    arm.goto_given_conf(res.end_jnt_values)
                except Exception:
                    try:
                        arm.robot.goto_given_conf(res.end_jnt_values)
                    except Exception:
                        pass

                sm = StepMotion(
                    step_id=-1,
                    part_id=pid,
                    arm_tag=arm_tag,
                    motion_tag=motion_tag,
                    mot_data=res.mot_data,
                )
                return sm, ""

        return None, last_err or f"{pid}: all arm/motion candidates failed"

    def _backup_robot_state(self):
        """备份左右臂状态。失败后恢复，避免失败尝试污染后续步骤。"""
        for arm in (self.robot.lft_arm, self.robot.rgt_arm):
            try:
                arm.backup_state()
            except Exception:
                pass

    def _restore_robot_state(self):
        """恢复左右臂状态。"""
        for arm in (self.robot.lft_arm, self.robot.rgt_arm):
            try:
                arm.restore_state()
            except Exception:
                pass

    def _attach_assembled_solid(self, pid: str, rgba=None):
        """把已经装好的零件以实心模型显示在 goal pose。

        用于 preassembled 件，例如 base_plate。
        """
        if pid not in self.goal_models:
            return None
        try:
            solid = self.goal_models[pid].copy()
        except Exception:
            solid = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
            gp, gr = self.world_poses[pid]
            solid.pos = np.asarray(gp, dtype=float)
            solid.rotmat = np.asarray(gr, dtype=float)

        try:
            solid.rgba = np.asarray(rgba if rgba is not None else [0.35, 0.75, 0.45, 0.88], dtype=float)
        except Exception:
            pass

        solid._sealp_part_id = pid
        solid._sealp_role = "assembled_at_goal_visual"
        solid.attach_to(self.base)
        return solid

    def execute_until_failure(self) -> ExecutionSummary:
        """按 asmdef 顺序尽力规划所有步骤。

        与旧版不同：
            - 某一步失败后不会停止；
            - 失败零件不会加入 placed，也不会从 staging 可视化中移除；
            - 后续零件仍继续尝试规划；
            - 最终播放所有成功规划出来的步骤动画。

        动态障碍物规则仍然保持：
            已成功装配的零件 -> goal pose，作为障碍；
            未成功装配的零件 -> staging pose，作为障碍；
            当前正在尝试的零件 -> 不作为静态障碍。
        """
        print("========== Layout Sequence Execute All Possible Steps ==========")
        print(f"assembly_station = {np.round(self.fixture_pos, 4).tolist()}")
        print(f"part_order       = {self.part_order}")
        print(f"cdprim_type      = {self.cdprim_type}")
        print("continue_on_fail = True")

        placed = set()
        success_steps: List[StepMotion] = []
        failed_steps: List[FailedStep] = []

        attach_goal_ghosts(self.base, self.asm, self.world_poses, self.cdprim_type)
        self.staging_visuals = attach_staging_visuals(
            self.base, self.asm, self.layout, self.part_order, self.cdprim_type
        )

        self.robot.gen_meshmodel(alpha=0.25).attach_to(self.base)

        step_id_lookup = {s.part_id: s.step_id for s in self.asm.steps}

        for step in self.asm.steps:
            pid = step.part_id
            if pid not in self.part_order:
                continue

            sid = getattr(step, "step_id", step_id_lookup.get(pid, len(success_steps)))

            if _is_preassembled(pid, self.layout):
                print(
                    f"[SKIP/PREASSEMBLED] step={sid} pid={pid}: "
                    f"layout 中标记为已装配，直接加入 goal 动态障碍，并以实心模型显示。"
                )
                if pid in self.staging_visuals:
                    try:
                        self.staging_visuals[pid].detach()
                    except Exception:
                        pass

                self._attach_assembled_solid(
                    pid,
                    rgba=np.array([0.45, 0.70, 0.45, 0.92]),
                )

                placed.add(pid)
                continue

            # 关键修改：该 step 失败后，恢复机械臂状态，然后继续尝试后续零件。
            self._backup_robot_state()
            sm, err = self._try_plan_step(pid, placed)

            if sm is None:
                self._restore_robot_state()

                print("" + "=" * 70)
                print(f"[FAIL/CONTINUE] step={sid} pid={pid} 规划失败，但继续尝试后续零件。")
                print(f"reason: {err}")
                print("说明：该零件仍保留在 staging 位置，不加入 placed；后续规划会继续把它当作未装零件障碍。")
                print("=" * 70)

                failed_steps.append(FailedStep(
                    step_id=sid,
                    part_id=pid,
                    reason=str(err),
                ))
                continue

            sm.step_id = sid
            success_steps.append(sm)
            print(f"[SUCCESS] step={sid} pid={pid} 规划成功，加入已装配集合。")

            if pid in self.staging_visuals:
                try:
                    self.staging_visuals[pid].detach()
                except Exception:
                    pass

            placed.add(pid)

            if pid in self.goal_models:
                try:
                    solid = self.goal_models[pid].copy()
                except Exception:
                    solid = self.goal_models[pid]
                try:
                    solid.rgba = np.array([0.25, 0.85, 0.35, 0.70])
                    solid.attach_to(self.base)
                except Exception:
                    pass

        print("" + "=" * 70)
        print("[DONE] 已遍历所有步骤。")
        print(f"成功步骤数: {len(success_steps)}")
        print(f"失败步骤数: {len(failed_steps)}")
        print("=" * 70)

        if failed_steps:
            last = failed_steps[-1]
            return ExecutionSummary(
                success_steps=success_steps,
                failed_steps=failed_steps,
                failed_step_id=last.step_id,
                failed_part_id=last.part_id,
                failed_reason=last.reason,
            )

        return ExecutionSummary(
            success_steps=success_steps,
            failed_steps=[],
        )


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Read .layout, execute assembly sequence with dynamic obstacles, and visualize successful motions."
    )
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help="asmdef 文件路径")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help=".layout 文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml")
    parser.add_argument("--grasp-dir", default=DEFAULT_GRASP_DIR, help="grasp pickle 文件夹")
    parser.add_argument(
        "--grasp-map-json",
        default="",
        help="可选，显式 grasp 映射，例如 '{\"post\":\"tower_post_grasps.pickle\"}'",
    )
    parser.add_argument(
        "--cdprim-type",
        default=DEFAULT_CDPRIM_TYPE,
        help="碰撞模型类型，默认 triangles(mesh)。想加速可以填 box / convex_hull。",
    )
    parser.add_argument(
        "--contact-exclusion-json",
        default="",
        help=(
            "可选，额外接触豁免表。规划某零件时临时排除这些已装件，"
            "例如 '{\"top_cross\":[\"middle_plate\"]}'。direct parent 会自动排除。"
        ),
    )
    parser.add_argument(
        "--no-middle-plate-regrasp",
        action="store_true",
        help="middle_plate 改回单臂 pick-place（默认直接换手）",
    )
    parser.add_argument(
        "--handover-dir",
        default=DEFAULT_HANDOVER_DIR,
        help="middle_plate 换手 hopg 目录",
    )
    parser.add_argument("--no-auto-play", action="store_true", help="不自动播放，按 SPACE 逐帧播放")
    parser.add_argument("--interval", type=float, default=0.02, help="动画播放间隔")
    return parser.parse_args()


def main():
    args = _parse_args()

    _patch_rrt()

    asmdef_path = os.path.abspath(args.asmdef)
    layout_path = os.path.abspath(args.layout)
    config_path = os.path.abspath(args.config)
    grasp_dir = os.path.abspath(args.grasp_dir)

    if not os.path.isfile(asmdef_path):
        raise FileNotFoundError(f"asmdef 不存在: {asmdef_path}")
    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"layout 不存在: {layout_path}")

    print("=" * 78)
    print("Execute Layout Sequence Visualizer [v5 middle_plate handover]")
    print(f"asmdef     = {asmdef_path}")
    print(f"layout     = {layout_path}")
    print(f"config     = {config_path}")
    print(f"grasp_dir  = {grasp_dir}")
    print(f"cdprim     = {args.cdprim_type}  # 默认 mesh/triangles")
    print(f"middle_plate_handover = {not args.no_middle_plate_regrasp}")
    if not args.no_middle_plate_regrasp:
        print(f"handover_dir = {os.path.abspath(args.handover_dir)}")
    print("=" * 78)

    asm = AssemblyDef.load(asmdef_path)
    layout = WorkspaceLayout.load(layout_path)

    print(f"Loaded asm    : {asm.name}, n_parts={asm.n_parts}, n_steps={asm.n_steps}")
    print(f"Loaded layout : {layout.name}")
    print(f"assembly pos  : {np.round(layout.assembly_station_pos, 4).tolist()}")
    print(f"metadata keys : {list((layout.metadata or {}).keys())}")

    base = wd.World(
        cam_pos=[1.05, -1.25, 0.85],
        lookat_pos=np.asarray(layout.assembly_station_pos, dtype=float) + np.array([0, 0, 0.10]),
    )
    mgm.gen_frame(pos=np.asarray(layout.assembly_station_pos, dtype=float), ax_length=0.10).attach_to(base)

    runner = LayoutSequenceVisualizer(
        asm=asm,
        layout=layout,
        config_path=config_path,
        grasp_dir=grasp_dir,
        base=base,
        cdprim_type=args.cdprim_type,
        grasp_map=_load_json_map(args.grasp_map_json),
        contact_exclusion_map=_load_json_map(args.contact_exclusion_json),
        enable_middle_plate_regrasp=not args.no_middle_plate_regrasp,
        handover_dir=os.path.abspath(args.handover_dir),
    )

    summary = runner.execute_until_failure()

    print("\n========== 执行总结 ==========")
    print(f"成功步骤数: {len(summary.success_steps)}")
    for sm in summary.success_steps:
        tag = sm.motion_tag or ""
        if tag.startswith("handover"):
            via = "handover"
        elif tag.startswith("regrasp"):
            via = "regrasp"
        else:
            via = "single-arm"
        print(
            f"  [OK]   step={sm.step_id:2d} pid={sm.part_id:14s} "
            f"arm={sm.arm_tag} motion={sm.motion_tag} via={via}"
        )

    print(f"失败步骤数: {len(summary.failed_steps)}")
    for fs in summary.failed_steps:
        print(
            f"  [FAIL] step={fs.step_id:2d} pid={fs.part_id:14s} "
            f"reason={fs.reason}"
        )

    if not summary.failed_steps:
        print("\n全部步骤成功。")
    else:
        print("\n注意：失败步骤不会播放动画；成功步骤会继续播放。")

    animate_success_steps(
        base,
        summary.success_steps,
        interval=float(args.interval),
        auto_play=not args.no_auto_play,
    )

    base.run()


if __name__ == "__main__":
    main()