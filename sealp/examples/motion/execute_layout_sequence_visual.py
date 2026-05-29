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
   - 同时显示初始布局、目标 ghost、已成功装配的零件。
4. middle_plate **直接走双臂换手**（跳过单臂 pick-place，需预生成 hopg）：
   - ``tower_handover/middle_plate_hopg.pickle``
   - 生成命令：``python -m sealp.examples.grasp.gen_middle_plate_regrasp_data``
5. 默认使用 mesh/triangles 碰撞，不用 box。
   - 想加速可以用 --cdprim-type box。
6. 运输障碍与落位接触豁免分离：
   - middle_plate 运输过程中会把四根柱子作为动态障碍，减少动画穿模；
   - 最终落位/插接附近仍保留必要的接触豁免。
7. 动画效果仿照 LRMate200id_ppp_animation.py：
   - 初始零件留在 layout 位置；
   - 抓住后由夹爪动画接管；
   - 放下后固定到目标位置；
   - 每帧补显示另一只手臂，避免“用到谁才出现谁”。
8. 全部步骤成功后保存运动路径 pkl，下次可直接缓存回放：
   - 默认缓存路径：sealp/examples/motion/_output/{layout名}_motions.pkl
   - 强制重算：--replan

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
import hashlib
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
DEFAULT_MOTION_CACHE_DIR = os.path.join(SEALP_ROOT, "examples", "motion", "_output")
MOTION_CACHE_FORMAT_VERSION = "2.2"

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
    # 用于保存/重建轨迹缓存：单臂通常是 [res.mot_data]，换手是 motion_list。
    motion_segments: Optional[List] = None
    # 每个 segment 对应的逐帧物体位姿 [(pos, rotmat) | None]。
    # 在 _add_other_arm_to_motion_* 把对侧手臂塞进帧 cm_list 之前抢先抽出来，
    # 这样后续 cache 重放时不再用 staging→goal 直线插值假装物体。
    obj_pose_per_segment: Optional[List[List[Optional[Tuple[np.ndarray, np.ndarray]]]]] = None


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
    mesh_list = []
    for md in motion_list:
        mesh_list.extend(getattr(md, "mesh_list", []) or [])
    return _AnimMotionData(mesh_list=mesh_list)


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


def _robot_arm_side(robot) -> str:
    """尽量判断 MotionData 对应的是左臂还是右臂。"""
    name = str(getattr(robot, "name", "") or "").lower()
    if "rgt" in name or "right" in name:
        return "rgt"
    if "lft" in name or "left" in name:
        return "lft"

    try:
        pos_y = float(np.asarray(getattr(robot, "pos", np.zeros(3)), dtype=float)[1])
        return "rgt" if pos_y < 0 else "lft"
    except Exception:
        return "lft"


def _get_arm_jv(arm) -> np.ndarray:
    try:
        return np.asarray(arm.get_jnt_values(), dtype=float)
    except Exception:
        return np.asarray(HOME_JV, dtype=float)


def _get_arm_ee(arm):
    try:
        return arm.get_ee_values()
    except Exception:
        return None


def _goto_arm(arm, jv, ee=None) -> None:
    try:
        if ee is not None:
            arm.goto_given_conf(np.asarray(jv, dtype=float), ee_values=ee)
        else:
            arm.goto_given_conf(np.asarray(jv, dtype=float))
    except TypeError:
        try:
            arm.goto_given_conf(np.asarray(jv, dtype=float))
        except Exception:
            pass
    except Exception:
        pass


def _gen_arm_mesh(arm, alpha: float = 0.35):
    """生成单臂 mesh。只作为补充显示，不参与碰撞。"""
    try:
        return arm.gen_meshmodel(alpha=alpha)
    except TypeError:
        try:
            return arm.gen_meshmodel()
        except Exception:
            return None
    except Exception:
        return None


def _extract_obj_pose_from_mesh(mesh) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从一帧 ModelCollection 里抓出代表 “被持物体” 的 cm 的世界位姿。

    必须在 ``_add_other_arm_to_motion_*`` 改写 cm_list 之前调用。规则：
    1. 任何 cm.``_sealp_part_id`` 非空者（我们自己塞过的）优先；
    2. 否则取 ``cm_list[-1]`` —— 单臂 ``gen_meshmodel`` / ``obj_cmodel_copy.attach_to``
       都把 obj 追加在末尾。
    """
    if mesh is None:
        return None
    cm_list = getattr(mesh, "cm_list", None) or []
    if not cm_list:
        return None

    target = None
    for cm in reversed(cm_list):
        if getattr(cm, "_sealp_part_id", None):
            target = cm
            break
    if target is None:
        target = cm_list[-1]

    try:
        return (
            np.asarray(target.pos, dtype=float).copy(),
            np.asarray(target.rotmat, dtype=float).copy(),
        )
    except Exception:
        return None


def _extract_obj_pose_per_frame(mesh_list: List) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """逐帧抽取 obj (pos, rotmat)；None 表示该帧无法识别。"""
    out: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    last_seen: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for mesh in mesh_list or []:
        pose = _extract_obj_pose_from_mesh(mesh)
        if pose is not None:
            last_seen = pose
            out.append(pose)
        else:
            # 抽不出就沿用上一帧；动画不会因为缺失帧而出现物体瞬移。
            out.append(last_seen)
    return out


def _add_other_arm_to_motion_mesh_list(
    mesh_list: List,
    dual_robot,
    active_arm_tag: str,
    alpha: float = 0.35,
) -> None:
    """给单臂 MotionData 的每一帧补上另一只手臂。

    原来的动画帧通常只包含“正在运动的那只手臂”，看起来会像另一只手臂突然消失。
    这里不改变原帧里的主动手臂和被抓物体，只额外把另一只手臂以当前关节角画出来。
    """
    if not mesh_list:
        return

    other = dual_robot.rgt_arm if active_arm_tag == "lft" else dual_robot.lft_arm
    other_jv = _get_arm_jv(other)
    other_ee = _get_arm_ee(other)

    try:
        other.backup_state()
    except Exception:
        pass

    try:
        for frame in mesh_list:
            if frame is None:
                continue
            _goto_arm(other, other_jv, other_ee)
            other_mesh = _gen_arm_mesh(other, alpha=alpha)
            if other_mesh is not None:
                try:
                    other_mesh.attach_to(frame)
                except Exception:
                    pass
    finally:
        try:
            other.restore_state()
        except Exception:
            pass


def _add_other_arm_to_motion_segments(motion_list: List, dual_robot, alpha: float = 0.35) -> None:
    """给换手 motion_list 的每一段补上另一只手臂，使动画始终双臂可见。"""
    if not motion_list:
        return

    lft_jv = _get_arm_jv(dual_robot.lft_arm)
    rgt_jv = _get_arm_jv(dual_robot.rgt_arm)
    lft_ee = _get_arm_ee(dual_robot.lft_arm)
    rgt_ee = _get_arm_ee(dual_robot.rgt_arm)

    for md in motion_list:
        side = _robot_arm_side(getattr(md, "robot", None))
        mesh_list = getattr(md, "mesh_list", None) or []

        if side == "rgt":
            other = dual_robot.lft_arm
            other_jv, other_ee = lft_jv, lft_ee
        else:
            other = dual_robot.rgt_arm
            other_jv, other_ee = rgt_jv, rgt_ee

        try:
            other.backup_state()
        except Exception:
            pass
        try:
            for frame in mesh_list:
                if frame is None:
                    continue
                _goto_arm(other, other_jv, other_ee)
                other_mesh = _gen_arm_mesh(other, alpha=alpha)
                if other_mesh is not None:
                    try:
                        other_mesh.attach_to(frame)
                    except Exception:
                        pass
        finally:
            try:
                other.restore_state()
            except Exception:
                pass

        # 更新主动臂末状态，给后续段作为“另一只手臂”的静态状态使用。
        jv_list = getattr(md, "jv_list", None) or []
        ev_list = getattr(md, "ev_list", None) or []
        if jv_list:
            if side == "rgt":
                rgt_jv = np.asarray(jv_list[-1], dtype=float)
                if ev_list:
                    rgt_ee = ev_list[-1]
            else:
                lft_jv = np.asarray(jv_list[-1], dtype=float)
                if ev_list:
                    lft_ee = ev_list[-1]


def _extract_ev_list(md) -> List:
    ev_list = getattr(md, "ev_list", None)
    if ev_list is None:
        ev_list = getattr(md, "ee_values_list", None)
    return list(ev_list or [])


def _serialize_ev(ev):
    if ev is None:
        return None
    try:
        if np.isscalar(ev):
            return float(ev)
    except Exception:
        pass
    try:
        return np.asarray(ev, dtype=float).tolist()
    except Exception:
        return None


def _deserialize_ev(ev):
    if ev is None:
        return None
    try:
        if isinstance(ev, (int, float)):
            return float(ev)
        return np.asarray(ev, dtype=float)
    except Exception:
        return None


def _interp_rotmat(r0, r1, t: float) -> np.ndarray:
    """两个旋转矩阵之间做近似插值，只用于缓存回放时补显示物体。"""
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    m = (1.0 - t) * r0 + t * r1
    try:
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return r
    except Exception:
        return r0 if t < 0.5 else r1


def _attach_object_at_pose_to_frame(
    frame,
    asm: AssemblyDef,
    part_id: str,
    pos: np.ndarray,
    rotmat: np.ndarray,
    cdprim_type: str,
) -> None:
    """缓存回放时按规划时记录的真实位姿，把物体贴到一帧上。"""
    if frame is None:
        return
    mesh_path = asm.model_path(part_id)
    if not mesh_path or not os.path.isfile(mesh_path):
        return

    try:
        obj = make_collision_model(mesh_path, cdprim_type=cdprim_type)
        obj.pos = np.asarray(pos, dtype=float)
        obj.rotmat = np.asarray(rotmat, dtype=float)
        obj._sealp_part_id = part_id
        obj._sealp_role = "cached_moving_object_visual"
        obj.attach_to(frame)
    except Exception:
        pass


def _attach_interpolated_object_to_frame(
    frame,
    asm: AssemblyDef,
    part_id: str,
    start_pose: Tuple[np.ndarray, np.ndarray],
    goal_pose: Tuple[np.ndarray, np.ndarray],
    t: float,
    cdprim_type: str,
) -> None:
    """缓存里没有该帧的精确 obj 位姿时（兼容旧 cache）才会用到的兜底。"""
    if frame is None:
        return
    sp, sr = start_pose
    gp, gr = goal_pose
    pos = (1.0 - t) * np.asarray(sp, dtype=float) + t * np.asarray(gp, dtype=float)
    rot = _interp_rotmat(sr, gr, t)
    _attach_object_at_pose_to_frame(
        frame, asm, part_id, pos=pos, rotmat=rot, cdprim_type=cdprim_type,
    )


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


def _safe_cm_pos(cm) -> Optional[np.ndarray]:
    if cm is None:
        return None
    try:
        return np.asarray(cm.pos, dtype=float).reshape(3)
    except Exception:
        return None


def _frame_has_cm_at(mesh, target_pos: np.ndarray, eps: float = 5e-3) -> bool:
    """判断一帧里是否有物体仍停在 staging 位置。"""
    if mesh is None:
        return False
    cm_list = getattr(mesh, "cm_list", None) or []
    for cm in cm_list:
        p = _safe_cm_pos(cm)
        if p is None:
            continue
        if float(np.linalg.norm(p - target_pos)) <= eps:
            return True
    return False


def _detect_carry_start_by_pose(mesh_list: List, staging_pos: Optional[np.ndarray], eps: float = 5e-3) -> Optional[int]:
    """仿照 LRMate 动画：从“物体不再贴在 staging”开始视为被夹爪接管。"""
    if staging_pos is None:
        return None
    staging_pos = np.asarray(staging_pos, dtype=float).reshape(3)
    for i, mesh in enumerate(mesh_list):
        if mesh is None:
            continue
        if not _frame_has_cm_at(mesh, staging_pos, eps):
            return i
    return None


def _hard_release_mesh(mesh) -> None:
    """释放上一帧。这里只 detach，不 remove，避免共享 NodePath 被提前销毁。"""
    if mesh is None:
        return
    try:
        mesh.detach()
    except Exception:
        pass


def animate_success_steps(
    base,
    step_motions: List[StepMotion],
    runner=None,
    interval: float = 0.03,  # 保留参数仅为向后兼容，已不再使用 task tick。
    auto_play: bool = False,  # 保留参数仅为向后兼容，自动播放已彻底关闭。
):
    """SPACE 驱动播放：按一次推进一帧；全部播放完后再按 SPACE 重新开始。

    动画状态机：
        PRE   ：零件保持在 layout 的 staging 初始位置；
        CARRY ：进入持物帧后，外部 staging 模型 detach，由夹爪 mesh 内的被持物体接管显示；
        DONE  ：整段动作结束后，把零件固定到目标装配位姿。

    循环播放：所有 step 播完后，恢复 staging 颜色块、清掉 assembled 实心件，
    下一次按 SPACE 重新从头播。
    """
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

    # 收掉规划阶段静态机器人，避免“静态机器人 + 动画机器人”重影。
    if runner is not None:
        static_robot_mesh = getattr(runner, "static_robot_mesh", None)
        if static_robot_mesh is not None:
            try:
                static_robot_mesh.detach()
            except Exception:
                pass
            runner.static_robot_mesh = None

    PRE, CARRY, DONE = 0, 1, 2
    phases = []

    print("\n========== 动画控制 ==========")
    print("  按 SPACE 推进一帧；全部播放完后再按 SPACE 重新开始（循环）。")
    print(f"  可播放成功步骤数：{len(motion_items)}")

    for sm in motion_items:
        mesh_list = getattr(sm.mot_data, "mesh_list", []) or []
        staging_pos = None
        if runner is not None:
            st_model = getattr(runner, "staging_models", {}).get(sm.part_id)
            staging_pos = _safe_cm_pos(st_model)
            if staging_pos is None:
                st = runner.layout.staging_positions.get(sm.part_id)
                if st is not None:
                    staging_pos = np.asarray(st[0], dtype=float).reshape(3)

        cached_cs = getattr(sm.mot_data, "cached_carry_start", None)
        if cached_cs is not None:
            cs = cached_cs
        else:
            cs = _detect_carry_start_by_pose(mesh_list, staging_pos)
            if cs is None and mesh_list:
                cs = 0

        phases.append({
            "state": PRE,
            "carry_start": cs,
            "done_solid": None,
        })
        print(
            f"  [anime] step={sm.step_id:2d} pid={sm.part_id:14s} "
            f"frames={len(mesh_list):4d} carry_start={cs} "
            f"staging_pos={None if staging_pos is None else np.round(staging_pos, 4).tolist()}"
        )

    def _detach_staging(pid: str) -> None:
        if runner is None:
            return
        vis = getattr(runner, "staging_visuals", {}).get(pid)
        if vis is not None:
            try:
                vis.detach()
            except Exception:
                pass

    def _reattach_staging(pid: str) -> None:
        if runner is None:
            return
        vis = getattr(runner, "staging_visuals", {}).get(pid)
        if vis is not None:
            try:
                vis.attach_to(base)
            except Exception:
                pass

    def _attach_done(pid: str):
        if runner is None:
            return None
        _detach_staging(pid)
        try:
            return runner._attach_assembled_solid(
                pid,
                rgba=np.array([0.25, 0.85, 0.35, 0.70]),
            )
        except Exception as e:
            print(f"[WARN] attach done visual failed for {pid}: {type(e).__name__}: {e}")
            return None

    def _detach_done_solid(solid) -> None:
        if solid is None:
            return
        try:
            solid.detach()
        except Exception:
            pass

    class _AnimeState:
        __slots__ = ("step_idx", "frame_idx", "last_attached", "finished", "loop_count")

        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.last_attached = None
            self.finished = False
            self.loop_count = 0

    st = _AnimeState()

    def _finalize_step(idx: int) -> None:
        if idx < 0 or idx >= len(motion_items):
            return
        sm = motion_items[idx]
        ph = phases[idx]
        if ph["state"] != DONE:
            ph["done_solid"] = _attach_done(sm.part_id)
            ph["state"] = DONE
        print(f"[动画] step={sm.step_id} {sm.part_id} 播放结束，零件固定到目标位。")

    def _advance_phase(idx: int, frame_idx: int) -> None:
        sm = motion_items[idx]
        ph = phases[idx]
        cs = ph["carry_start"]

        if cs is not None and ph["state"] == PRE and frame_idx >= int(cs):
            _detach_staging(sm.part_id)
            ph["state"] = CARRY
            print(
                f"[动画] step={sm.step_id} {sm.part_id} 进入 CARRY: "
                f"frame={frame_idx}, staging 模型 detach。"
            )

    def _reset_for_loop() -> None:
        """循环重置：恢复 staging 颜色块、清掉 assembled 实心件。"""
        _hard_release_mesh(st.last_attached)
        st.last_attached = None
        for idx, ph in enumerate(phases):
            sm = motion_items[idx]
            _detach_done_solid(ph.get("done_solid"))
            ph["done_solid"] = None
            ph["state"] = PRE
            _reattach_staging(sm.part_id)
        st.step_idx = 0
        st.frame_idx = 0
        st.finished = False
        st.loop_count += 1
        print(f"\n[动画] 开始第 {st.loop_count + 1} 轮播放。")

    def _show_current_frame() -> None:
        """显示当前 (step_idx, frame_idx) 帧；跳过空 mesh，越界则推进到下一 step。"""
        while st.step_idx < len(motion_items):
            sm = motion_items[st.step_idx]
            mesh_list = getattr(sm.mot_data, "mesh_list", []) or []

            if st.frame_idx >= len(mesh_list):
                _hard_release_mesh(st.last_attached)
                st.last_attached = None
                _finalize_step(st.step_idx)
                st.step_idx += 1
                st.frame_idx = 0
                continue

            mesh = mesh_list[st.frame_idx]
            if mesh is None:
                st.frame_idx += 1
                continue

            _advance_phase(st.step_idx, st.frame_idx)

            _hard_release_mesh(st.last_attached)
            st.last_attached = None

            mesh.attach_to(base)
            st.last_attached = mesh
            return

        _hard_release_mesh(st.last_attached)
        st.last_attached = None
        st.finished = True
        print(
            f"[动画] 第 {st.loop_count + 1} 轮播放完毕。再按 SPACE 重新开始。"
        )

    def _on_space():
        # 已经全部播完 → 下一次 SPACE 重置并播第一帧。
        if st.finished:
            _reset_for_loop()
            _show_current_frame()
            return

        # 首次按下 → 直接渲染第 0 帧；之后每按一次 frame_idx +1。
        if st.last_attached is None:
            _show_current_frame()
        else:
            st.frame_idx += 1
            _show_current_frame()

    base.accept("space", _on_space)


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
        self.static_robot_mesh = None

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

    def _transit_obstacles(self, current_pid: str, placed: set) -> List:
        """运输/RRT 阶段使用的完整动态障碍，不做接触豁免。

        关键修复：
            middle_plate 在移动过程中必须把四根 post 当作真实障碍物，
            否则虽然最终落位需要接触豁免，但运输路径可能直接穿过柱子。
        """
        obs = list(self.env_obstacles)

        for pid in placed:
            if pid in self.goal_models:
                obs.append(self.goal_models[pid])

        for pid in self.part_order:
            if pid == current_pid or pid in placed:
                continue
            if pid in self.staging_models:
                obs.append(self.staging_models[pid])

        return obs

    def _placement_obstacles(self, current_pid: str, placed: set) -> List:
        """抓取/落位校验用的动态障碍，允许父件/支撑件接触豁免。"""
        obs = list(self.env_obstacles)
        excluded = self._contact_exclusion_set(current_pid, placed)

        if excluded:
            print(f"    [接触豁免/落位] 当前 {current_pid!r} 规划时临时排除已装件: {sorted(excluded)}")

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

    def _current_obstacles(self, current_pid: str, placed: set) -> List:
        """兼容旧调用：默认返回落位校验障碍。"""
        return self._placement_obstacles(current_pid, placed)

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
        obj_cm.pose = start_pose
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

            # 在 _add_other_arm_to_motion_segments 改写 cm_list 之前，
            # 先抓出每个 segment 的逐帧 obj 真位姿，供 cache 重放精确复现。
            obj_pose_per_segment: List[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = []
            for md in motion_list:
                obj_pose_per_segment.append(
                    _extract_obj_pose_per_frame(getattr(md, "mesh_list", []) or [])
                )

            # 动画显示修复：每一帧额外补上另一只手臂，避免“用到谁才出现谁”。
            _add_other_arm_to_motion_segments(motion_list, self.robot)
            _apply_motion_end_states(motion_list)
            anim_md = _merge_motion_mesh_list(motion_list)
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
                motion_segments=list(motion_list),
                obj_pose_per_segment=obj_pose_per_segment,
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
        # middle_plate 运输路径必须使用完整障碍物，不能把四根柱子排除；
        # 只有最终落位/接触校验才允许豁免支撑件。
        transit_obs = self._transit_obstacles(pid, placed)
        placement_obs = self._placement_obstacles(pid, placed)

        meta = getattr(self.layout, "metadata", {}) or {}
        preferred_arm = (meta.get("arm_choice", {}) or {}).get(pid)
        print(f"\n[HANDOVER] pid={pid} 直接换手（跳过单臂） preferred_sender={preferred_arm}")
        print(f"  hopg = {self._middle_plate_hopg}")
        print(f"  transit_obs={len(transit_obs)}, placement_obs={len(placement_obs)}")

        return self._try_middle_plate_handover(
            pid, start_pose, goal_pose, gc, transit_obs
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

                # 运输阶段使用完整动态障碍；落位/插接校验才使用接触豁免障碍。
                transit_obs = self._transit_obstacles(pid, placed)
                placement_obs = self._placement_obstacles(pid, placed)

                try:
                    try:
                        res = transport.plan(
                            obj_cmodel=moving,
                            grasp_collection=gc,
                            goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                            obstacle_list=transit_obs,
                            grasp_obstacle_list=placement_obs,
                            approach_distance=APPROACH_DIST,
                            depart_distance=PICK_DEPART_DIST,
                            linear_granularity=LINEAR_GRANULARITY,
                            **motion_kwargs,
                        )
                    except TypeError:
                        # 兼容旧版 TransportPrimitive.plan：没有 grasp_obstacle_list 时仍能运行。
                        res = transport.plan(
                            obj_cmodel=moving,
                            grasp_collection=gc,
                            goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                            obstacle_list=transit_obs,
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
                    f"transit_obs={len(transit_obs)} placement_obs={len(placement_obs)} "
                    f"frames={len(res.mot_data.mesh_list) if getattr(res, 'mot_data', None) else 0}"
                )

                # 先抢在 cm_list 被对侧手臂污染前，抽出逐帧 obj 真位姿，
                # 后续 cache 重放靠它，不再做 staging→goal 直线插值。
                obj_pose_seg: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
                if getattr(res, "mot_data", None) is not None:
                    obj_pose_seg = _extract_obj_pose_per_frame(
                        getattr(res.mot_data, "mesh_list", []) or []
                    )

                # 动画显示修复：每一帧额外补上另一只手臂，避免“用到谁才出现谁”。
                if getattr(res, "mot_data", None) is not None:
                    _add_other_arm_to_motion_mesh_list(
                        getattr(res.mot_data, "mesh_list", []) or [],
                        self.robot,
                        active_arm_tag=arm_tag,
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
                    motion_segments=[res.mot_data],
                    obj_pose_per_segment=[obj_pose_seg],
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

        self.static_robot_mesh = self.robot.gen_meshmodel(alpha=0.25)
        self.static_robot_mesh.attach_to(self.base)

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

            # 注意：这里不再立刻 detach staging / attach goal。
            # 仿照 LRMate200id_ppp_animation.py：动画开始时零件仍在真实 layout 初始位置；
            # 进入 CARRY 帧时由动画状态机 detach，动作结束后再固定到 goal pose。
            placed.add(pid)

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
# 运动轨迹缓存：全部成功后保存，下次直接重建动画，不重新规划
# ============================================================

def _file_md5(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except Exception:
        return ""


def _default_motion_cache_path(layout_path: str) -> str:
    name = os.path.splitext(os.path.basename(layout_path))[0]
    return os.path.join(DEFAULT_MOTION_CACHE_DIR, f"{name}_motions.pkl")


def _motion_cache_fingerprint(
    asmdef_path: str,
    layout_path: str,
    grasp_dir: str,
    cdprim_type: str,
    enable_middle_plate_regrasp: bool,
    handover_dir: str,
    grasp_map: Optional[Dict[str, str]] = None,
    contact_exclusion_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    """用关键输入生成缓存指纹，避免 layout/asmdef 改了还误用旧路径。"""
    meta = {
        "format_version": MOTION_CACHE_FORMAT_VERSION,
        "asmdef_md5": _file_md5(asmdef_path),
        "layout_md5": _file_md5(layout_path),
        "grasp_dir": os.path.abspath(grasp_dir),
        "cdprim_type": str(cdprim_type),
        "middle_plate_handover": bool(enable_middle_plate_regrasp),
        "handover_dir": os.path.abspath(handover_dir),
        "grasp_map": grasp_map or {},
        "contact_exclusion_map": contact_exclusion_map or {},
    }
    return hashlib.md5(
        json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _serialize_obj_pose_entry(entry) -> Optional[List]:
    """把一帧的 (pos, rotmat) 变成可 pickle 的 list；None 保持 None。"""
    if entry is None:
        return None
    try:
        pos, rot = entry
        return [
            np.asarray(pos, dtype=float).tolist(),
            np.asarray(rot, dtype=float).tolist(),
        ]
    except Exception:
        return None


def _deserialize_obj_pose_entry(entry) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if entry is None:
        return None
    try:
        pos, rot = entry
        return (
            np.asarray(pos, dtype=float),
            np.asarray(rot, dtype=float),
        )
    except Exception:
        return None


def _serialize_motion_data(md, obj_pose_list: Optional[List] = None) -> dict:
    jv_list = getattr(md, "jv_list", None) or []
    ev_list = _extract_ev_list(md)
    if len(ev_list) < len(jv_list):
        ev_list = ev_list + [None] * (len(jv_list) - len(ev_list))

    # 对齐 obj_pose_list 长度。少了用 None 填，超了截掉，避免后期 zip 越界。
    obj_pose_list = list(obj_pose_list or [])
    if len(obj_pose_list) < len(jv_list):
        obj_pose_list = obj_pose_list + [None] * (len(jv_list) - len(obj_pose_list))
    elif len(obj_pose_list) > len(jv_list):
        obj_pose_list = obj_pose_list[: len(jv_list)]

    return {
        "arm_side": _robot_arm_side(getattr(md, "robot", None)),
        "jv_list": [np.asarray(jv, dtype=float).tolist() for jv in jv_list],
        "ev_list": [_serialize_ev(ev) for ev in ev_list],
        "obj_pose_list": [_serialize_obj_pose_entry(p) for p in obj_pose_list],
    }


def _infer_carry_start_for_step(sm: StepMotion, runner) -> Optional[int]:
    """保存缓存时记录 CARRY 起点，便于缓存回放仍接近 LRMate 动画效果。"""
    try:
        mesh_list = getattr(sm.mot_data, "mesh_list", []) or []
        st_model = getattr(runner, "staging_models", {}).get(sm.part_id)
        staging_pos = _safe_cm_pos(st_model)
        if staging_pos is None:
            st = runner.layout.staging_positions.get(sm.part_id)
            if st is not None:
                staging_pos = np.asarray(st[0], dtype=float).reshape(3)
        cs = _detect_carry_start_by_pose(mesh_list, staging_pos)
        return 0 if cs is None and mesh_list else cs
    except Exception:
        return None


def _serialize_step_motion(sm: StepMotion, runner=None) -> dict:
    segments = []
    src = sm.motion_segments if sm.motion_segments else [sm.mot_data]
    obj_pose_per_seg = list(sm.obj_pose_per_segment or [])
    for idx, md in enumerate(src):
        if md is None:
            continue
        if getattr(md, "jv_list", None):
            obj_pose_seg = obj_pose_per_seg[idx] if idx < len(obj_pose_per_seg) else None
            segments.append(_serialize_motion_data(md, obj_pose_list=obj_pose_seg))

    carry_start = _infer_carry_start_for_step(sm, runner) if runner is not None else None
    return {
        "step_id": int(sm.step_id),
        "part_id": sm.part_id,
        "arm_tag": sm.arm_tag,
        "motion_tag": sm.motion_tag,
        "segments": segments,
        "carry_start": carry_start,
    }


def _is_full_success(asm: AssemblyDef, layout: WorkspaceLayout, part_order: List[str],
                     summary: ExecutionSummary) -> bool:
    if summary.failed_steps:
        return False
    required = set()
    for step in asm.steps:
        pid = step.part_id
        if pid not in part_order:
            continue
        if _is_preassembled(pid, layout):
            continue
        required.add(pid)
    got = {sm.part_id for sm in summary.success_steps}
    return required.issubset(got)


def save_motion_cache(
    cache_path: str,
    fingerprint: str,
    asmdef_path: str,
    layout_path: str,
    summary: ExecutionSummary,
    runner=None,
    extra_meta: Optional[dict] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "format_version": MOTION_CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "asmdef_path": os.path.abspath(asmdef_path),
        "layout_path": os.path.abspath(layout_path),
        "meta": extra_meta or {},
        "steps": [_serialize_step_motion(sm, runner=runner) for sm in summary.success_steps],
    }
    with open(cache_path, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"[CACHE/SAVE] 全部可规划步骤成功，已保存运动路径 -> {cache_path}")
    print(f"[CACHE/SAVE] steps={len(payload['steps'])}")
    return cache_path


def load_motion_cache(cache_path: str, expected_fingerprint: str) -> Optional[dict]:
    if not cache_path or not os.path.isfile(cache_path):
        return None

    try:
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as e:
        print(f"[CACHE/LOAD] 读取缓存失败，忽略: {type(e).__name__}: {e}")
        return None

    if payload.get("format_version") != MOTION_CACHE_FORMAT_VERSION:
        print(f"[CACHE/LOAD] 缓存版本不匹配，忽略: {cache_path}")
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        print(f"[CACHE/LOAD] 缓存 fingerprint 不匹配，忽略: {cache_path}")
        return None
    if not payload.get("steps"):
        print(f"[CACHE/LOAD] 缓存为空，忽略: {cache_path}")
        return None

    print(f"[CACHE/LOAD] 命中缓存 -> {cache_path}  steps={len(payload['steps'])}")
    return payload


def _make_cached_dual_frame(
    runner,
    lft_jv: np.ndarray,
    rgt_jv: np.ndarray,
    lft_ee=None,
    rgt_ee=None,
):
    """缓存回放时用左右臂关节角 + 夹爪宽度重建一帧双臂动画。

    早期版本只设 jnt_values，gripper 始终维持默认 jaw width，回放里看不到
    夹爪开合。这里把 ev (jaw width) 也灌进去，pick 之后会合上、place 之后会松开。
    """
    try:
        _goto_arm(runner.robot.lft_arm, lft_jv, lft_ee)
        _goto_arm(runner.robot.rgt_arm, rgt_jv, rgt_ee)
        return runner.robot.gen_meshmodel(alpha=0.85)
    except Exception:
        try:
            return runner.robot.gen_meshmodel(alpha=0.85)
        except Exception:
            return None


def _rebuild_mot_data_from_segments(
    step_payload: dict,
    runner,
) -> _AnimMotionData:
    part_id = step_payload["part_id"]
    segments = step_payload.get("segments") or []

    st_model = runner.staging_models.get(part_id)
    if st_model is not None:
        start_pose = (np.asarray(st_model.pos, dtype=float), np.asarray(st_model.rotmat, dtype=float))
    else:
        st = runner.layout.staging_positions.get(part_id)
        start_pose = (np.asarray(st[0], dtype=float), np.asarray(st[1], dtype=float))

    gp, gr = runner.world_poses[part_id]
    goal_pose = (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))

    # 让不同 step 之间的缓存回放保持关节 + 夹爪宽度连续。
    lft_jv = getattr(runner, "_cache_replay_lft_jv", np.asarray(HOME_JV, dtype=float))
    rgt_jv = getattr(runner, "_cache_replay_rgt_jv", np.asarray(HOME_JV, dtype=float))
    lft_ee = getattr(runner, "_cache_replay_lft_ee", None)
    rgt_ee = getattr(runner, "_cache_replay_rgt_ee", None)

    total_frames = sum(len(seg.get("jv_list") or []) for seg in segments)
    carry_start = step_payload.get("carry_start")
    if carry_start is None:
        carry_start = 0 if total_frames > 0 else None

    mesh_list = []
    global_i = 0

    try:
        runner.robot.backup_state()
    except Exception:
        pass

    try:
        for seg in segments:
            side = seg.get("arm_side", "lft")
            jv_list = seg.get("jv_list") or []
            ev_list = seg.get("ev_list") or []
            obj_pose_list = seg.get("obj_pose_list") or []

            for local_i, jv in enumerate(jv_list):
                jv_arr = np.asarray(jv, dtype=float)
                ev = _deserialize_ev(ev_list[local_i]) if local_i < len(ev_list) else None
                if side == "rgt":
                    rgt_jv = jv_arr
                    if ev is not None:
                        rgt_ee = ev
                else:
                    lft_jv = jv_arr
                    if ev is not None:
                        lft_ee = ev

                frame = _make_cached_dual_frame(
                    runner, lft_jv, rgt_jv, lft_ee=lft_ee, rgt_ee=rgt_ee,
                )

                # 物体可视化：优先用规划时记录的逐帧真实位姿（让物体真正跟着夹爪走）；
                # 实在缺失才退回 staging→goal 直线插值兜底。
                stored_pose = None
                if local_i < len(obj_pose_list):
                    stored_pose = _deserialize_obj_pose_entry(obj_pose_list[local_i])

                if frame is not None and stored_pose is not None:
                    obj_pos, obj_rot = stored_pose
                    _attach_object_at_pose_to_frame(
                        frame,
                        runner.asm,
                        part_id,
                        pos=obj_pos,
                        rotmat=obj_rot,
                        cdprim_type=runner.cdprim_type,
                    )
                elif (
                    frame is not None
                    and carry_start is not None
                    and global_i >= int(carry_start)
                ):
                    # 兼容旧 cache（没有 obj_pose_list）。
                    denom = max(1, total_frames - int(carry_start) - 1)
                    t = (global_i - int(carry_start)) / float(denom)
                    _attach_interpolated_object_to_frame(
                        frame,
                        runner.asm,
                        part_id,
                        start_pose=start_pose,
                        goal_pose=goal_pose,
                        t=t,
                        cdprim_type=runner.cdprim_type,
                    )

                mesh_list.append(frame)
                global_i += 1
    finally:
        try:
            runner.robot.restore_state()
        except Exception:
            pass

    runner._cache_replay_lft_jv = np.asarray(lft_jv, dtype=float)
    runner._cache_replay_rgt_jv = np.asarray(rgt_jv, dtype=float)
    runner._cache_replay_lft_ee = lft_ee
    runner._cache_replay_rgt_ee = rgt_ee

    md = _AnimMotionData(mesh_list=mesh_list)
    md.cached_carry_start = carry_start
    return md


def rebuild_summary_from_cache(payload: dict, runner) -> ExecutionSummary:
    runner._cache_replay_lft_jv = np.asarray(HOME_JV, dtype=float)
    runner._cache_replay_rgt_jv = np.asarray(HOME_JV, dtype=float)
    runner._cache_replay_lft_ee = None
    runner._cache_replay_rgt_ee = None

    success_steps: List[StepMotion] = []
    for step in payload.get("steps", []):
        mot_data = _rebuild_mot_data_from_segments(step, runner)
        success_steps.append(StepMotion(
            step_id=int(step.get("step_id", -1)),
            part_id=step["part_id"],
            arm_tag=step.get("arm_tag", ""),
            motion_tag=step.get("motion_tag", ""),
            mot_data=mot_data,
            motion_segments=None,
        ))

    return ExecutionSummary(success_steps=success_steps, failed_steps=[])


def setup_scene_for_cached_playback(runner) -> None:
    """缓存回放时搭建与正常规划一致的初始场景。"""
    attach_goal_ghosts(runner.base, runner.asm, runner.world_poses, runner.cdprim_type)
    runner.staging_visuals = attach_staging_visuals(
        runner.base, runner.asm, runner.layout, runner.part_order, runner.cdprim_type
    )

    runner.static_robot_mesh = runner.robot.gen_meshmodel(alpha=0.25)
    runner.static_robot_mesh.attach_to(runner.base)

    for pid in runner.part_order:
        if _is_preassembled(pid, runner.layout):
            if pid in runner.staging_visuals:
                try:
                    runner.staging_visuals[pid].detach()
                except Exception:
                    pass
            runner._attach_assembled_solid(
                pid,
                rgba=np.array([0.45, 0.70, 0.45, 0.92]),
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
    parser.add_argument(
        "--no-auto-play",
        action="store_true",
        help="[已废弃] 当前播放固定为 SPACE 逐帧驱动，本参数保留仅为向后兼容，不再生效。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.03,
        help="[已废弃] 当前不再使用 task tick，本参数保留仅为向后兼容，不再生效。",
    )
    parser.add_argument(
        "--motion-cache",
        default="",
        help="运动轨迹缓存 pkl 路径；默认 sealp/examples/motion/_output/{layout名}_motions.pkl",
    )
    parser.add_argument(
        "--replan",
        action="store_true",
        help="忽略已有缓存，重新规划全部步骤",
    )
    parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="即使全部成功也不写入缓存",
    )
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
    # 与 LRMate200id_ppp_animation.py 保持一致：让 RRT/PPP 内部能拿到 ShowBase，
    # 从而生成完整 mesh_list，避免动画段缺帧或直接跳过。
    sys.modules['__main__'].base = base
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

    grasp_map = _load_json_map(args.grasp_map_json)
    contact_exclusion_map = _load_json_map(args.contact_exclusion_json)
    cache_path = os.path.abspath(args.motion_cache) if args.motion_cache else _default_motion_cache_path(layout_path)
    fingerprint = _motion_cache_fingerprint(
        asmdef_path=asmdef_path,
        layout_path=layout_path,
        grasp_dir=grasp_dir,
        cdprim_type=args.cdprim_type,
        enable_middle_plate_regrasp=not args.no_middle_plate_regrasp,
        handover_dir=os.path.abspath(args.handover_dir),
        grasp_map=grasp_map,
        contact_exclusion_map=contact_exclusion_map,
    )
    print(f"motion_cache = {cache_path}")
    print(f"cache_fp     = {fingerprint}")

    summary: ExecutionSummary
    loaded_from_cache = False
    if not args.replan:
        cached = load_motion_cache(cache_path, fingerprint)
        if cached is not None:
            summary = rebuild_summary_from_cache(cached, runner)
            setup_scene_for_cached_playback(runner)
            loaded_from_cache = True
            print("[CACHE/LOAD] 跳过规划，直接使用缓存关节路径重建动画。")

    if not loaded_from_cache:
        summary = runner.execute_until_failure()
        if (
            not args.no_save_cache
            and _is_full_success(asm, layout, runner.part_order, summary)
        ):
            save_motion_cache(
                cache_path=cache_path,
                fingerprint=fingerprint,
                asmdef_path=asmdef_path,
                layout_path=layout_path,
                summary=summary,
                runner=runner,
                extra_meta={
                    "cdprim_type": args.cdprim_type,
                    "middle_plate_handover": not args.no_middle_plate_regrasp,
                },
            )
        elif summary.failed_steps:
            print("[CACHE/SAVE] 存在失败步骤，不写入缓存。")
        elif args.no_save_cache:
            print("[CACHE/SAVE] 已跳过保存（--no-save-cache）。")

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
        runner=runner,
    )

    base.run()


if __name__ == "__main__":
    main()