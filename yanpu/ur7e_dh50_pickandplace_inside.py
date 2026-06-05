import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from trac_ik import TracIK as _TracIK

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wrs import wd, mgm
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.motion.probabilistic import multi_arm_rrt_connect as marrtc
from wrs.robot_sim.end_effectors.grippers.dh50.dh50 import Dh50
from wrs.robot_sim.robots.ur7e.dual_ur7e_dh50 import DualUR7EDH50


MESH_DIR = os.path.join(project_root, "old_version", "rbt", "ur7e", "meshes")
MODEL_DIR = os.path.join(project_root, "yanpu", "models")
IK_TIMING_PROGRESS_INTERVAL = 10.0
IK_TIMING_SLOW_TARGET_COUNT = 8

from yanpu.ur7e_dh50_pickandplace_params import (
    BOX1_CENTER,
    BOX1_PART_OFFSETS,
    BOX2_CENTER,
    BOX2_PART_OFFSETS,
    BOX_RGBA,
    DUAL_PICK_PLACE_SPECS,
    OBJECT_SPECS,
    PLACE_APPROACH_DISTANCE,
    PICK_LIFT_HEIGHT,
    RACK_ARM_Y_OFFSET,
    RACK_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH,
    RACK_BASE_POS,
    RACK_HORIZONTAL_FRAME_ALPHA,
    RACK_HORIZONTAL_FRAME_RGB,
    RACK_HORIZONTAL_FRAME_THICKNESS,
    RACK_HORIZONTAL_FRAME_X_LENGTH,
    RACK_HORIZONTAL_FRAME_Y_LENGTH,
    RACK_LFT_ARM_LOC_ROTMAT,
    RACK_RGT_ARM_LOC_ROTMAT,
    RACK_ROT,
    RACK_VERTICAL_FRAME_ALPHA,
    RACK_VERTICAL_FRAME_HEIGHT,
    RACK_VERTICAL_FRAME_RGB,
    RACK_VERTICAL_FRAME_X_LENGTH,
    RACK_VERTICAL_FRAME_XY,
    RACK_VERTICAL_FRAME_Y_LENGTH,
    RRT_TCP_Z_AXIS_MAX_ANGLE,
    RRT_TCP_Z_AXIS_WORLD,
    UR3_DUAL_LFT_HOME_CONF,
    UR3_DUAL_RGT_HOME_CONF,
)


@dataclass
class FrameState:
    conf_dict: dict
    jaw_width_dict: dict
    payload_mode_dict: dict


@dataclass
class PickPlaceTask:
    arm_name: str
    object_name: str
    grasp_index: int
    pick_pose_index: int
    symmetry_angle: float
    pick_conf: np.ndarray
    lift_conf: np.ndarray
    pre_place_conf: np.ndarray
    place_conf: np.ndarray
    jaw_width: float
    pick_pose: tuple
    lift_pose: tuple
    pre_place_pose: tuple
    place_pose: tuple
    payload_rel_pose: tuple
    pick_solution_type: str
    lift_solution_type: str
    pre_place_solution_type: str
    place_solution_type: str


class AnimationData:

    def __init__(self, robot, robot_mesh_list, frame_list, payload_dict, task_dict):
        self.robot = robot
        self.robot_mesh_list = robot_mesh_list
        self.frame_list = frame_list
        self.payload_dict = payload_dict
        self.task_dict = task_dict
        self.counter = 0
        self.current_robot_mesh = None
        self.end_hold_counter = 0


class PickPlacePlanningError(RuntimeError):

    def __init__(self, message, conf_dict=None, arm_name=None, object_name=None, grasp_indices=None):
        super().__init__(message)
        self.conf_dict = {} if conf_dict is None else conf_dict
        self.arm_name = arm_name
        self.object_name = object_name
        self.grasp_indices = [] if grasp_indices is None else list(grasp_indices)


class MultiArmPlanningError(RuntimeError):

    def __init__(self, message, debug_info=None):
        super().__init__(message)
        self.debug_info = {} if debug_info is None else debug_info


def make_collision_model(mesh_name,
                         pos=np.zeros(3),
                         rotmat=np.eye(3),
                         rgba=None,
                         ex_radius=.001,
                         attach_to=None,
                         mesh_dir=MESH_DIR):
    mesh_path = os.path.join(mesh_dir, mesh_name)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    cmodel = mcm.CollisionModel(mesh_path,
                                cdprim_type=mcm.const.CDPrimType.AABB,
                                ex_radius=ex_radius)
    cmodel.pos = np.asarray(pos, dtype=float)
    cmodel.rotmat = rotmat
    if rgba is not None:
        cmodel.rgba = np.asarray(rgba, dtype=float)
    if attach_to is not None:
        cmodel.attach_to(attach_to)
    return cmodel


def load_grasp_info_list(spec):
    pickle_path = os.path.join(MODEL_DIR, spec["grasp_pickle"])
    with open(pickle_path, "rb") as f:
        grasp_info_dict = pickle.load(f)
    return grasp_info_dict[spec["grasp_key"]]


def _new_ik_timing_stats(arm_name, object_name):
    return {
        "arm_name": arm_name,
        "object_name": object_name,
        "wall_start": time.perf_counter(),
        "target_calls": 0,
        "target_successes": 0,
        "target_failures": 0,
        "seed_trials": 0,
        "ik_none": 0,
        "collision_rejects": 0,
        "pose_rejects": 0,
        "duplicate_rejects": 0,
        "accepted_confs": 0,
        "ik_time": 0.0,
        "fk_time": 0.0,
        "collision_time": 0.0,
        "pose_check_time": 0.0,
        "solve_time": 0.0,
        "stage_stats": {},
        "slow_targets": [],
    }


def _target_stage(target_label):
    for stage in ("pre-place", "place", "lift", "pick"):
        if f" {stage} " in f" {target_label} ":
            return stage
    return "target"


def _stage_timing_stats(timing_stats, stage):
    stage_stats = timing_stats["stage_stats"].setdefault(stage, {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "time": 0.0,
        "seed_trials": 0,
        "accepted_confs": 0,
    })
    return stage_stats


def _fmt_seconds(seconds):
    return f"{seconds:.3f}s"


def _print_ik_timing_progress(timing_stats,
                              candidate_attempt_count,
                              total_candidate_count,
                              pick_failure_count,
                              lift_failure_count,
                              pre_place_failure_count,
                              place_failure_count):
    elapsed = time.perf_counter() - timing_stats["wall_start"]
    print(f"{timing_stats['arm_name']}: search progress "
          f"{candidate_attempt_count}/{total_candidate_count} candidates, elapsed={_fmt_seconds(elapsed)}, "
          f"failures pick/lift/pre-place/place="
          f"{pick_failure_count}/{lift_failure_count}/{pre_place_failure_count}/{place_failure_count}, "
          f"seed_trials={timing_stats['seed_trials']}, "
          f"tracik={_fmt_seconds(timing_stats['ik_time'])}, "
          f"fk={_fmt_seconds(timing_stats['fk_time'])}, "
          f"collision={_fmt_seconds(timing_stats['collision_time'])}.")


def _print_ik_timing_summary(timing_stats, total_candidate_count, selected_task=None):
    elapsed = time.perf_counter() - timing_stats["wall_start"]
    print(f"{timing_stats['arm_name']}: IK timing summary for {timing_stats['object_name']}: "
          f"elapsed={_fmt_seconds(elapsed)}, candidates={total_candidate_count}, "
          f"target_calls={timing_stats['target_calls']}, successes={timing_stats['target_successes']}, "
          f"failures={timing_stats['target_failures']}, seed_trials={timing_stats['seed_trials']}.")
    print("  time breakdown: "
          f"tracik={_fmt_seconds(timing_stats['ik_time'])}, "
          f"fk={_fmt_seconds(timing_stats['fk_time'])}, "
          f"collision={_fmt_seconds(timing_stats['collision_time'])}, "
          f"pose_check={_fmt_seconds(timing_stats['pose_check_time'])}, "
          f"solve_total={_fmt_seconds(timing_stats['solve_time'])}.")
    print("  rejects: "
          f"ik_none={timing_stats['ik_none']}, "
          f"collision={timing_stats['collision_rejects']}, "
          f"pose_error={timing_stats['pose_rejects']}, "
          f"duplicate={timing_stats['duplicate_rejects']}, "
          f"accepted_confs={timing_stats['accepted_confs']}.")
    for stage, stage_stats in timing_stats["stage_stats"].items():
        print(f"  stage {stage}: calls={stage_stats['calls']}, "
              f"successes={stage_stats['successes']}, failures={stage_stats['failures']}, "
              f"seed_trials={stage_stats['seed_trials']}, accepted_confs={stage_stats['accepted_confs']}, "
              f"time={_fmt_seconds(stage_stats['time'])}.")
    slow_targets = sorted(timing_stats["slow_targets"], reverse=True)[:IK_TIMING_SLOW_TARGET_COUNT]
    if slow_targets:
        print("  slowest IK targets:")
        for elapsed_target, stage, label, seed_trials, accepted_count in slow_targets:
            print(f"    {_fmt_seconds(elapsed_target)} {stage}: {label} "
                  f"(seed_trials={seed_trials}, accepted_confs={accepted_count})")
    if selected_task is not None:
        print(f"  selected: pick_pose=#{selected_task.pick_pose_index}, "
              f"symmetry={np.degrees(selected_task.symmetry_angle):.1f}deg, "
              f"grasp=#{selected_task.grasp_index}.")


def print_tracik_status(robot):
    print(f"TracIK top-level import: {bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__}).")
    for arm_name, arm in robot.arm_dict.items():
        solver_cache_size = len(getattr(arm, "iksolver_cache", {}))
        print(f"{arm.name} ({arm_name}): prefer_tracik={getattr(arm, '_prefer_tracik', False)}, "
              f"solver_cache={solver_cache_size} before first IK.")


def attach_position_markers(base, positions, rgb, alpha, radius):
    for pos in positions:
        mgm.gen_sphere(pos=np.asarray(pos),
                       radius=radius,
                       rgb=np.asarray(rgb),
                       alpha=alpha).attach_to(base)


def attach_grasp_previews(base, spec):
    obj_pos, obj_rotmat = spec["pick_pose"]
    grasp_info_list = load_grasp_info_list(spec)
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
        gl_jaw_center_pos = obj_pos + obj_rotmat @ jaw_center_pos
        gl_jaw_center_rotmat = obj_rotmat @ jaw_center_rotmat
        gripper = Dh50()
        gripper.grip_at_by_pose(jaw_center_pos=gl_jaw_center_pos,
                                jaw_center_rotmat=gl_jaw_center_rotmat,
                                jaw_width=jaw_width)
        gripper.gen_meshmodel(rgb=spec["rgba"][:3],
                              alpha=.24,
                              toggle_tcp_frame=True).attach_to(base)


def attach_grasp_debug_previews(base, spec, grasp_indices):
    obj_pos, obj_rotmat = spec["pick_pose"]
    grasp_info_list = load_grasp_info_list(spec)
    for grasp_index in grasp_indices:
        grasp_info = grasp_info_list[grasp_index]
        jaw_width, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
        gl_jaw_center_pos = obj_pos + obj_rotmat @ jaw_center_pos
        gl_jaw_center_rotmat = obj_rotmat @ jaw_center_rotmat
        mgm.gen_sphere(pos=gl_jaw_center_pos,
                       radius=.022,
                       rgb=np.array([1.0, .08, .04]),
                       alpha=.82).attach_to(base)
        gripper = Dh50()
        gripper.grip_at_by_pose(jaw_center_pos=gl_jaw_center_pos,
                                jaw_center_rotmat=gl_jaw_center_rotmat,
                                jaw_width=jaw_width)
        gripper.gen_meshmodel(rgb=np.array([1.0, .08, .04]),
                              alpha=.52,
                              toggle_tcp_frame=True).attach_to(base)


def attach_place_object_previews(base, spec, alpha=.18):
    rgba = np.asarray(spec["rgba"], dtype=float).copy()
    rgba[3] = alpha
    for place_pose in spec["place_poses"]:
        make_collision_model(spec["mesh"],
                             pos=place_pose[0],
                             rotmat=place_pose[1],
                             rgba=rgba,
                             attach_to=base,
                             mesh_dir=MODEL_DIR)


def build_inside_scene(base):
    static_obstacles = [
        make_collision_model("pengzhuang1.STL",
                             pos=np.array([-0.13287, -0.36, 0.753]),
                             rgba=np.array([.5, .5, .5, .22]),
                             attach_to=base),
        make_collision_model("pengzhuang2.STL",
                             pos=np.array([-0.095, 0.202, 0.683]),
                             rgba=np.array([.5, .5, .5, .22]),
                             attach_to=base),
        make_collision_model("cdprimit1.STL",
                             pos=np.array([0.155, -0.905, 0.0]),
                             rotmat=rm.rotmat_from_euler(0, 0, math.pi),
                             rgba=np.array([.45, .45, .45, .2]),
                             attach_to=base),
        make_collision_model("cdprimit2.STL",
                             pos=np.array([-0.14, 0.825, 0.5]),
                             rgba=np.array([.45, .45, .45, .2]),
                             attach_to=base),
        make_collision_model("cdprimit3.STL",
                             pos=np.array([-0.075, 0.425, 1.55]),
                             rgba=np.array([.45, .45, .45, .2]),
                             attach_to=base),
        make_collision_model("cdprimit4.STL",
                             pos=np.array([0.73, -0.525, 1.4]),
                             rgba=np.array([.45, .45, .45, .2]),
                             attach_to=base),
        make_collision_model("cdprimit5.STL",
                             pos=np.array([0.73, -0.325, 1.75]),
                             rgba=np.array([.45, .45, .45, .2]),
                             attach_to=base),
    ]

    make_collision_model("xipan.STL",
                         pos=np.array([0.22, -0.13, 0.77]),
                         rotmat=rm.rotmat_from_euler(0, 0, math.pi),
                         rgba=np.array([.2, .55, .9, .55]),
                         attach_to=base)
    static_obstacles += build_box_stack(base,
                                        center=BOX1_CENTER,
                                        rotmat=rm.rotmat_from_euler(0, 0, math.pi / 2),
                                        part_offsets=BOX1_PART_OFFSETS)
    static_obstacles += build_box_stack(base,
                                        center=BOX2_CENTER,
                                        rotmat=rm.rotmat_from_euler(0, 0, math.pi),
                                        part_offsets=BOX2_PART_OFFSETS)

    payload_dict = {}
    for object_name, spec in OBJECT_SPECS.items():
        payload_dict[object_name] = make_collision_model(spec["mesh"],
                                                         pos=spec["pick_pose"][0],
                                                         rotmat=spec["pick_pose"][1],
                                                         rgba=spec["rgba"],
                                                         attach_to=base,
                                                         mesh_dir=MODEL_DIR)
        attach_position_markers(base,
                                spec["grasp_positions"],
                                rgb=spec["rgba"][:3],
                                alpha=.42,
                                radius=.012)
        attach_position_markers(base,
                                [pose[0] for pose in spec["place_poses"]],
                                rgb=spec["rgba"][:3],
                                alpha=.16,
                                radius=.018)
        attach_place_object_previews(base, spec, alpha=.16)
        attach_grasp_previews(base, spec)
    return static_obstacles, payload_dict


def build_box_stack(base, center, rotmat, part_offsets):
    part_list = []
    for suffix, offset in part_offsets.items():
        part_list.append(make_collision_model(f"600400148_{suffix}.STL",
                                              pos=center + offset,
                                              rotmat=rotmat,
                                              rgba=BOX_RGBA,
                                              ex_radius=0.0,
                                              attach_to=base))
    return part_list


def make_planning_obstacle_list(robot, obstacle_list):
    planning_obstacle_list = list(obstacle_list) if obstacle_list is not None else []
    for frame_cmodel in robot.frame_collision_models:
        if all(frame_cmodel is not obstacle for obstacle in planning_obstacle_list):
            planning_obstacle_list.append(frame_cmodel)
    return planning_obstacle_list


def _stable_seed(*values):
    seed = 17
    for value in values:
        for char in str(value):
            seed = (seed * 31 + ord(char)) % (2 ** 32)
    return seed


def _grasp_tcp_pose(object_pose, grasp_info):
    obj_pos, obj_rotmat = object_pose
    _, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
    return obj_pos + obj_rotmat @ jaw_center_pos, obj_rotmat @ jaw_center_rotmat


def _is_arm_conf_collision_free(arm, conf, obstacle_list, other_robot_list=None, timing_stats=None):
    tic = time.perf_counter()
    arm.goto_given_conf(conf)
    if timing_stats is not None:
        timing_stats["fk_time"] += time.perf_counter() - tic
    tic = time.perf_counter()
    is_free = not arm.is_collided(obstacle_list=obstacle_list,
                                  other_robot_list=[] if other_robot_list is None else other_robot_list,
                                  toggle_dbg=False)
    if timing_stats is not None:
        timing_stats["collision_time"] += time.perf_counter() - tic
    return is_free


def solve_ik_conf(arm,
                  tgt_pos,
                  tgt_rotmat,
                  obstacle_list,
                  seed_conf_list,
                  other_robot_list=None,
                  pos_tol=.015,
                  rot_tol=.08):
    for seed_conf in seed_conf_list:
        if seed_conf is None:
            continue
        conf = arm.ik(tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rotmat,
                      seed_jnt_values=seed_conf,
                      toggle_dbg=False)
        if conf is None or not _is_arm_conf_collision_free(arm, conf, obstacle_list, other_robot_list):
            continue
        pos_err = np.linalg.norm(arm.gl_tcp_pos - tgt_pos)
        rot_err = np.linalg.norm(rm.delta_w_between_rotmat(arm.gl_tcp_rotmat, tgt_rotmat))
        if pos_err <= pos_tol and rot_err <= rot_tol:
            return np.asarray(conf, dtype=float)
    return None


def collect_ik_confs(arm,
                     tgt_pos,
                     tgt_rotmat,
                     obstacle_list,
                     seed_conf_list,
                     other_robot_list=None,
                     pos_tol=.015,
                     rot_tol=.08,
                     duplicate_tol=1e-4,
                     timing_stats=None):
    conf_list = []
    for seed_conf in seed_conf_list:
        if seed_conf is None:
            continue
        if timing_stats is not None:
            timing_stats["seed_trials"] += 1
        tic = time.perf_counter()
        conf = arm.ik(tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rotmat,
                      seed_jnt_values=seed_conf,
                      toggle_dbg=False)
        if timing_stats is not None:
            timing_stats["ik_time"] += time.perf_counter() - tic
        if conf is None:
            if timing_stats is not None:
                timing_stats["ik_none"] += 1
            continue
        if not _is_arm_conf_collision_free(arm,
                                           conf,
                                           obstacle_list,
                                           other_robot_list,
                                           timing_stats=timing_stats):
            if timing_stats is not None:
                timing_stats["collision_rejects"] += 1
            continue
        tic = time.perf_counter()
        pos_err = np.linalg.norm(arm.gl_tcp_pos - tgt_pos)
        rot_err = np.linalg.norm(rm.delta_w_between_rotmat(arm.gl_tcp_rotmat, tgt_rotmat))
        if timing_stats is not None:
            timing_stats["pose_check_time"] += time.perf_counter() - tic
        if pos_err > pos_tol or rot_err > rot_tol:
            if timing_stats is not None:
                timing_stats["pose_rejects"] += 1
            continue
        conf = np.asarray(conf, dtype=float)
        if any(np.linalg.norm(conf - existing_conf) <= duplicate_tol for existing_conf in conf_list):
            if timing_stats is not None:
                timing_stats["duplicate_rejects"] += 1
            continue
        conf_list.append(conf)
        if timing_stats is not None:
            timing_stats["accepted_confs"] += 1
    return conf_list


def solve_task_conf(arm,
                    tgt_pos,
                    tgt_rotmat,
                    obstacle_list,
                    seed_conf_list,
                    other_robot_list=None,
                    ik_seed_count=80,
                    target_label="target",
                    reference_conf=None,
                    timing_stats=None):
    solve_start = time.perf_counter()
    stage = _target_stage(target_label)
    target_seed_start = 0 if timing_stats is None else timing_stats["seed_trials"]
    target_accepted_start = 0 if timing_stats is None else timing_stats["accepted_confs"]
    stage_stats = None
    if timing_stats is not None:
        timing_stats["target_calls"] += 1
        stage_stats = _stage_timing_stats(timing_stats, stage)
        stage_stats["calls"] += 1
    rng = np.random.default_rng(_stable_seed("ik", arm.name, np.round(tgt_pos, 4)))
    jnt_ranges = arm.arm.jnt_ranges
    ik_seed_list = list(seed_conf_list)
    ik_seed_list.extend(rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1], size=(ik_seed_count, len(jnt_ranges))))
    conf_list = collect_ik_confs(arm,
                                 tgt_pos,
                                 tgt_rotmat,
                                 obstacle_list,
                                 ik_seed_list,
                                 other_robot_list,
                                 timing_stats=timing_stats)
    elapsed = time.perf_counter() - solve_start
    if timing_stats is not None:
        timing_stats["solve_time"] += elapsed
        target_seed_trials = timing_stats["seed_trials"] - target_seed_start
        target_accepted_count = timing_stats["accepted_confs"] - target_accepted_start
        stage_stats["time"] += elapsed
        stage_stats["seed_trials"] += target_seed_trials
        stage_stats["accepted_confs"] += target_accepted_count
        timing_stats["slow_targets"].append((elapsed, stage, target_label, target_seed_trials, target_accepted_count))
    if conf_list:
        if timing_stats is not None:
            timing_stats["target_successes"] += 1
            stage_stats["successes"] += 1
        if reference_conf is None:
            return conf_list[0], "exact_ik"
        reference_conf = np.asarray(reference_conf, dtype=float)
        distance_list = [np.linalg.norm(conf - reference_conf) for conf in conf_list]
        best_id = int(np.argmin(distance_list))
        print(f"{arm.name}: selected {target_label} IK nearest to reference "
              f"from {len(conf_list)} exact candidates; joint_delta={distance_list[best_id]:.3f}.")
        return conf_list[best_id], "nearest_exact_ik"
    if timing_stats is not None:
        timing_stats["target_failures"] += 1
        stage_stats["failures"] += 1
    raise RuntimeError(
        f"Exact IK failed for {arm.name} {target_label}; closest-TCP fallback is disabled, so this object will not be grasped.")


def try_solve_task_conf(*args, **kwargs):
    try:
        return solve_task_conf(*args, **kwargs)
    except RuntimeError:
        return None


def _candidate_grasp_indices(task_spec, grasp_info_list):
    if "grasp_indices" in task_spec:
        return list(task_spec["grasp_indices"])
    return list(range(len(grasp_info_list)))


def _candidate_pick_poses(task_spec, object_spec):
    if "pick_pose_candidates" in task_spec:
        return list(task_spec["pick_pose_candidates"])
    if "pick_pose_candidates" in object_spec:
        return list(object_spec["pick_pose_candidates"])
    return [object_spec["pick_pose"]]


def _candidate_symmetry_angles(task_spec, object_spec):
    if "symmetry_angles" in task_spec:
        return [float(angle) for angle in task_spec["symmetry_angles"]]
    if "symmetry_angles" in object_spec:
        return [float(angle) for angle in object_spec["symmetry_angles"]]
    angle_count = int(task_spec.get("rotational_symmetry_angle_count",
                                    object_spec.get("rotational_symmetry_angle_count", 1)))
    if angle_count <= 1:
        return [0.0]
    return [float(angle) for angle in np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)]


def _apply_pose_symmetry(pose, object_spec, angle):
    pos, rotmat = pose
    pos = np.asarray(pos, dtype=float)
    rotmat = np.asarray(rotmat, dtype=float)
    if abs(angle) <= 1e-9:
        return pos.copy(), rotmat.copy()
    axis = np.asarray(object_spec.get("rotational_symmetry_axis", rm.const.z_ax), dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-9:
        return pos.copy(), rotmat.copy()
    return pos.copy(), rotmat @ rm.rotmat_from_axangle(axis / axis_norm, angle)


def _current_conf_dict(robot):
    return {arm_name: arm.get_jnt_values().copy() for arm_name, arm in robot.arm_dict.items()}


def build_pick_place_tasks(robot, obstacle_list=None):
    planning_obstacle_list = make_planning_obstacle_list(robot, obstacle_list)
    task_list = []
    robot.backup_state()
    try:
        for arm_name, task_spec in DUAL_PICK_PLACE_SPECS.items():
            object_name = task_spec["object_name"]
            object_spec = OBJECT_SPECS[object_name]
            grasp_info_list = load_grasp_info_list(object_spec)
            arm = robot.arm_dict[arm_name]
            initial_conf = arm.get_jnt_values().copy()
            selected_task = None
            selected_transfer_delta = np.inf
            pick_failure_count = 0
            lift_failure_count = 0
            pre_place_failure_count = 0
            place_failure_count = 0
            candidate_grasp_indices = _candidate_grasp_indices(task_spec, grasp_info_list)
            candidate_pick_poses = _candidate_pick_poses(task_spec, object_spec)
            symmetry_angles = _candidate_symmetry_angles(task_spec, object_spec)
            total_candidate_count = len(candidate_pick_poses) * len(symmetry_angles) * len(candidate_grasp_indices)
            timing_stats = _new_ik_timing_stats(arm.name, object_name)
            candidate_attempt_count = 0
            last_progress_time = time.perf_counter()

            def maybe_print_progress(force=False):
                nonlocal last_progress_time
                now = time.perf_counter()
                if force or now - last_progress_time >= IK_TIMING_PROGRESS_INTERVAL:
                    _print_ik_timing_progress(timing_stats,
                                              candidate_attempt_count,
                                              total_candidate_count,
                                              pick_failure_count,
                                              lift_failure_count,
                                              pre_place_failure_count,
                                              place_failure_count)
                    last_progress_time = now

            print(f"{arm.name}: trying {object_name} with {len(candidate_pick_poses)} pick poses, "
                  f"{len(symmetry_angles)} symmetry rotations, {len(candidate_grasp_indices)} grasps "
                  f"({total_candidate_count} total candidates).")
            for pick_pose_index, raw_pick_pose in enumerate(candidate_pick_poses):
                for symmetry_angle in symmetry_angles:
                    pick_pose = _apply_pose_symmetry(raw_pick_pose, object_spec, symmetry_angle)
                    lift_pose = (pick_pose[0] + np.array([0.0, 0.0, PICK_LIFT_HEIGHT]), pick_pose[1])
                    raw_place_pose = object_spec["place_poses"][task_spec["place_index"]]
                    place_pose = _apply_pose_symmetry(raw_place_pose, object_spec, symmetry_angle)
                    pre_place_pose = (place_pose[0] + np.array([0.0, 0.0, PLACE_APPROACH_DISTANCE]), place_pose[1])
                    for grasp_index in candidate_grasp_indices:
                        candidate_attempt_count += 1
                        grasp_info = grasp_info_list[grasp_index]
                        jaw_width = float(np.clip(grasp_info[0],
                                                  arm.hnd.jaw_range[0],
                                                  arm.hnd.jaw_range[1]))
                        pick_tcp_pos, pick_tcp_rotmat = _grasp_tcp_pose(pick_pose, grasp_info)
                        lift_tcp_pos, lift_tcp_rotmat = _grasp_tcp_pose(lift_pose, grasp_info)
                        pre_place_tcp_pos, pre_place_tcp_rotmat = _grasp_tcp_pose(pre_place_pose, grasp_info)
                        place_tcp_pos, place_tcp_rotmat = _grasp_tcp_pose(place_pose, grasp_info)
                        seed_conf_list = [arm.arm.home_conf, initial_conf]
                        pick_other_robot_list = []
                        for solved_task in task_list:
                            solved_arm = robot.arm_dict[solved_task.arm_name]
                            solved_arm.goto_given_conf(solved_task.pick_conf)
                            pick_other_robot_list.append(solved_arm)
                        target_suffix = (f"pick pose #{pick_pose_index}, symmetry {np.degrees(symmetry_angle):.1f} deg, "
                                         f"grasp #{grasp_index}")
                        pick_result = try_solve_task_conf(
                            arm,
                            pick_tcp_pos,
                            pick_tcp_rotmat,
                            planning_obstacle_list,
                            seed_conf_list,
                            other_robot_list=pick_other_robot_list,
                            target_label=f"{object_name} pick {target_suffix}",
                            timing_stats=timing_stats)
                        if pick_result is None:
                            pick_failure_count += 1
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue
                        pick_conf, pick_solution_type = pick_result
                        lift_other_robot_list = []
                        for solved_task in task_list:
                            solved_arm = robot.arm_dict[solved_task.arm_name]
                            solved_arm.goto_given_conf(solved_task.lift_conf)
                            lift_other_robot_list.append(solved_arm)
                        lift_result = try_solve_task_conf(
                            arm,
                            lift_tcp_pos,
                            lift_tcp_rotmat,
                            planning_obstacle_list,
                            [pick_conf, arm.arm.home_conf, initial_conf],
                            other_robot_list=lift_other_robot_list,
                            target_label=f"{object_name} lift {target_suffix}",
                            reference_conf=pick_conf,
                            timing_stats=timing_stats)
                        if lift_result is None:
                            lift_failure_count += 1
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue
                        lift_conf, lift_solution_type = lift_result
                        pre_place_other_robot_list = []
                        for solved_task in task_list:
                            solved_arm = robot.arm_dict[solved_task.arm_name]
                            solved_arm.goto_given_conf(solved_task.pre_place_conf)
                            pre_place_other_robot_list.append(solved_arm)
                        pre_place_result = try_solve_task_conf(
                            arm,
                            pre_place_tcp_pos,
                            pre_place_tcp_rotmat,
                            planning_obstacle_list,
                            [lift_conf, pick_conf, arm.arm.home_conf, initial_conf],
                            other_robot_list=pre_place_other_robot_list,
                            target_label=f"{object_name} pre-place {target_suffix}",
                            reference_conf=lift_conf,
                            timing_stats=timing_stats)
                        if pre_place_result is None:
                            pre_place_failure_count += 1
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue
                        pre_place_conf, pre_place_solution_type = pre_place_result
                        place_other_robot_list = []
                        for solved_task in task_list:
                            solved_arm = robot.arm_dict[solved_task.arm_name]
                            solved_arm.goto_given_conf(solved_task.place_conf)
                            place_other_robot_list.append(solved_arm)
                        place_result = try_solve_task_conf(
                            arm,
                            place_tcp_pos,
                            place_tcp_rotmat,
                            planning_obstacle_list,
                            [pre_place_conf, lift_conf, pick_conf, arm.arm.home_conf, initial_conf],
                            other_robot_list=place_other_robot_list,
                            target_label=f"{object_name} place {target_suffix}",
                            reference_conf=pre_place_conf,
                            timing_stats=timing_stats)
                        if place_result is None:
                            place_failure_count += 1
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue
                        place_conf, place_solution_type = place_result
                        arm.goto_given_conf(pick_conf)
                        payload_rel_pose = arm.cvt_gl_pose_to_tcp(pick_pose[0], pick_pose[1])
                        candidate_task = PickPlaceTask(arm_name=arm_name,
                                                       object_name=object_name,
                                                       grasp_index=grasp_index,
                                                       pick_pose_index=pick_pose_index,
                                                       symmetry_angle=symmetry_angle,
                                                       pick_conf=pick_conf,
                                                       lift_conf=lift_conf,
                                                       pre_place_conf=pre_place_conf,
                                                       place_conf=place_conf,
                                                       jaw_width=jaw_width,
                                                       pick_pose=pick_pose,
                                                       lift_pose=lift_pose,
                                                       pre_place_pose=pre_place_pose,
                                                       place_pose=place_pose,
                                                       payload_rel_pose=payload_rel_pose,
                                                       pick_solution_type=pick_solution_type,
                                                       lift_solution_type=lift_solution_type,
                                                       pre_place_solution_type=pre_place_solution_type,
                                                       place_solution_type=place_solution_type)
                        pick_lift_delta = np.linalg.norm(lift_conf - pick_conf)
                        lift_pre_place_delta = np.linalg.norm(pre_place_conf - lift_conf)
                        pre_place_place_delta = np.linalg.norm(place_conf - pre_place_conf)
                        transfer_delta = pick_lift_delta + lift_pre_place_delta + pre_place_place_delta
                        print(f"{arm.name}: feasible {object_name} pick pose #{pick_pose_index}, "
                              f"symmetry {np.degrees(symmetry_angle):.1f} deg, grasp #{grasp_index}; "
                              f"pick-lift joint_delta={pick_lift_delta:.3f}, "
                              f"lift-pre-place joint_delta={lift_pre_place_delta:.3f}, "
                              f"pre-place-place joint_delta={pre_place_place_delta:.3f}, total={transfer_delta:.3f}.")
                        if transfer_delta < selected_transfer_delta:
                            selected_task = candidate_task
                            selected_transfer_delta = transfer_delta
                        arm.goto_given_conf(initial_conf)
                        maybe_print_progress()
            if selected_task is None:
                arm.goto_given_conf(initial_conf)
                maybe_print_progress(force=True)
                _print_ik_timing_summary(timing_stats, total_candidate_count)
                raise PickPlacePlanningError(
                    f"No exact IK grasp found for {arm.name} {object_name}; tried {total_candidate_count} candidates "
                    f"({len(candidate_pick_poses)} pick poses x {len(symmetry_angles)} symmetry rotations x "
                    f"{len(candidate_grasp_indices)} grasps) "
                    f"({pick_failure_count} failed at pick, {lift_failure_count} failed at lift, "
                    f"{pre_place_failure_count} failed at pre-place, {place_failure_count} failed at place).",
                    conf_dict=_current_conf_dict(robot),
                    arm_name=arm_name,
                    object_name=object_name,
                    grasp_indices=candidate_grasp_indices)
            print(f"{arm.name}: selected {object_name} grasp #{selected_task.grasp_index} "
                  f"at pick pose #{selected_task.pick_pose_index}, "
                  f"symmetry {np.degrees(selected_task.symmetry_angle):.1f} deg "
                  f"from {total_candidate_count} candidates; "
                  f"best pick-lift-pre-place-place joint_delta={selected_transfer_delta:.3f}.")
            maybe_print_progress(force=True)
            _print_ik_timing_summary(timing_stats, total_candidate_count, selected_task)
            task_list.append(selected_task)
    finally:
        robot.restore_state()
    return task_list


def densify_multi_arm_conf_list(conf_list, max_joint_step=.05):
    if len(conf_list) < 2:
        return conf_list
    dense_conf_list = [conf_list[0]]
    arm_names = list(conf_list[0].keys())
    for current_conf_dict, next_conf_dict in zip(conf_list[:-1], conf_list[1:]):
        max_delta = max(np.linalg.norm(next_conf_dict[arm_name] - current_conf_dict[arm_name])
                        for arm_name in arm_names)
        n_steps = max(1, int(np.ceil(max_delta / max_joint_step)))
        for step in range(1, n_steps + 1):
            ratio = step / n_steps
            dense_conf_list.append({
                arm_name: current_conf_dict[arm_name] + (next_conf_dict[arm_name] - current_conf_dict[arm_name]) * ratio
                for arm_name in arm_names
            })
    return dense_conf_list


def plan_multi_arm_transfer(robot, start_conf_dict, goal_conf_dict, obstacle_list, other_robot_list=None):
    arm_name_list = sorted(start_conf_dict,
                           key=lambda arm_name: np.linalg.norm(goal_conf_dict[arm_name] - start_conf_dict[arm_name]),
                           reverse=True)
    ext_dist = .1
    max_n_iter = 20000
    max_time = 180.0
    per_arm_max_time = max_time/2
    smoothing_n_iter = 80
    coordination_ext_dist = ext_dist
    max_wait_steps = 1000
    planner = marrtc.MultiArmRRTConnect(robot)
    for arm_name in arm_name_list:
        planner.add_arm(name=arm_name,
                        start_conf=start_conf_dict[arm_name],
                        goal_conf=goal_conf_dict[arm_name])
    print(f"Multi-arm RRT: arm order {arm_name_list}, ext_dist={ext_dist}.")
    mot_data = planner.plan(obstacle_list=obstacle_list,
                            other_robot_list=[] if other_robot_list is None else other_robot_list,
                            ext_dist=ext_dist,
                            max_n_iter=max_n_iter,
                            max_time=max_time,
                            per_arm_max_time=per_arm_max_time,
                            smoothing_n_iter=smoothing_n_iter,
                            coordination_ext_dist=coordination_ext_dist,
                            moving_tcp_clearance=None,
                            tcp_z_axis_max_angle=RRT_TCP_Z_AXIS_MAX_ANGLE,
                            tcp_z_axis_world=RRT_TCP_Z_AXIS_WORLD,
                            max_wait_steps=max_wait_steps,
                            toggle_dbg=True)
    if mot_data is None:
        raise MultiArmPlanningError("Multi-arm RRT failed to plan a synchronized pick-place transfer.",
                                    debug_info=planner.last_debug_info)
    dense_conf_list = densify_multi_arm_conf_list(mot_data.conf_list)
    print(f"Planned {len(mot_data)} synchronized pick-place transfer states "
          f"({len(dense_conf_list)} playback frames after densifying).")
    return dense_conf_list


def _task_by_arm(task_list):
    return {task.arm_name: task for task in task_list}


def _frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode):
    return FrameState(conf_dict={arm_name: np.asarray(conf, dtype=float) for arm_name, conf in conf_dict.items()},
                      jaw_width_dict=jaw_width_dict.copy(),
                      payload_mode_dict={task.object_name: payload_mode for task in task_list})


def append_dual_jaw_motion(frame_list, task_list, conf_dict, start_width_dict, end_width_dict, payload_mode, n_frames=12):
    arm_names = list(conf_dict.keys())
    for ratio in np.linspace(0.0, 1.0, n_frames):
        jaw_width_dict = {
            arm_name: float(start_width_dict[arm_name] +
                            (end_width_dict[arm_name] - start_width_dict[arm_name]) * ratio)
            for arm_name in arm_names
        }
        frame_list.append(_frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def append_multi_arm_motion(frame_list, task_list, conf_list, jaw_width_dict, payload_mode):
    for conf_dict in conf_list:
        frame_list.append(_frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def append_conf_dict_interpolation(frame_list,
                                   task_list,
                                   start_conf_dict,
                                   goal_conf_dict,
                                   jaw_width_dict,
                                   payload_mode,
                                   max_joint_step=.04):
    max_delta = max(np.linalg.norm(goal_conf_dict[arm_name] - start_conf_dict[arm_name])
                    for arm_name in start_conf_dict)
    n_steps = max(1, int(np.ceil(max_delta / max_joint_step)))
    for step in range(1, n_steps + 1):
        ratio = step / n_steps
        conf_dict = {
            arm_name: start_conf_dict[arm_name] + (goal_conf_dict[arm_name] - start_conf_dict[arm_name]) * ratio
            for arm_name in start_conf_dict
        }
        frame_list.append(_frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def build_frame_list(robot, obstacle_list, task_list):
    random.seed(4)
    np.random.seed(4)
    planning_obstacle_list = make_planning_obstacle_list(robot, obstacle_list)
    pick_conf_dict = {task.arm_name: task.pick_conf for task in task_list}
    lift_conf_dict = {task.arm_name: task.lift_conf for task in task_list}
    pre_place_conf_dict = {task.arm_name: task.pre_place_conf for task in task_list}
    place_conf_dict = {task.arm_name: task.place_conf for task in task_list}
    open_width_dict = {arm_name: robot.arm_dict[arm_name].hnd.jaw_range[1] for arm_name in pick_conf_dict}
    closed_width_dict = {task.arm_name: task.jaw_width for task in task_list}
    frame_list = []
    frame_list.append(_frame_from_tasks(task_list, pick_conf_dict, open_width_dict, "pick"))
    append_dual_jaw_motion(frame_list,
                           task_list,
                           pick_conf_dict,
                           open_width_dict,
                           closed_width_dict,
                           "pick")
    append_conf_dict_interpolation(frame_list,
                                   task_list,
                                   pick_conf_dict,
                                   lift_conf_dict,
                                   closed_width_dict,
                                   "hold")
    transfer_conf_list = plan_multi_arm_transfer(robot,
                                                lift_conf_dict,
                                                pre_place_conf_dict,
                                                planning_obstacle_list)
    append_multi_arm_motion(frame_list,
                            task_list,
                            transfer_conf_list,
                            closed_width_dict,
                            "hold")
    append_conf_dict_interpolation(frame_list,
                                   task_list,
                                   pre_place_conf_dict,
                                   place_conf_dict,
                                   closed_width_dict,
                                   "hold")
    append_dual_jaw_motion(frame_list,
                           task_list,
                           place_conf_dict,
                           closed_width_dict,
                           open_width_dict,
                           "hold")
    append_dual_jaw_motion(frame_list,
                           task_list,
                           place_conf_dict,
                           open_width_dict,
                           open_width_dict,
                           "place",
                           n_frames=24)
    return frame_list


def apply_frame_state(robot, frame):
    robot.goto_conf_dict(frame.conf_dict)
    for arm_name, jaw_width in frame.jaw_width_dict.items():
        robot.arm_dict[arm_name].change_ee_values(jaw_width)


def set_payload_pose(payload, pose):
    payload.pose = pose
    _ = payload.pdndp


def apply_payload_state(robot, frame, payload_dict, task_dict):
    for object_name, pick_place_task in task_dict.items():
        payload = payload_dict[object_name]
        payload_mode = frame.payload_mode_dict[object_name]
        if payload_mode == "hold":
            rel_pos, rel_rotmat = pick_place_task.payload_rel_pose
            arm = robot.arm_dict[pick_place_task.arm_name]
            set_payload_pose(payload, arm.cvt_pose_in_tcp_to_gl(rel_pos, rel_rotmat))
        elif payload_mode == "place":
            set_payload_pose(payload, pick_place_task.place_pose)
        else:
            set_payload_pose(payload, pick_place_task.pick_pose)


def _sample_indices(path_len, max_samples):
    if path_len <= 0:
        return []
    if path_len <= max_samples:
        return list(range(path_len))
    return sorted(set(np.linspace(0, path_len - 1, max_samples, dtype=int).tolist()))


def _copy_conf_dict(conf_dict):
    return {arm_name: np.asarray(conf, dtype=float).copy() for arm_name, conf in conf_dict.items()}


def attach_tcp_trace(base, robot, path_dict, start_conf_dict, max_samples=70):
    color_dict = {
        "lft_arm": np.array([.0, .62, 1.0]),
        "rgt_arm": np.array([1.0, .42, .0]),
    }
    for arm_name, path in path_dict.items():
        if arm_name not in robot.arm_dict or not path:
            continue
        rgb = color_dict.get(arm_name, np.array([.85, .15, .95]))
        prev_pos = None
        for path_id in _sample_indices(len(path), max_samples):
            conf_dict = _copy_conf_dict(start_conf_dict)
            conf_dict[arm_name] = np.asarray(path[path_id], dtype=float)
            robot.goto_conf_dict(conf_dict)
            tcp_pos = robot.arm_dict[arm_name].gl_tcp_pos.copy()
            mgm.gen_sphere(pos=tcp_pos,
                           radius=.008,
                           rgb=rgb,
                           alpha=.65).attach_to(base)
            if prev_pos is not None:
                mgm.gen_stick(spos=prev_pos,
                              epos=tcp_pos,
                              radius=.0025,
                              rgb=rgb,
                              alpha=.35).attach_to(base)
            prev_pos = tcp_pos


def attach_multi_arm_planning_debug(base, robot, debug_info):
    if not debug_info:
        robot.gen_meshmodel(alpha=.9,
                            toggle_tcp_frame=True,
                            toggle_cdprim=True).attach_to(base)
        return
    start_conf_dict = debug_info.get("start_conf_dict", {})
    goal_conf_dict = debug_info.get("goal_conf_dict", {})
    path_dict = debug_info.get("path_dict", {})
    sipp_conf_dict = debug_info.get("sipp_conf_dict")
    failed_arm = debug_info.get("failed_arm")
    stage = debug_info.get("stage")
    sipp_state = debug_info.get("sipp_state")
    print("Drawing multi-arm planning debug:",
          f"stage={stage}, failed_arm={failed_arm}, sipp_state={sipp_state},",
          f"path_lengths={debug_info.get('path_lengths')}.")

    robot.backup_state()
    try:
        if start_conf_dict:
            robot.goto_conf_dict(start_conf_dict)
            robot.gen_meshmodel(alpha=.18,
                                toggle_tcp_frame=False,
                                toggle_cdprim=False).attach_to(base)
        if goal_conf_dict:
            robot.goto_conf_dict(goal_conf_dict)
            robot.gen_meshmodel(alpha=.22,
                                toggle_tcp_frame=True,
                                toggle_cdprim=False).attach_to(base)
        if start_conf_dict and path_dict:
            attach_tcp_trace(base, robot, path_dict, start_conf_dict)
        if sipp_conf_dict:
            robot.goto_conf_dict(sipp_conf_dict)
            robot.gen_meshmodel(alpha=.92,
                                toggle_tcp_frame=True,
                                toggle_cdprim=True).attach_to(base)
    finally:
        robot.restore_state()
    if sipp_conf_dict:
        robot.goto_conf_dict(sipp_conf_dict)


def attach_planning_failure_debug(base, robot, error):
    if isinstance(error, MultiArmPlanningError):
        attach_multi_arm_planning_debug(base, robot, error.debug_info)
        return
    if isinstance(error, PickPlacePlanningError) and error.conf_dict:
        robot.goto_conf_dict(error.conf_dict)
    robot.gen_meshmodel(alpha=.9,
                        toggle_tcp_frame=True,
                        toggle_cdprim=True).attach_to(base)
    if isinstance(error, PickPlacePlanningError) and error.object_name in OBJECT_SPECS:
        attach_grasp_debug_previews(base,
                                    OBJECT_SPECS[error.object_name],
                                    error.grasp_indices)


def precompute_robot_meshes(robot, frame_list):
    mesh_list = []
    robot.backup_state()
    for frame in frame_list:
        apply_frame_state(robot, frame)
        mesh_list.append(robot.gen_meshmodel(alpha=.88, toggle_tcp_frame=True))
    robot.restore_state()
    return mesh_list


def update(animation_data, panda_task):
    frame = animation_data.frame_list[animation_data.counter]
    if animation_data.current_robot_mesh is not None:
        animation_data.current_robot_mesh.detach()
    animation_data.current_robot_mesh = animation_data.robot_mesh_list[animation_data.counter]
    animation_data.current_robot_mesh.attach_to(base)

    apply_frame_state(animation_data.robot, frame)
    apply_payload_state(animation_data.robot,
                        frame,
                        animation_data.payload_dict,
                        animation_data.task_dict)

    if animation_data.counter == len(animation_data.frame_list) - 1:
        animation_data.end_hold_counter += 1
        if animation_data.end_hold_counter >= 24:
            animation_data.counter = 0
            animation_data.end_hold_counter = 0
    else:
        animation_data.counter += 1
    return panda_task.again


def main(toggle_visual=True,
         vertical_frame_height=RACK_VERTICAL_FRAME_HEIGHT,
         vertical_frame_x_length=RACK_VERTICAL_FRAME_X_LENGTH,
         vertical_frame_y_length=RACK_VERTICAL_FRAME_Y_LENGTH,
         horizontal_frame_thickness=RACK_HORIZONTAL_FRAME_THICKNESS,
         horizontal_frame_x_length=RACK_HORIZONTAL_FRAME_X_LENGTH,
         horizontal_frame_y_length=RACK_HORIZONTAL_FRAME_Y_LENGTH,
         debug_on_failure=True):
    global base
    base = wd.World(cam_pos=[1.8, 1.6, 1.35], lookat_pos=[0.35, 0.0, 0.95])
    mgm.gen_frame().attach_to(base)

    robot = DualUR7EDH50(enable_cc=True,
                         body_root_pos=RACK_BASE_POS,
                         body_root_rotmat=RACK_ROT,
                         vertical_frame_height=vertical_frame_height,
                         vertical_frame_xy=RACK_VERTICAL_FRAME_XY,
                         vertical_frame_x_length=vertical_frame_x_length,
                         vertical_frame_y_length=vertical_frame_y_length,
                         horizontal_frame_thickness=horizontal_frame_thickness,
                         horizontal_frame_x_length=horizontal_frame_x_length,
                         horizontal_frame_y_length=horizontal_frame_y_length,
                         vertical_frame_rgb=RACK_VERTICAL_FRAME_RGB,
                         horizontal_frame_rgb=RACK_HORIZONTAL_FRAME_RGB,
                         vertical_frame_alpha=RACK_VERTICAL_FRAME_ALPHA,
                         horizontal_frame_alpha=RACK_HORIZONTAL_FRAME_ALPHA,
                         arm_y_offset=RACK_ARM_Y_OFFSET,
                         arm_y_offset_reference_frame_y_length=RACK_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH,
                         lft_arm_loc_rotmat=RACK_LFT_ARM_LOC_ROTMAT,
                         rgt_arm_loc_rotmat=RACK_RGT_ARM_LOC_ROTMAT,
                         lft_home_conf=UR3_DUAL_LFT_HOME_CONF,
                         rgt_home_conf=UR3_DUAL_RGT_HOME_CONF)
    robot.lft_arm.hndopen()
    robot.rgt_arm.hndopen()
    print_tracik_status(robot)

    obstacle_list, payload_dict = build_inside_scene(base)
    try:
        task_list = build_pick_place_tasks(robot, obstacle_list)
        task_dict = {task.object_name: task for task in task_list}
        frame_list = build_frame_list(robot, obstacle_list, task_list)
    except RuntimeError as error:
        if debug_on_failure:
            attach_planning_failure_debug(base, robot, error)
            if toggle_visual:
                print(f"Planning failed; debug scene is displayed for: {error}")
                base.run()
            else:
                print(f"Planning failed; debug scene is prepared for: {error}")
        raise
    print(f"Generated {len(frame_list)} dual-arm pick-and-place frames for UR7E + DH50.")

    if toggle_visual:
        robot_mesh_list = precompute_robot_meshes(robot, frame_list)
        animation_data = AnimationData(robot=robot,
                                       robot_mesh_list=robot_mesh_list,
                                       frame_list=frame_list,
                                       payload_dict=payload_dict,
                                       task_dict=task_dict)
        apply_frame_state(robot, frame_list[0])
        apply_payload_state(robot, frame_list[0], payload_dict, task_dict)
        taskMgr.doMethodLater(.05,
                              update,
                              "ur7e_dh50_pickandplace_inside_update",
                              extraArgs=[animation_data],
                              appendTask=True)
        base.run()
    return frame_list


if __name__ == "__main__":
    main(toggle_visual=True)
