import math
import os
import pickle
import random
import sys
from dataclasses import dataclass

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wrs import wd, mgm
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.motion.probabilistic import rrt_connect as rrtc
from wrs.robot_sim.end_effectors.grippers.dh50.dh50 import Dh50
from wrs.robot_sim.robots.ur7e.dual_ur7e_dh50 import DualUR7EDH50


MESH_DIR = os.path.join(project_root, "old_version", "rbt", "ur7e", "meshes")
MODEL_DIR = os.path.join(project_root, "yanpu", "models")

from yanpu.ur7e_dh50_pickandplace_params import (
    BOX1_CENTER,
    BOX1_PART_OFFSETS,
    BOX2_CENTER,
    BOX2_PART_OFFSETS,
    BOX_RGBA,
    DUAL_PICK_PLACE_SPECS,
    OBJECT_SPECS,
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
    RACK_VERTICAL_FRAME_XY,
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
    pick_conf: np.ndarray
    place_conf: np.ndarray
    jaw_width: float
    pick_pose: tuple
    place_pose: tuple
    payload_rel_pose: tuple
    pick_solution_type: str
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


def _is_arm_conf_collision_free(arm, conf, obstacle_list, other_robot_list=None):
    arm.goto_given_conf(conf)
    return not arm.is_collided(obstacle_list=obstacle_list,
                               other_robot_list=[] if other_robot_list is None else other_robot_list,
                               toggle_dbg=False)


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


def solve_task_conf(arm,
                    tgt_pos,
                    tgt_rotmat,
                    obstacle_list,
                    seed_conf_list,
                    other_robot_list=None,
                    ik_seed_count=240,
                    target_label="target"):
    rng = np.random.default_rng(_stable_seed("ik", arm.name, np.round(tgt_pos, 4)))
    jnt_ranges = arm.arm.jnt_ranges
    ik_seed_list = list(seed_conf_list)
    ik_seed_list.extend(rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1], size=(ik_seed_count, len(jnt_ranges))))
    conf = solve_ik_conf(arm, tgt_pos, tgt_rotmat, obstacle_list, ik_seed_list, other_robot_list)
    if conf is not None:
        return conf, "exact_ik"
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


def build_pick_place_tasks(robot, obstacle_list=None):
    planning_obstacle_list = make_planning_obstacle_list(robot, obstacle_list)
    task_list = []
    robot.backup_state()
    try:
        for arm_name, task_spec in DUAL_PICK_PLACE_SPECS.items():
            object_name = task_spec["object_name"]
            object_spec = OBJECT_SPECS[object_name]
            grasp_info_list = load_grasp_info_list(object_spec)
            pick_pose = object_spec["pick_pose"]
            place_pose = object_spec["place_poses"][task_spec["place_index"]]
            arm = robot.arm_dict[arm_name]
            initial_conf = arm.get_jnt_values().copy()
            selected_task = None
            pick_failure_count = 0
            place_failure_count = 0
            candidate_grasp_indices = _candidate_grasp_indices(task_spec, grasp_info_list)
            for grasp_index in candidate_grasp_indices:
                grasp_info = grasp_info_list[grasp_index]
                jaw_width = float(np.clip(grasp_info[0],
                                          arm.hnd.jaw_range[0],
                                          arm.hnd.jaw_range[1]))
                pick_tcp_pos, pick_tcp_rotmat = _grasp_tcp_pose(pick_pose, grasp_info)
                place_tcp_pos, place_tcp_rotmat = _grasp_tcp_pose(place_pose, grasp_info)
                seed_conf_list = [arm.arm.home_conf, initial_conf]
                pick_other_robot_list = []
                for solved_task in task_list:
                    solved_arm = robot.arm_dict[solved_task.arm_name]
                    solved_arm.goto_given_conf(solved_task.pick_conf)
                    pick_other_robot_list.append(solved_arm)
                pick_result = try_solve_task_conf(
                    arm,
                    pick_tcp_pos,
                    pick_tcp_rotmat,
                    planning_obstacle_list,
                    seed_conf_list,
                    other_robot_list=pick_other_robot_list,
                    target_label=f"{object_name} pick grasp #{grasp_index}")
                if pick_result is None:
                    pick_failure_count += 1
                    arm.goto_given_conf(initial_conf)
                    continue
                pick_conf, pick_solution_type = pick_result
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
                    [pick_conf, arm.arm.home_conf, initial_conf],
                    other_robot_list=place_other_robot_list,
                    target_label=f"{object_name} place grasp #{grasp_index}")
                if place_result is None:
                    place_failure_count += 1
                    arm.goto_given_conf(initial_conf)
                    continue
                place_conf, place_solution_type = place_result
                arm.goto_given_conf(pick_conf)
                payload_rel_pose = arm.cvt_gl_pose_to_tcp(pick_pose[0], pick_pose[1])
                selected_task = PickPlaceTask(arm_name=arm_name,
                                             object_name=object_name,
                                             grasp_index=grasp_index,
                                             pick_conf=pick_conf,
                                             place_conf=place_conf,
                                             jaw_width=jaw_width,
                                             pick_pose=pick_pose,
                                             place_pose=place_pose,
                                             payload_rel_pose=payload_rel_pose,
                                             pick_solution_type=pick_solution_type,
                                             place_solution_type=place_solution_type)
                print(f"{arm.name}: selected {object_name} grasp #{grasp_index} from {len(candidate_grasp_indices)} candidates.")
                break
            if selected_task is None:
                raise RuntimeError(
                    f"No exact IK grasp found for {arm.name} {object_name}; tried {len(candidate_grasp_indices)} grasp candidates "
                    f"({pick_failure_count} failed at pick, {place_failure_count} failed at place).")
            task_list.append(selected_task)
    finally:
        robot.restore_state()
    return task_list


def plan_arm_segment(arm, start_conf, goal_conf, obstacle_list, other_robot_list=None):
    planner = rrtc.RRTConnect(arm)
    planner.rbt = arm
    mot_data = planner.plan(start_conf=start_conf,
                            goal_conf=goal_conf,
                            obstacle_list=obstacle_list,
                            other_robot_list=[] if other_robot_list is None else other_robot_list,
                            ext_dist=.75,
                            max_time=12.0,
                            smoothing_n_iter=0,
                            toggle_dbg=False)
    if mot_data is None:
        print(f"RRT failed for {arm.name}; falling back to a joint-space interpolation.")
        return list(np.linspace(start_conf, goal_conf, 60))
    return mot_data.jv_list


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


def append_dual_path(frame_list, task_list, path_dict, static_conf_dict, jaw_width_dict, payload_mode):
    path_len = max(len(path) for path in path_dict.values())
    for i in range(path_len):
        conf_dict = {}
        for arm_name, path in path_dict.items():
            if path_len <= 1:
                path_id = 0
            else:
                path_id = int(round(i * (len(path) - 1) / (path_len - 1)))
            conf_dict[arm_name] = np.asarray(path[path_id], dtype=float)
        for arm_name, conf in static_conf_dict.items():
            conf_dict.setdefault(arm_name, conf)
        frame_list.append(_frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def build_frame_list(robot, obstacle_list, task_list):
    random.seed(4)
    np.random.seed(4)
    planning_obstacle_list = make_planning_obstacle_list(robot, obstacle_list)
    pick_conf_dict = {task.arm_name: task.pick_conf for task in task_list}
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
    current_conf_dict = {arm_name: conf.copy() for arm_name, conf in pick_conf_dict.items()}
    for task in task_list:
        arm_name = task.arm_name
        static_conf_dict = {
            other_arm_name: conf
            for other_arm_name, conf in current_conf_dict.items()
            if other_arm_name != arm_name
        }
        robot.goto_conf_dict(static_conf_dict)
        other_robot_list = [robot.arm_dict[other_arm_name] for other_arm_name in static_conf_dict]
        transfer_path = plan_arm_segment(robot.arm_dict[arm_name],
                                         current_conf_dict[arm_name],
                                         task.place_conf,
                                         planning_obstacle_list,
                                         other_robot_list=other_robot_list)
        append_dual_path(frame_list,
                         task_list,
                         {arm_name: transfer_path},
                         static_conf_dict=static_conf_dict,
                         jaw_width_dict=closed_width_dict,
                         payload_mode="hold")
        current_conf_dict[arm_name] = task.place_conf
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
    for object_name, pick_place_task in animation_data.task_dict.items():
        payload = animation_data.payload_dict[object_name]
        payload_mode = frame.payload_mode_dict[object_name]
        if payload_mode == "hold":
            rel_pos, rel_rotmat = pick_place_task.payload_rel_pose
            arm = animation_data.robot.arm_dict[pick_place_task.arm_name]
            payload.pose = arm.cvt_pose_in_tcp_to_gl(rel_pos, rel_rotmat)
        elif payload_mode == "place":
            payload.pose = pick_place_task.place_pose
        else:
            payload.pose = pick_place_task.pick_pose

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
         horizontal_frame_thickness=RACK_HORIZONTAL_FRAME_THICKNESS,
         horizontal_frame_x_length=RACK_HORIZONTAL_FRAME_X_LENGTH,
         horizontal_frame_y_length=RACK_HORIZONTAL_FRAME_Y_LENGTH):
    global base
    base = wd.World(cam_pos=[1.8, 1.6, 1.35], lookat_pos=[0.35, 0.0, 0.95])
    mgm.gen_frame().attach_to(base)

    robot = DualUR7EDH50(enable_cc=True,
                         body_root_pos=RACK_BASE_POS,
                         body_root_rotmat=RACK_ROT,
                         vertical_frame_height=vertical_frame_height,
                         vertical_frame_xy=RACK_VERTICAL_FRAME_XY,
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

    obstacle_list, payload_dict = build_inside_scene(base)
    task_list = build_pick_place_tasks(robot, obstacle_list)
    task_dict = {task.object_name: task for task in task_list}
    frame_list = build_frame_list(robot, obstacle_list, task_list)
    print(f"Generated {len(frame_list)} dual-arm pick-and-place frames for UR7E + DH50.")

    if toggle_visual:
        robot_mesh_list = precompute_robot_meshes(robot, frame_list)
        animation_data = AnimationData(robot=robot,
                                       robot_mesh_list=robot_mesh_list,
                                       frame_list=frame_list,
                                       payload_dict=payload_dict,
                                       task_dict=task_dict)
        taskMgr.doMethodLater(.05,
                              update,
                              "ur7e_dh50_pickandplace_inside_update",
                              extraArgs=[animation_data],
                              appendTask=True)
        base.run()
    return frame_list


if __name__ == "__main__":
    main(toggle_visual=True)
