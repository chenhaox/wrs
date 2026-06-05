import random

import numpy as np

from wrs.motion.probabilistic import multi_arm_rrt_connect as marrtc

from yanpu_pnp import scene
from yanpu_pnp.models import FrameState, MultiArmPlanningError


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


def _tcp_axis_cfg(rrt_cfg):
    axis = np.asarray(rrt_cfg.get("tcp_z_axis_world", [0.0, 0.0, -1.0]), dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle_deg = rrt_cfg.get("tcp_z_axis_max_angle_deg", None)
    if angle_deg is None:
        return axis, None
    return axis, np.radians(float(angle_deg))


def _plan_info_from_motion_data(mot_data, planner):
    raw_path_lengths = {
        arm_name: int(len(path))
        for arm_name, path in planner.arm_path_dict.items()
    }
    synchronized_state_count = int(len(mot_data))
    return {
        "raw_arm_path_lengths": raw_path_lengths,
        "raw_arm_path_point_count_sum": int(sum(raw_path_lengths.values())),
        "synchronized_state_count": synchronized_state_count,
        "planned_arm_path_point_count_sum": int(synchronized_state_count * len(mot_data.arm_names)),
    }


def plan_multi_arm_transfer(robot,
                            cfg,
                            start_conf_dict,
                            goal_conf_dict,
                            obstacle_list,
                            other_robot_list=None,
                            return_plan_info=False):
    rrt_cfg = cfg["pnp"]["rrt"]
    arm_name_list = sorted(start_conf_dict,
                           key=lambda arm_name: np.linalg.norm(goal_conf_dict[arm_name] - start_conf_dict[arm_name]),
                           reverse=True)
    planner = marrtc.MultiArmRRTConnect(robot)
    for arm_name in arm_name_list:
        planner.add_arm(name=arm_name,
                        start_conf=start_conf_dict[arm_name],
                        goal_conf=goal_conf_dict[arm_name])
    tcp_axis_world, tcp_axis_angle = _tcp_axis_cfg(rrt_cfg)
    print(f"Multi-arm RRT: arm order {arm_name_list}, ext_dist={rrt_cfg['ext_dist']}.")
    mot_data = planner.plan(obstacle_list=obstacle_list,
                            other_robot_list=[] if other_robot_list is None else other_robot_list,
                            ext_dist=float(rrt_cfg["ext_dist"]),
                            max_n_iter=int(rrt_cfg["max_n_iter"]),
                            max_time=float(rrt_cfg["max_time"]),
                            per_arm_max_time=float(rrt_cfg["per_arm_max_time"]),
                            smoothing_n_iter=int(rrt_cfg["smoothing_n_iter"]),
                            coordination_ext_dist=float(rrt_cfg["coordination_ext_dist"]),
                            moving_tcp_clearance=None,
                            tcp_z_axis_max_angle=tcp_axis_angle,
                            tcp_z_axis_world=tcp_axis_world,
                            max_wait_steps=int(rrt_cfg["max_wait_steps"]),
                            toggle_dbg=True)
    if mot_data is None:
        raise MultiArmPlanningError("Multi-arm RRT failed to plan a synchronized pick-place transfer.",
                                    debug_info=planner.last_debug_info)
    plan_info = _plan_info_from_motion_data(mot_data, planner)
    dense_conf_list = densify_multi_arm_conf_list(mot_data.conf_list)
    print(f"Planned {len(mot_data)} synchronized pick-place transfer states "
          f"({len(dense_conf_list)} playback frames after densifying).")
    if return_plan_info:
        plan_info["dense_playback_frame_count"] = int(len(dense_conf_list))
        return dense_conf_list, plan_info
    return dense_conf_list


def frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode):
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
        frame_list.append(frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def append_multi_arm_motion(frame_list, task_list, conf_list, jaw_width_dict, payload_mode):
    for conf_dict in conf_list:
        frame_list.append(frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


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
        frame_list.append(frame_from_tasks(task_list, conf_dict, jaw_width_dict, payload_mode))


def build_frame_list_with_plan_info(robot, cfg, obstacle_list, task_list):
    random.seed(4)
    np.random.seed(4)
    planning_obstacle_list = scene.make_planning_obstacle_list(robot, obstacle_list)
    pick_conf_dict = {task.arm_name: task.pick_conf for task in task_list}
    lift_conf_dict = {task.arm_name: task.lift_conf for task in task_list}
    pre_place_conf_dict = {task.arm_name: task.pre_place_conf for task in task_list}
    place_conf_dict = {task.arm_name: task.place_conf for task in task_list}
    open_width_dict = {arm_name: robot.arm_dict[arm_name].hnd.jaw_range[1] for arm_name in pick_conf_dict}
    closed_width_dict = {task.arm_name: task.jaw_width for task in task_list}
    frame_list = []
    frame_list.append(frame_from_tasks(task_list, pick_conf_dict, open_width_dict, "pick"))
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
    transfer_conf_list, plan_info = plan_multi_arm_transfer(robot,
                                                            cfg,
                                                            lift_conf_dict,
                                                            pre_place_conf_dict,
                                                            planning_obstacle_list,
                                                            return_plan_info=True)
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
    plan_info["animation_frame_count"] = int(len(frame_list))
    return frame_list, plan_info


def build_frame_list(robot, cfg, obstacle_list, task_list):
    frame_list, _plan_info = build_frame_list_with_plan_info(robot, cfg, obstacle_list, task_list)
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
