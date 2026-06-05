import copy
import itertools
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wrs import wd, mgm
import wrs.basis.robot_math as rm

from yanpu import ur7e_dh50_pickandplace_inside as pp
from yanpu import ur7e_dh50_pickandplace_params as params


@dataclass
class CandidatePlan:
    u_pick_index: int
    u625_pick_index: int
    object_specs: dict
    task_list: list
    frame_list: list
    score: float


class PickPoseSearchError(RuntimeError):

    def __init__(self, message, failures):
        super().__init__(message)
        self.failures = failures


class _TemporaryPickPlaceSpecs:

    def __init__(self, object_specs):
        self.object_specs = object_specs
        self._old_object_specs = None
        self._old_dual_specs = None

    def __enter__(self):
        self._old_object_specs = pp.OBJECT_SPECS
        self._old_dual_specs = pp.DUAL_PICK_PLACE_SPECS
        pp.OBJECT_SPECS = self.object_specs
        pp.DUAL_PICK_PLACE_SPECS = params.DUAL_PICK_PLACE_SPECS

    def __exit__(self, exc_type, exc_value, traceback):
        pp.OBJECT_SPECS = self._old_object_specs
        pp.DUAL_PICK_PLACE_SPECS = self._old_dual_specs


def _copy_pose(pose):
    return np.asarray(pose[0], dtype=float).copy(), np.asarray(pose[1], dtype=float).copy()


def make_candidate_object_specs(u_pick_index, u625_pick_index):
    object_specs = copy.deepcopy(params.OBJECT_SPECS)
    u_pick_rotmat = _copy_pose(params.OBJECT_SPECS["u"]["pick_pose"])[1]
    u625_pick_rotmat = _copy_pose(params.OBJECT_SPECS["U625"]["pick_pose"])[1]
    object_specs["u"]["pick_pose"] = (
        np.asarray(params.U_GRASP_POSITIONS[u_pick_index], dtype=float).copy(),
        u_pick_rotmat,
    )
    object_specs["u"]["pick_pose_candidates"] = [object_specs["u"]["pick_pose"]]
    object_specs["U625"]["pick_pose"] = (
        np.asarray(params.U625_GRASP_POSITIONS[u625_pick_index], dtype=float).copy(),
        u625_pick_rotmat,
    )
    object_specs["U625"]["pick_pose_candidates"] = [object_specs["U625"]["pick_pose"]]
    return object_specs


def build_robot(vertical_frame_height=params.RACK_VERTICAL_FRAME_HEIGHT,
                vertical_frame_x_length=params.RACK_VERTICAL_FRAME_X_LENGTH,
                vertical_frame_y_length=params.RACK_VERTICAL_FRAME_Y_LENGTH,
                horizontal_frame_thickness=params.RACK_HORIZONTAL_FRAME_THICKNESS,
                horizontal_frame_x_length=params.RACK_HORIZONTAL_FRAME_X_LENGTH,
                horizontal_frame_y_length=params.RACK_HORIZONTAL_FRAME_Y_LENGTH):
    robot = pp.DualUR7EDH50(enable_cc=True,
                            body_root_pos=params.RACK_BASE_POS,
                            body_root_rotmat=params.RACK_ROT,
                            vertical_frame_height=vertical_frame_height,
                            vertical_frame_xy=params.RACK_VERTICAL_FRAME_XY,
                            vertical_frame_x_length=vertical_frame_x_length,
                            vertical_frame_y_length=vertical_frame_y_length,
                            horizontal_frame_thickness=horizontal_frame_thickness,
                            horizontal_frame_x_length=horizontal_frame_x_length,
                            horizontal_frame_y_length=horizontal_frame_y_length,
                            vertical_frame_rgb=params.RACK_VERTICAL_FRAME_RGB,
                            horizontal_frame_rgb=params.RACK_HORIZONTAL_FRAME_RGB,
                            vertical_frame_alpha=params.RACK_VERTICAL_FRAME_ALPHA,
                            horizontal_frame_alpha=params.RACK_HORIZONTAL_FRAME_ALPHA,
                            arm_y_offset=params.RACK_ARM_Y_OFFSET,
                            arm_y_offset_reference_frame_y_length=params.RACK_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH,
                            lft_arm_loc_rotmat=params.RACK_LFT_ARM_LOC_ROTMAT,
                            rgt_arm_loc_rotmat=params.RACK_RGT_ARM_LOC_ROTMAT,
                            lft_home_conf=params.UR3_DUAL_LFT_HOME_CONF,
                            rgt_home_conf=params.UR3_DUAL_RGT_HOME_CONF)
    robot.lft_arm.hndopen()
    robot.rgt_arm.hndopen()
    return robot


def build_planning_static_obstacles():
    obstacle_list = [
        pp.make_collision_model("pengzhuang1.STL",
                                pos=np.array([-0.13287, -0.36, 0.753]),
                                rgba=np.array([.5, .5, .5, .22])),
        pp.make_collision_model("pengzhuang2.STL",
                                pos=np.array([-0.095, 0.202, 0.683]),
                                rgba=np.array([.5, .5, .5, .22])),
        pp.make_collision_model("cdprimit1.STL",
                                pos=np.array([0.155, -0.905, 0.0]),
                                rotmat=rm.rotmat_from_euler(0, 0, math.pi),
                                rgba=np.array([.45, .45, .45, .2])),
        pp.make_collision_model("cdprimit2.STL",
                                pos=np.array([-0.14, 0.825, 0.5]),
                                rgba=np.array([.45, .45, .45, .2])),
        pp.make_collision_model("cdprimit3.STL",
                                pos=np.array([-0.075, 0.425, 1.55]),
                                rgba=np.array([.45, .45, .45, .2])),
        pp.make_collision_model("cdprimit4.STL",
                                pos=np.array([0.73, -0.525, 1.4]),
                                rgba=np.array([.45, .45, .45, .2])),
        pp.make_collision_model("cdprimit5.STL",
                                pos=np.array([0.73, -0.325, 1.75]),
                                rgba=np.array([.45, .45, .45, .2])),
    ]
    obstacle_list += _build_box_stack_obstacles(params.BOX1_CENTER,
                                                rm.rotmat_from_euler(0, 0, math.pi / 2),
                                                params.BOX1_PART_OFFSETS)
    obstacle_list += _build_box_stack_obstacles(params.BOX2_CENTER,
                                                rm.rotmat_from_euler(0, 0, math.pi),
                                                params.BOX2_PART_OFFSETS)
    return obstacle_list


def _build_box_stack_obstacles(center, rotmat, part_offsets):
    return [
        pp.make_collision_model(f"600400148_{suffix}.STL",
                                pos=center + offset,
                                rotmat=rotmat,
                                rgba=params.BOX_RGBA,
                                ex_radius=0.0)
        for suffix, offset in part_offsets.items()
    ]


def _frame_path_score(frame_list):
    score = 0.0
    for last_frame, current_frame in zip(frame_list[:-1], frame_list[1:]):
        for arm_name in current_frame.conf_dict:
            score += np.linalg.norm(current_frame.conf_dict[arm_name] - last_frame.conf_dict[arm_name])
    return float(score)


def _is_tcp_axis_valid(robot, frame):
    if params.RRT_TCP_Z_AXIS_MAX_ANGLE is None:
        return True
    tcp_z_axis_world = np.asarray(params.RRT_TCP_Z_AXIS_WORLD, dtype=float)
    tcp_z_axis_world = tcp_z_axis_world / np.linalg.norm(tcp_z_axis_world)
    min_dot = float(np.cos(params.RRT_TCP_Z_AXIS_MAX_ANGLE))
    for arm_name in frame.conf_dict:
        tcp_z_axis = np.asarray(robot.arm_dict[arm_name].gl_tcp_rotmat, dtype=float)[:, 2]
        tcp_z_axis = tcp_z_axis / np.linalg.norm(tcp_z_axis)
        if float(np.dot(tcp_z_axis, tcp_z_axis_world)) < min_dot:
            return False
    return True


def validate_frame_list(robot, obstacle_list, frame_list):
    planning_obstacle_list = pp.make_planning_obstacle_list(robot, obstacle_list)
    robot.backup_state()
    try:
        for frame_id, frame in enumerate(frame_list):
            pp.apply_frame_state(robot, frame)
            if not _is_tcp_axis_valid(robot, frame):
                return False, f"frame {frame_id} violates TCP z-axis constraint"
            if robot.is_collided(obstacle_list=planning_obstacle_list, other_robot_list=[]):
                return False, f"frame {frame_id} is in collision"
    finally:
        robot.restore_state()
    return True, "ok"


def plan_candidate(robot, obstacle_list, u_pick_index, u625_pick_index):
    object_specs = make_candidate_object_specs(u_pick_index, u625_pick_index)
    print("Trying pick-pose candidate "
          f"u#{u_pick_index}={object_specs['u']['pick_pose'][0]}, "
          f"U625#{u625_pick_index}={object_specs['U625']['pick_pose'][0]}.")
    robot.backup_state()
    try:
        with _TemporaryPickPlaceSpecs(object_specs):
            task_list = pp.build_pick_place_tasks(robot, obstacle_list)
            if len(task_list) != len(params.DUAL_PICK_PLACE_SPECS):
                raise RuntimeError(f"only {len(task_list)} tasks planned")
            frame_list = pp.build_frame_list(robot, obstacle_list, task_list)
            is_valid, reason = validate_frame_list(robot, obstacle_list, frame_list)
            if not is_valid:
                raise RuntimeError(reason)
            score = _frame_path_score(frame_list)
            print(f"Candidate u#{u_pick_index}/U625#{u625_pick_index} accepted; "
                  f"frames={len(frame_list)}, score={score:.3f}.")
            return CandidatePlan(u_pick_index=u_pick_index,
                                 u625_pick_index=u625_pick_index,
                                 object_specs=object_specs,
                                 task_list=task_list,
                                 frame_list=frame_list,
                                 score=score)
    finally:
        robot.restore_state()


def search_pick_pose_plan(robot, obstacle_list):
    best_plan = None
    failures = []
    for u_pick_index, u625_pick_index in itertools.product(range(len(params.U_GRASP_POSITIONS)),
                                                           range(len(params.U625_GRASP_POSITIONS))):
        try:
            candidate_plan = plan_candidate(robot, obstacle_list, u_pick_index, u625_pick_index)
        except RuntimeError as error:
            failures.append((u_pick_index, u625_pick_index, str(error)))
            print(f"Candidate u#{u_pick_index}/U625#{u625_pick_index} rejected: {error}")
            continue
        if best_plan is None or candidate_plan.score < best_plan.score:
            best_plan = candidate_plan
    if best_plan is None:
        raise PickPoseSearchError("No pick-pose pair produced a complete dual-arm plan.", failures)
    print("Selected coordinated pick-pose pair "
          f"u#{best_plan.u_pick_index}, U625#{best_plan.u625_pick_index}; "
          f"score={best_plan.score:.3f}.")
    print(f"Rejected {len(failures)} pick-pose pairs during coordinated search.")
    return best_plan, failures


def start_animation_on_space(animation_data):
    if getattr(animation_data, "started", False):
        return
    animation_data.started = True
    print("Space pressed; animation playback started.")
    pp.base.taskMgr.doMethodLater(.05,
                                  pp.update,
                                  "ur7e_dh50_pickpose_search_update",
                                  extraArgs=[animation_data],
                                  appendTask=True)


def main(toggle_visual=True,
         vertical_frame_height=params.RACK_VERTICAL_FRAME_HEIGHT,
         vertical_frame_x_length=params.RACK_VERTICAL_FRAME_X_LENGTH,
         vertical_frame_y_length=params.RACK_VERTICAL_FRAME_Y_LENGTH,
         horizontal_frame_thickness=params.RACK_HORIZONTAL_FRAME_THICKNESS,
         horizontal_frame_x_length=params.RACK_HORIZONTAL_FRAME_X_LENGTH,
         horizontal_frame_y_length=params.RACK_HORIZONTAL_FRAME_Y_LENGTH):
    pp.base = wd.World(cam_pos=[1.8, 1.6, 1.35], lookat_pos=[0.35, 0.0, 0.95])
    mgm.gen_frame().attach_to(pp.base)

    robot = build_robot(vertical_frame_height=vertical_frame_height,
                        vertical_frame_x_length=vertical_frame_x_length,
                        vertical_frame_y_length=vertical_frame_y_length,
                        horizontal_frame_thickness=horizontal_frame_thickness,
                        horizontal_frame_x_length=horizontal_frame_x_length,
                        horizontal_frame_y_length=horizontal_frame_y_length)
    planning_obstacle_list = build_planning_static_obstacles()
    selected_plan, failures = search_pick_pose_plan(robot, planning_obstacle_list)

    with _TemporaryPickPlaceSpecs(selected_plan.object_specs):
        _, payload_dict = pp.build_inside_scene(pp.base)

    print(f"Generated {len(selected_plan.frame_list)} frames from coordinated pick-pose search.")
    if toggle_visual:
        task_dict = {task.object_name: task for task in selected_plan.task_list}
        robot_mesh_list = pp.precompute_robot_meshes(robot, selected_plan.frame_list)
        animation_data = pp.AnimationData(robot=robot,
                                          robot_mesh_list=robot_mesh_list,
                                          frame_list=selected_plan.frame_list,
                                          payload_dict=payload_dict,
                                          task_dict=task_dict)
        animation_data.started = False
        pp.apply_frame_state(robot, selected_plan.frame_list[0])
        pp.apply_payload_state(robot, selected_plan.frame_list[0], payload_dict, task_dict)
        animation_data.current_robot_mesh = robot_mesh_list[0]
        animation_data.current_robot_mesh.attach_to(pp.base)
        pp.base.accept("space", start_animation_on_space, [animation_data])
        print("Press space in the Panda3D window to start playback.")
        pp.base.run()
    return selected_plan.frame_list, failures


if __name__ == "__main__":
    main(toggle_visual=True)
