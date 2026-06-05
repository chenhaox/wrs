import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wrs import wd, mgm
import wrs.basis.robot_math as rm
import wrs.manipulation.reachability.rm4d as rm4d
from wrs.robot_sim.robots.ur7e.dual_ur7e_dh50 import DualUR7EDH50
from yanpu.ur7e_dh50_pickandplace_inside import build_inside_scene

from yanpu.ur7e_dh50_pickandplace_params import (
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
    UR3_DUAL_LFT_HOME_CONF,
    UR3_DUAL_RGT_HOME_CONF,
)


REACHABILITY_MAP_PATH = os.path.join(project_root,
                                     "wrs",
                                     "manipulation",
                                     "reachability",
                                     "rm4d_data",
                                     "ur7e_rm4d",
                                     "rmap.npy")
LFT_WORLD_Z_SCORE_NAME = "lft_world_z_linear_manipulability"
RGT_WORLD_Z_SCORE_NAME = "rgt_world_z_linear_manipulability"


def build_robot(enable_cc=True):
    robot = DualUR7EDH50(enable_cc=enable_cc,
                         body_root_pos=RACK_BASE_POS,
                         body_root_rotmat=RACK_ROT,
                         vertical_frame_height=RACK_VERTICAL_FRAME_HEIGHT,
                         vertical_frame_xy=RACK_VERTICAL_FRAME_XY,
                         vertical_frame_x_length=RACK_VERTICAL_FRAME_X_LENGTH,
                         vertical_frame_y_length=RACK_VERTICAL_FRAME_Y_LENGTH,
                         horizontal_frame_thickness=RACK_HORIZONTAL_FRAME_THICKNESS,
                         horizontal_frame_x_length=RACK_HORIZONTAL_FRAME_X_LENGTH,
                         horizontal_frame_y_length=RACK_HORIZONTAL_FRAME_Y_LENGTH,
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
    robot.lft_arm.goto_given_conf(np.asarray(UR3_DUAL_LFT_HOME_CONF, dtype=float))
    robot.rgt_arm.goto_given_conf(np.asarray(UR3_DUAL_RGT_HOME_CONF, dtype=float))
    robot.lft_arm.hndopen()
    robot.rgt_arm.hndopen()
    return robot


def load_reachability_map():
    if not os.path.exists(REACHABILITY_MAP_PATH):
        print("Reachability map not found; skip RM4D query:", REACHABILITY_MAP_PATH)
        return None
    return rm4d.ReachabilityMap4D.from_file(REACHABILITY_MAP_PATH)


def tcp_pose_in_arm_base(arm):
    manipulator = getattr(arm, "arm", arm)
    arm_base_homomat = rm.homomat_from_posrot(manipulator.pos, manipulator.rotmat)
    tcp_homomat = rm.homomat_from_posrot(arm.gl_tcp_pos, arm.gl_tcp_rotmat)
    return np.linalg.inv(arm_base_homomat) @ tcp_homomat


def world_z_axis_in_arm_base(arm):
    manipulator = getattr(arm, "arm", arm)
    axis = manipulator.rotmat.T @ rm.const.z_ax
    return axis / np.linalg.norm(axis)


def query_reachability(rmap, arm):
    if rmap is None:
        return None
    tcp_homomat = tcp_pose_in_arm_base(arm)
    try:
        return rmap.is_reachable_world_coords(tcp_homomat)
    except IndexError as error:
        return f"out_of_map ({error})"


def query_world_z_manipulability(rmap, arm, score_name):
    if rmap is None:
        return None
    if not rmap.has_named_score_map(score_name):
        return f"score_map_unavailable ({score_name})"
    tcp_homomat = tcp_pose_in_arm_base(arm)
    try:
        return rmap.get_score_world_coords(tcp_homomat, score_name=score_name)
    except IndexError as error:
        return f"out_of_map ({error})"


def attach_reachability_volume(base, arm, rmap, label, rgb, score_name):
    if rmap is None:
        return
    manipulator = getattr(arm, "arm", arm)
    try:
        rmap.gen_volume_visualization(base_origin=manipulator.pos,
                                      base_rotmat=manipulator.rotmat,
                                      occupied_rgb=rgb,
                                      alpha=.08,
                                      step_z=4,
                                      step_xy=4,
                                      use_score_map=True,
                                      score_name=score_name).attach_to(base)
    except ValueError as error:
        print(f"{label} world-Z manipulability map is not plotted: {error}")
    mgm.gen_frame(pos=manipulator.pos,
                  rotmat=manipulator.rotmat,
                  ax_length=.12,
                  ax_radius=.004).attach_to(base)
    print(f"{label} reachability map base pos:", manipulator.pos)
    print(f"{label} world Z axis in arm base:", world_z_axis_in_arm_base(arm))
    print(f"{label} expected RM4D score map:", score_name)
    print(f"{label} TCP pose in arm base:", tcp_pose_in_arm_base(arm)[:3, 3])
    print(f"{label} RM4D reachability:", query_reachability(rmap, arm))
    print(f"{label} RM4D world-Z manipulability:", query_world_z_manipulability(rmap, arm, score_name))


def main():
    base = wd.World(cam_pos=[2.0, -1.8, 1.55], lookat_pos=RACK_BASE_POS + np.array([0.0, 0.0, 0.55]))
    mgm.gen_frame().attach_to(base)
    mgm.gen_frame(pos=RACK_BASE_POS, rotmat=RACK_ROT, ax_length=.18, ax_radius=.006).attach_to(base)

    robot = build_robot(enable_cc=True)
    robot.gen_meshmodel(alpha=.9,
                        toggle_tcp_frame=True,
                        toggle_jnt_frames=True,
                        toggle_flange_frame=True,
                        toggle_cdprim=True).attach_to(base)
    obstacle_list, payload_dict = build_inside_scene(base)
    rmap = load_reachability_map()
    attach_reachability_volume(base,
                               robot.lft_arm,
                               rmap,
                               label="Left arm",
                               rgb=np.array([.0, .72, .22]),
                               score_name=LFT_WORLD_Z_SCORE_NAME)
    attach_reachability_volume(base,
                               robot.rgt_arm,
                               rmap,
                               label="Right arm",
                               rgb=np.array([1.0, .08, .04]),
                               score_name=RGT_WORLD_Z_SCORE_NAME)

    print("Rack base pos:", RACK_BASE_POS)
    print("Vertical frame xyz:",
          np.array([RACK_VERTICAL_FRAME_X_LENGTH, RACK_VERTICAL_FRAME_Y_LENGTH, RACK_VERTICAL_FRAME_HEIGHT]))
    print("Horizontal frame xyz:",
          np.array([RACK_HORIZONTAL_FRAME_X_LENGTH, RACK_HORIZONTAL_FRAME_Y_LENGTH, RACK_HORIZONTAL_FRAME_THICKNESS]))
    print("Left mount pos:", robot.lft_mount_pos)
    print("Right mount pos:", robot.rgt_mount_pos)
    print("Left TCP:", robot.lft_arm.gl_tcp_pos)
    print("Right TCP:", robot.rgt_arm.gl_tcp_pos)
    print("Obstacle count:", len(obstacle_list))
    print("Payloads:", list(payload_dict))
    print("Self/environment collision:", robot.is_collided())
    base.run()


if __name__ == "__main__":
    main()
