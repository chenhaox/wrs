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
import wrs.modeling.model_collection as mmc
from wrs.motion.probabilistic import multi_arm_rrt_connect as marrtc
from wrs.motion.probabilistic import rrt_connect as rrtc
from wrs.robot_sim.end_effectors.grippers.dh50.dh50 import Dh50
from wrs.robot_sim.robots.ur7e._ur7e_common import UR7EBase


MESH_DIR = os.path.join(project_root, "old_version", "rbt", "ur7e", "meshes")
MODEL_DIR = os.path.join(project_root, "yanpu", "models")

PICK_CONF = np.array([0.16862, -1.8969, 1.5116, -1.1859, -1.5709, -1.4026])
TRANSFER_CONF = np.array([-0.45, -1.9, 1.4, -1.1, -1.57, -1.7])
PLACE_CONF = np.array([0.8, -1.7, 1.3, -1.2, -1.57, -0.7])
BOX_RGBA = np.array([.05, .24, .56, .72])
BOX1_CENTER = np.array([0.807, -0.245, 0.8])
BOX2_CENTER = np.array([0.232, 0.32, 0.72])
BOX1_PLACE = np.array([0.323, -0.36, 1.0])
BOX2_PLACE = np.array([-0.094, -0.149, 0.9])
OBJECT_UP_DOWN_ROTMAT = rm.rotmat_from_euler(0, np.pi, 0)
U_POSE = (BOX2_PLACE, OBJECT_UP_DOWN_ROTMAT)
U625_POSE = (BOX1_PLACE, OBJECT_UP_DOWN_ROTMAT)
U_RGBA = np.array([.02, .58, .72, .96])
U625_RGBA = np.array([.95, .48, .08, .96])
DEMO_OBJECT_NAME = "U625"
RACK_BASE_POS = np.array([0.7, 0.2, 0.7])
RACK_YAW = -np.pi / 4.0
RACK_ROT = rm.rotmat_from_axangle(rm.const.z_ax, RACK_YAW)
RACK_VERTICAL_FRAME_HEIGHT = 0.7
RACK_VERTICAL_FRAME_XY = np.array([.18, .18])
RACK_HORIZONTAL_FRAME_X_LENGTH = .24
RACK_HORIZONTAL_FRAME_Y_LENGTH = .70
RACK_HORIZONTAL_FRAME_THICKNESS = .08
RACK_VERTICAL_FRAME_RGB = np.array([.60, .62, .62])
RACK_HORIZONTAL_FRAME_RGB = np.array([.05, .16, .32])
RACK_VERTICAL_FRAME_ALPHA = .58
RACK_HORIZONTAL_FRAME_ALPHA = .86
RACK_COLUMN_HEIGHT = RACK_VERTICAL_FRAME_HEIGHT
RACK_ARM_Y_OFFSET = 0.258485281374
RACK_LFT_ARM_LOC_POS = np.array([0.0, RACK_ARM_Y_OFFSET, RACK_COLUMN_HEIGHT])
RACK_RGT_ARM_LOC_POS = np.array([0.0, -RACK_ARM_Y_OFFSET, RACK_COLUMN_HEIGHT])
RACK_LFT_ARM_LOC_ROTMAT = rm.rotmat_from_euler(-3.0 * np.pi / 4.0, 0, 0)
RACK_RGT_ARM_LOC_ROTMAT = (rm.rotmat_from_euler(3.0 * np.pi / 4.0, 0, 0) @
                           rm.rotmat_from_euler(0, 0, np.pi))
BOX1_PART_OFFSETS = {
    "1": np.array([0.0, -0.017, 0.02]),
    "2": np.array([0.0, 0.013, 0.02]),
    "3": np.array([-0.013, 0.0, 0.02]),
    "4": np.array([0.013, 0.0, 0.02]),
    "5": np.zeros(3),
}
BOX2_PART_OFFSETS = {
    "1": np.array([0.007, 0.0, 0.02]),
    "2": np.array([-0.01, 0.0, 0.02]),
    "3": np.array([0.0, -0.017, 0.02]),
    "4": np.array([0.0, 0.005, 0.02]),
    "5": np.zeros(3),
}
U_PLACE_POSITIONS = [BOX1_PLACE,
                     BOX1_PLACE + np.array([-0.1, 0.0, 0.0]),
                     BOX1_PLACE + np.array([-0.2, 0.0, 0.0])]
U_GRASP_POSITIONS = [BOX1_CENTER + np.array([0.2, 0.1, 0.0]),
                     BOX1_CENTER + np.array([-0.2, 0.1, 0.0]),
                     BOX1_CENTER + np.array([0.2, -0.1, 0.0]),
                     BOX1_CENTER + np.array([-0.2, -0.1, 0.0])]
U625_PLACE_POSITIONS = [BOX2_PLACE,
                        BOX2_PLACE + np.array([0.0, 0.1, 0.0]),
                        BOX2_PLACE + np.array([0.0, 0.2, 0.0])]
U625_GRASP_POSITIONS = [BOX2_CENTER + np.array([0.1, 0.0, 0.0]),
                        BOX2_CENTER + np.array([-0.1, 0.2, 0.0]),
                        BOX2_CENTER + np.array([0.1, -0.2, 0.0]),
                        BOX2_CENTER + np.array([-0.1, -0.2, 0.0])]
OBJECT_SPECS = {
    "u": {
        "mesh": "u.STL",
        "pose": U_POSE,
        "rgba": U_RGBA,
        "grasp_pickle": "U1_dh50.pickle",
        "grasp_key": "u",
        "grasp_positions": U_GRASP_POSITIONS,
        "place_positions": U_PLACE_POSITIONS,
    },
    "U625": {
        "mesh": "U625.STL",
        "pose": U625_POSE,
        "rgba": U625_RGBA,
        "grasp_pickle": "U625_dh50.pickle",
        "grasp_key": "U625",
        "grasp_positions": U625_GRASP_POSITIONS,
        "place_positions": U625_PLACE_POSITIONS,
    },
}


class DualUR7EDH50:

    def __init__(self,
                 name="dual_ur7e_dh50",
                 enable_cc=True,
                 vertical_frame_height=RACK_VERTICAL_FRAME_HEIGHT,
                 horizontal_frame_thickness=RACK_HORIZONTAL_FRAME_THICKNESS,
                 horizontal_frame_x_length=RACK_HORIZONTAL_FRAME_X_LENGTH,
                 horizontal_frame_y_length=RACK_HORIZONTAL_FRAME_Y_LENGTH):
        self.name = name
        self.body_root_pos = RACK_BASE_POS
        self.body_root_rotmat = RACK_ROT
        self.vertical_frame_height = float(vertical_frame_height)
        self.horizontal_frame_thickness = float(horizontal_frame_thickness)
        self.horizontal_frame_x_length = float(horizontal_frame_x_length)
        self.horizontal_frame_y_length = float(horizontal_frame_y_length)
        self.column_height = self.vertical_frame_height
        self.lft_mount_loc_pos = np.array([0.0, RACK_ARM_Y_OFFSET, self.vertical_frame_height])
        self.rgt_mount_loc_pos = np.array([0.0, -RACK_ARM_Y_OFFSET, self.vertical_frame_height])
        self.lft_mount_loc_rotmat = RACK_LFT_ARM_LOC_ROTMAT
        self.rgt_mount_loc_rotmat = RACK_RGT_ARM_LOC_ROTMAT
        self.lft_mount_pos, self.lft_mount_rotmat = self._rack_pose_to_world(self.lft_mount_loc_pos,
                                                                             self.lft_mount_loc_rotmat)
        self.rgt_mount_pos, self.rgt_mount_rotmat = self._rack_pose_to_world(self.rgt_mount_loc_pos,
                                                                             self.rgt_mount_loc_rotmat)
        self.lft_arm = self._make_arm(name + "_lft",
                                      self.lft_mount_pos,
                                      self.lft_mount_rotmat,
                                      np.array([np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0,
                                                -np.pi, -np.pi / 2.0, 0.0]),
                                      enable_cc=enable_cc)
        self.rgt_arm = self._make_arm(name + "_rgt",
                                      self.rgt_mount_pos,
                                      self.rgt_mount_rotmat,
                                      np.array([0.0, -np.pi / 2.0, np.pi / 2.0,
                                                -np.pi / 2.0, 0.0, 0.0]),
                                      enable_cc=enable_cc)
        self.arm_dict = {"lft_arm": self.lft_arm, "rgt_arm": self.rgt_arm}
        self.manipulator_dict = self.arm_dict.copy()
        self.set_active_arm("rgt_arm")
        self.lft_arm.hndopen()
        self.rgt_arm.hndopen()

    def _rack_pose_to_world(self, loc_pos, loc_rotmat):
        return (self.body_root_pos + self.body_root_rotmat @ loc_pos,
                self.body_root_rotmat @ loc_rotmat)

    @staticmethod
    def _make_arm(name, mount_pos, mount_rotmat, home_conf, enable_cc):
        return UR7EBase(pos=np.zeros(3),
                        rotmat=np.eye(3),
                        name=name,
                        enable_cc=enable_cc,
                        arm_home_conf=home_conf,
                        arm_loc_pos=mount_pos,
                        arm_loc_rotmat=mount_rotmat,
                        hnd_cls=Dh50,
                        hnd_loc_rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.pi / 2),
                        ik_solver="n")

    def set_active_arm(self, arm_name):
        if arm_name not in self.arm_dict:
            raise ValueError(f"Unknown arm name: {arm_name}")
        self.active_arm_name = arm_name
        self.active_arm = self.arm_dict[arm_name]
        self.arm = self.active_arm.arm
        self.hnd = self.active_arm.hnd

    @property
    def gl_tcp_pos(self):
        return self.active_arm.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self.active_arm.gl_tcp_rotmat

    @property
    def oiee_list(self):
        return self.active_arm.oiee_list

    def backup_state(self):
        self.lft_arm.backup_state()
        self.rgt_arm.backup_state()

    def restore_state(self):
        self.rgt_arm.restore_state()
        self.lft_arm.restore_state()

    def get_jnt_values(self):
        return self.active_arm.get_jnt_values()

    def rand_conf(self):
        return self.active_arm.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        return self.active_arm.are_jnts_in_ranges(jnt_values)

    def fk(self, component_name="arm", jnt_values=None):
        if jnt_values is None and not isinstance(component_name, str):
            jnt_values = component_name
        return self.goto_given_conf(jnt_values)

    def goto_given_conf(self, jnt_values):
        return self.active_arm.goto_given_conf(jnt_values)

    def goto_conf_dict(self, conf_dict):
        for arm_name, conf in conf_dict.items():
            self.arm_dict[arm_name].goto_given_conf(conf)

    def get_ee_values(self):
        return self.active_arm.get_ee_values()

    def change_ee_values(self, ee_values):
        return self.active_arm.change_ee_values(ee_values)

    def hndopen(self):
        self.active_arm.hndopen()

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        if obstacle_list is None:
            obstacle_list = []
        external_robot_list = [] if other_robot_list is None else list(other_robot_list)
        contacts = []
        for arm_name, arm in self.arm_dict.items():
            robot_list = external_robot_list + [
                other_arm for other_name, other_arm in self.arm_dict.items()
                if other_name != arm_name and other_arm.cc is not None
            ]
            result = arm.is_collided(obstacle_list=obstacle_list,
                                     other_robot_list=robot_list,
                                     toggle_contacts=toggle_contacts,
                                     toggle_dbg=toggle_dbg)
            if toggle_contacts:
                if result[0]:
                    contacts.extend(result[1])
            elif result:
                return True
        return (len(contacts) > 0, contacts) if toggle_contacts else False

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=True,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        mcm.gen_box(xyz_lengths=np.array([RACK_VERTICAL_FRAME_XY[0],
                                           RACK_VERTICAL_FRAME_XY[1],
                                           self.vertical_frame_height]),
                    pos=self.body_root_pos + self.body_root_rotmat @ np.array([0.0, 0.0,
                                                                               self.vertical_frame_height / 2.0]),
                    rotmat=self.body_root_rotmat,
                    rgb=RACK_VERTICAL_FRAME_RGB,
                    alpha=RACK_VERTICAL_FRAME_ALPHA).attach_to(m_col)
        mcm.gen_box(xyz_lengths=np.array([self.horizontal_frame_x_length,
                                           self.horizontal_frame_y_length,
                                           self.horizontal_frame_thickness]),
                    pos=self.body_root_pos + self.body_root_rotmat @ np.array(
                        [0.0, 0.0, self.vertical_frame_height + self.horizontal_frame_thickness / 2.0]),
                    rotmat=self.body_root_rotmat,
                    rgb=RACK_HORIZONTAL_FRAME_RGB,
                    alpha=RACK_HORIZONTAL_FRAME_ALPHA).attach_to(m_col)
        self.lft_arm.gen_meshmodel(alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.rgt_arm.gen_meshmodel(alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


@dataclass
class FrameState:
    conf: np.ndarray
    jaw_width: float
    payload_mode: str


class AnimationData:

    def __init__(self, robot, robot_mesh_list, frame_list, payload, pick_pose, place_pose, payload_rel_pose):
        self.robot = robot
        self.robot_mesh_list = robot_mesh_list
        self.frame_list = frame_list
        self.payload = payload
        self.pick_pose = pick_pose
        self.place_pose = place_pose
        self.payload_rel_pose = payload_rel_pose
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
    obj_pos, obj_rotmat = spec["pose"]
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
                                                         pos=spec["pose"][0],
                                                         rotmat=spec["pose"][1],
                                                         rgba=spec["rgba"],
                                                         attach_to=base,
                                                         mesh_dir=MODEL_DIR)
        attach_position_markers(base,
                                spec["grasp_positions"],
                                rgb=spec["rgba"][:3],
                                alpha=.42,
                                radius=.012)
        attach_position_markers(base,
                                spec["place_positions"],
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


def plan_segment(planner, start_conf, goal_conf, obstacle_list):
    mot_data = planner.plan(start_conf=start_conf,
                            goal_conf=goal_conf,
                            obstacle_list=obstacle_list,
                            ext_dist=.55,
                            max_time=4.0,
                            smoothing_n_iter=0,
                            toggle_dbg=False)
    if mot_data is None:
        print("RRT failed for one segment; falling back to a joint-space interpolation.")
        return list(np.linspace(start_conf, goal_conf, 45))
    return mot_data.jv_list


def append_path(frame_list, path, jaw_width, payload_mode, skip_first=True):
    path = path[1:] if skip_first and len(path) > 1 else path
    for conf in path:
        frame_list.append(FrameState(conf=np.asarray(conf), jaw_width=jaw_width, payload_mode=payload_mode))


def append_jaw_motion(frame_list, conf, start_width, end_width, payload_mode, n_frames=12):
    for jaw_width in np.linspace(start_width, end_width, n_frames):
        frame_list.append(FrameState(conf=np.asarray(conf), jaw_width=float(jaw_width), payload_mode=payload_mode))


def build_frame_list(robot, obstacle_list):
    random.seed(4)
    np.random.seed(4)
    open_width = robot.hnd.jaw_range[1]
    closed_width = robot.hnd.jaw_range[0]
    planner = rrtc.RRTConnect(robot)
    planner.rbt = robot

    home_conf = robot.arm.home_conf.copy()
    home_to_pick = plan_segment(planner, home_conf, PICK_CONF, obstacle_list)
    pick_to_transfer = plan_segment(planner, PICK_CONF, TRANSFER_CONF, obstacle_list)
    transfer_to_place = plan_segment(planner, TRANSFER_CONF, PLACE_CONF, obstacle_list)

    frame_list = []
    append_path(frame_list, home_to_pick, open_width, "pick", skip_first=False)
    append_jaw_motion(frame_list, PICK_CONF, open_width, closed_width, "pick")
    append_path(frame_list, pick_to_transfer, closed_width, "hold")
    append_path(frame_list, transfer_to_place, closed_width, "hold")
    append_jaw_motion(frame_list, PLACE_CONF, closed_width, open_width, "hold")
    append_jaw_motion(frame_list, PLACE_CONF, open_width, open_width, "place", n_frames=24)
    return frame_list


def precompute_robot_meshes(robot, frame_list):
    mesh_list = []
    robot.backup_state()
    for frame in frame_list:
        robot.goto_given_conf(frame.conf)
        robot.change_ee_values(frame.jaw_width)
        mesh_list.append(robot.gen_meshmodel(alpha=.88, toggle_tcp_frame=True))
    robot.restore_state()
    return mesh_list


def update(animation_data, task):
    frame = animation_data.frame_list[animation_data.counter]
    if animation_data.current_robot_mesh is not None:
        animation_data.current_robot_mesh.detach()
    animation_data.current_robot_mesh = animation_data.robot_mesh_list[animation_data.counter]
    animation_data.current_robot_mesh.attach_to(base)

    if frame.payload_mode == "hold":
        rel_pos, rel_rotmat = animation_data.payload_rel_pose
        animation_data.robot.goto_given_conf(frame.conf)
        animation_data.robot.change_ee_values(frame.jaw_width)
        payload_pos, payload_rotmat = animation_data.robot.arm.cvt_pose_in_tcp_to_gl(rel_pos, rel_rotmat)
        animation_data.payload.pose = (payload_pos, payload_rotmat)
    elif frame.payload_mode == "place":
        animation_data.payload.pose = animation_data.place_pose
    else:
        animation_data.payload.pose = animation_data.pick_pose

    if animation_data.counter == len(animation_data.frame_list) - 1:
        animation_data.end_hold_counter += 1
        if animation_data.end_hold_counter >= 24:
            animation_data.counter = 0
            animation_data.end_hold_counter = 0
    else:
        animation_data.counter += 1
    return task.again


def main(toggle_visual=True):
    global base
    base = wd.World(cam_pos=[1.8, 1.6, 1.35], lookat_pos=[0.35, 0.0, 0.95])
    mgm.gen_frame().attach_to(base)

    robot = DualUR7EDH50(enable_cc=True)
    robot.hndopen()

    demo_spec = OBJECT_SPECS[DEMO_OBJECT_NAME]
    pick_pose = demo_spec["pose"]
    place_pose = (demo_spec["place_positions"][0], demo_spec["pose"][1])
    robot.goto_given_conf(PICK_CONF)
    payload_rel_pose = robot.arm.cvt_gl_pose_to_tcp(pick_pose[0], pick_pose[1])

    obstacle_list, payload_dict = build_inside_scene(base)
    payload = payload_dict[DEMO_OBJECT_NAME]
    frame_list = build_frame_list(robot, obstacle_list)
    print(f"Generated {len(frame_list)} animation frames for UR7E + DH50.")

    if toggle_visual:
        robot_mesh_list = precompute_robot_meshes(robot, frame_list)
        animation_data = AnimationData(robot=robot,
                                       robot_mesh_list=robot_mesh_list,
                                       frame_list=frame_list,
                                       payload=payload,
                                       pick_pose=pick_pose,
                                       place_pose=place_pose,
                                       payload_rel_pose=payload_rel_pose)
        taskMgr.doMethodLater(.05,
                              update,
                              "ur7e_dh50_pickandplace_inside_update",
                              extraArgs=[animation_data],
                              appendTask=True)
        base.run()
    return frame_list


if __name__ == "__main__":
    main(toggle_visual=True)
