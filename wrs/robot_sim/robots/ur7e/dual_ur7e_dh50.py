import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
from wrs.robot_sim.end_effectors.grippers.dh50.dh50 import Dh50
from wrs.robot_sim.robots.ur7e._ur7e_common import UR7EBase


DEFAULT_BODY_ROOT_POS = np.array([0.7, 0.2, 0.7])
DEFAULT_BODY_ROOT_ROTMAT = rm.rotmat_from_axangle(rm.const.z_ax, np.pi / 4.0)
DEFAULT_VERTICAL_FRAME_HEIGHT = 0.8
DEFAULT_VERTICAL_FRAME_X_LENGTH = .18
DEFAULT_VERTICAL_FRAME_Y_LENGTH = .18
DEFAULT_VERTICAL_FRAME_XY = np.array([DEFAULT_VERTICAL_FRAME_X_LENGTH, DEFAULT_VERTICAL_FRAME_Y_LENGTH])
DEFAULT_HORIZONTAL_FRAME_X_LENGTH = .24
DEFAULT_HORIZONTAL_FRAME_Y_LENGTH = .70
DEFAULT_HORIZONTAL_FRAME_THICKNESS = .08
DEFAULT_VERTICAL_FRAME_RGB = np.array([.60, .62, .62])
DEFAULT_HORIZONTAL_FRAME_RGB = np.array([.05, .16, .32])
DEFAULT_VERTICAL_FRAME_ALPHA = .58
DEFAULT_HORIZONTAL_FRAME_ALPHA = .86
DEFAULT_ARM_Y_OFFSET = 0.258485281374
DEFAULT_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH = .70
DEFAULT_LFT_ARM_LOC_ROTMAT = rm.rotmat_from_euler(-3.0 * np.pi / 4.0, 0, 0)
DEFAULT_RGT_ARM_LOC_ROTMAT = (rm.rotmat_from_euler(3.0 * np.pi / 4.0, 0, 0) @
                              rm.rotmat_from_euler(0, 0, np.pi))
DEFAULT_LFT_HOME_CONF = np.array([np.pi / 12.0,
                                  -np.pi / 3.0,
                                  -2.0 * np.pi / 3.0,
                                  -np.pi,
                                  -2.0 * np.pi / 3.0,
                                  0.0])
DEFAULT_RGT_HOME_CONF = np.array([-np.pi / 12.0,
                                  -2.0 * np.pi / 3.0,
                                  2.0 * np.pi / 3.0,
                                  0.0,
                                  2.0 * np.pi / 3.0,
                                  np.pi])


class DualUR7EDH50:

    def __init__(self,
                 name="dual_ur7e_dh50",
                 enable_cc=True,
                 body_root_pos=DEFAULT_BODY_ROOT_POS,
                 body_root_rotmat=DEFAULT_BODY_ROOT_ROTMAT,
                 vertical_frame_height=DEFAULT_VERTICAL_FRAME_HEIGHT,
                 vertical_frame_xy=DEFAULT_VERTICAL_FRAME_XY,
                 vertical_frame_x_length=None,
                 vertical_frame_y_length=None,
                 horizontal_frame_thickness=DEFAULT_HORIZONTAL_FRAME_THICKNESS,
                 horizontal_frame_x_length=DEFAULT_HORIZONTAL_FRAME_X_LENGTH,
                 horizontal_frame_y_length=DEFAULT_HORIZONTAL_FRAME_Y_LENGTH,
                 vertical_frame_rgb=DEFAULT_VERTICAL_FRAME_RGB,
                 horizontal_frame_rgb=DEFAULT_HORIZONTAL_FRAME_RGB,
                 vertical_frame_alpha=DEFAULT_VERTICAL_FRAME_ALPHA,
                 horizontal_frame_alpha=DEFAULT_HORIZONTAL_FRAME_ALPHA,
                 arm_y_offset=DEFAULT_ARM_Y_OFFSET,
                 arm_y_offset_reference_frame_y_length=DEFAULT_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH,
                 lft_arm_loc_rotmat=DEFAULT_LFT_ARM_LOC_ROTMAT,
                 rgt_arm_loc_rotmat=DEFAULT_RGT_ARM_LOC_ROTMAT,
                 lft_home_conf=DEFAULT_LFT_HOME_CONF,
                 rgt_home_conf=DEFAULT_RGT_HOME_CONF,
                 ik_solver="tracik"):
        self.name = name
        self.ik_solver = ik_solver
        self.body_root_pos = np.asarray(body_root_pos, dtype=float)
        self.body_root_rotmat = np.asarray(body_root_rotmat, dtype=float)
        self.vertical_frame_height = float(vertical_frame_height)
        self.vertical_frame_xy = np.asarray(vertical_frame_xy, dtype=float).copy()
        if vertical_frame_x_length is not None:
            self.vertical_frame_xy[0] = float(vertical_frame_x_length)
        if vertical_frame_y_length is not None:
            self.vertical_frame_xy[1] = float(vertical_frame_y_length)
        self.vertical_frame_x_length = float(self.vertical_frame_xy[0])
        self.vertical_frame_y_length = float(self.vertical_frame_xy[1])
        self.horizontal_frame_thickness = float(horizontal_frame_thickness)
        self.horizontal_frame_x_length = float(horizontal_frame_x_length)
        self.horizontal_frame_y_length = float(horizontal_frame_y_length)
        self.vertical_frame_rgb = np.asarray(vertical_frame_rgb, dtype=float)
        self.horizontal_frame_rgb = np.asarray(horizontal_frame_rgb, dtype=float)
        self.vertical_frame_alpha = float(vertical_frame_alpha)
        self.horizontal_frame_alpha = float(horizontal_frame_alpha)
        self.base_arm_y_offset = float(arm_y_offset)
        self.arm_y_offset_reference_frame_y_length = float(arm_y_offset_reference_frame_y_length)
        self.column_height = self.vertical_frame_height
        self.arm_y_offset = self._compute_arm_y_offset()
        self.lft_mount_loc_pos = np.array([0.0, self.arm_y_offset, self.vertical_frame_height])
        self.rgt_mount_loc_pos = np.array([0.0, -self.arm_y_offset, self.vertical_frame_height])
        self.lft_mount_loc_rotmat = np.asarray(lft_arm_loc_rotmat, dtype=float)
        self.rgt_mount_loc_rotmat = np.asarray(rgt_arm_loc_rotmat, dtype=float)
        self.lft_mount_pos, self.lft_mount_rotmat = self._rack_pose_to_world(self.lft_mount_loc_pos,
                                                                             self.lft_mount_loc_rotmat)
        self.rgt_mount_pos, self.rgt_mount_rotmat = self._rack_pose_to_world(self.rgt_mount_loc_pos,
                                                                             self.rgt_mount_loc_rotmat)
        self.lft_arm = self._make_arm(name + "_lft",
                                      self.lft_mount_pos,
                                      self.lft_mount_rotmat,
                                      lft_home_conf,
                                      ik_solver=self.ik_solver,
                                      enable_cc=enable_cc)
        self.rgt_arm = self._make_arm(name + "_rgt",
                                      self.rgt_mount_pos,
                                      self.rgt_mount_rotmat,
                                      rgt_home_conf,
                                      ik_solver=self.ik_solver,
                                      enable_cc=enable_cc)
        self.arm_dict = {"lft_arm": self.lft_arm, "rgt_arm": self.rgt_arm}
        self.manipulator_dict = self.arm_dict.copy()
        self.frame_collision_models = self._make_frame_collision_models()
        self.set_active_arm("rgt_arm")
        self.lft_arm.hndopen()
        self.rgt_arm.hndopen()

    def _rack_pose_to_world(self, loc_pos, loc_rotmat):
        return (self.body_root_pos + self.body_root_rotmat @ loc_pos,
                self.body_root_rotmat @ loc_rotmat)

    def _compute_arm_y_offset(self):
        if self.arm_y_offset_reference_frame_y_length <= 0:
            return self.base_arm_y_offset
        return self.base_arm_y_offset * self.horizontal_frame_y_length / self.arm_y_offset_reference_frame_y_length

    def _frame_box_specs(self):
        vertical_pos = self.body_root_pos + self.body_root_rotmat @ np.array([0.0, 0.0,
                                                                              self.vertical_frame_height / 2.0])
        horizontal_pos = self.body_root_pos + self.body_root_rotmat @ np.array(
            [0.0, 0.0, self.vertical_frame_height + self.horizontal_frame_thickness / 2.0])
        return [
            {
                "xyz_lengths": np.array([self.vertical_frame_xy[0],
                                         self.vertical_frame_xy[1],
                                         self.vertical_frame_height]),
                "pos": vertical_pos,
                "rotmat": self.body_root_rotmat,
                "rgb": self.vertical_frame_rgb,
                "alpha": self.vertical_frame_alpha,
            },
            {
                "xyz_lengths": np.array([self.horizontal_frame_x_length,
                                         self.horizontal_frame_y_length,
                                         self.horizontal_frame_thickness]),
                "pos": horizontal_pos,
                "rotmat": self.body_root_rotmat,
                "rgb": self.horizontal_frame_rgb,
                "alpha": self.horizontal_frame_alpha,
            },
        ]

    def _make_frame_collision_models(self):
        return [mcm.gen_box(xyz_lengths=spec["xyz_lengths"],
                            pos=spec["pos"],
                            rotmat=spec["rotmat"],
                            rgb=spec["rgb"],
                            alpha=spec["alpha"])
                for spec in self._frame_box_specs()]

    @staticmethod
    def _make_arm(name, mount_pos, mount_rotmat, home_conf, ik_solver, enable_cc):
        return UR7EBase(pos=np.zeros(3),
                        rotmat=np.eye(3),
                        name=name,
                        enable_cc=enable_cc,
                        arm_home_conf=np.asarray(home_conf, dtype=float),
                        arm_loc_pos=mount_pos,
                        arm_loc_rotmat=mount_rotmat,
                        hnd_cls=Dh50,
                        hnd_loc_rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.pi / 2),
                        ik_solver=ik_solver)

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
        obstacle_list = list(obstacle_list)
        for frame_cmodel in self.frame_collision_models:
            if all(frame_cmodel is not obstacle for obstacle in obstacle_list):
                obstacle_list.append(frame_cmodel)
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
        for frame_cmodel in self.frame_collision_models:
            frame_cmodel.copy().attach_to(m_col)
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


if __name__ == "__main__":
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2.2, -2.0, 1.7], lookat_pos=[0.7, 0.2, 1.1])
    mgm.gen_frame().attach_to(base)
    robot = DualUR7EDH50(enable_cc=True)
    robot.fk("lft_arm", np.array(
        [np.radians(120), 0, 0, 0, 0, 0]
    ))
    robot.gen_meshmodel(alpha=.7,
                        toggle_tcp_frame=True,
                        toggle_jnt_frames=True,
                        toggle_flange_frame=True,
                        toggle_cdprim=True).attach_to(base)

    print("left TCP:", robot.lft_arm.gl_tcp_pos)
    print("right TCP:", robot.rgt_arm.gl_tcp_pos)
    print("collided:", robot.is_collided())
    base.run()
