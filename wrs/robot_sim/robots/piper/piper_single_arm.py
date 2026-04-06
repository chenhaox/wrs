#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/10 18:25
# @Author : ZhangXi
import math
import numpy as np
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.robot_sim.robots.single_arm_robot_interface as sari

import wrs.basis.robot_math as rm
from wrs.robot_sim.manipulators.piper.piper import Piper
from wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper import PiperGripper
import wrs.modeling.geometric_model as mgm

class PiperSglArm(sari.SglArmRobotInterface):
    """
    Piper 机械臂整合类：基于 Piper 本体与 PiperGripper 夹爪。
    模仿 RealmanR 结构，提供高层接口（例如 fk、ik、fix_to、goto_given_conf）。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="piper_arm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)
        # home_conf[1] = -math.pi / 3
        # home_conf[2] = math.pi / 2
        # home_conf[4] = math.pi / 6
        # 初始化机械臂
        self.manipulator = Piper(pos=self.pos,
                                 rotmat=self.rotmat,
                                 name=name + "_arm",
                                 enable_cc=True)
        self.manipulator.home_conf = home_conf
        # self.manipulator._ik_solver = None
        # self.manipulator.is_trac_ik = False

        compensation_rotmat = rm.rotmat_from_euler(0, 0, math.pi / 2)  #

        # 将法兰的旋转矩阵与修正旋转矩阵相乘
        # 注意：这里假设您希望夹爪的局部坐标系（rotmat）相对于机械臂法兰（self.manipulator.gl_flange_rotmat）进行旋转
        corrected_rotmat = np.dot(self.manipulator.gl_flange_rotmat, compensation_rotmat)  #

        self.end_effector = PiperGripper(
            pos=self.manipulator.gl_flange_pos,
            rotmat=corrected_rotmat,
            name=name + "_piper_gripper")

        # 设置工具中心点（TCP）
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """Setup collision detection — matching Cobotta/XArm7 pattern.

        Three collision detection mechanisms:
        1. **cdpair** (self-collision): distal links vs proximal links
        2. **extcd** (external): robot links vs obstacles in environment
        3. **innercd**: held objects vs proximal robot links
        """
        # end effector — use cdelements (the proper way)
        ee_cces = []
        for id, cdlnk in enumerate(self.end_effector.cdelements):
            ee_cces.append(self.cc.add_cce(cdlnk))
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        # self-collision: distal (ee + ml3, ml4) vs proximal (base, ml0, ml1)
        from_list = ee_cces + [ml3, ml4]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext collision: enable manipulator links to collide with obstacles
        self.cc.enable_extcd_by_id_list(
            id_list=[ml0, ml1, ml2, ml3, ml4], type="from")
        # inner collision: held objects collide with proximal links
        self.cc.enable_innercd_by_id_list(
            id_list=[mlb, ml0, ml1, ml2], type="into")
        # dynamic_ext_list: EE cces for held-object collision checking
        self.cc.dynamic_ext_list = ee_cces[1:]



    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        """前向运动学"""
        results = self.manipulator.fk(jnt_values=jnt_values,
                                      toggle_jacobian=toggle_jacobian,
                                      update=update)
        if update:
            self.update_end_effector()
        return results

    def fix_to(self, pos, rotmat):
        """固定机械臂基座到指定位置"""
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.basis.robot_math as rm

    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    robot = PiperSglArm(enable_cc=True)
    robot.change_jaw_width(0)
    start_conf = np.array([0, 0, 0, 0, 0, 0])
    robot.goto_given_conf(start_conf)
    robot.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
    tgt_pos = np.array([0, 0.00, 0.7])
    robot.show_cdprim()
    print("is robot collided", robot.is_collided())
    base.run()
    bound_lower = -40
    bound_upper = 40
    grad = 1
    plane_normal = np.array([1, 0, 0])
    goal_conf = None
    print(f"--- Searching IK solutions for position {tgt_pos} ---")
    for theta in range(bound_lower, bound_upper + 1, grad):
        hand_x = np.array([0, 0, -1])
        hand_z = plane_normal
        hand_y = np.cross(hand_z, hand_x)
        tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
        tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
        mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
        current_goal_conf = robot.ik(tgt_pos=tgt_pos,
                                     tgt_rotmat=tgt_rotmat,
                                     seed_jnt_values=start_conf)

        print(f"Theta={theta}°: {'Found' if current_goal_conf is not None else 'Failed'}")

        if current_goal_conf is not None:
            goal_conf = current_goal_conf
            break
    if goal_conf is not None:
        robot.goto_given_conf(jnt_values=goal_conf)
        robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)
    else:
        print(1111)

    base.run()
