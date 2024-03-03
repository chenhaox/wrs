import os
import math
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import robot_sim.end_effectors.gripper.cobotta_gripper.cobotta_gripper as cbtg
import robot_sim.robots.single_arm_robot_interface as ri


class Cobotta(ri.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = cbta.CobottaArm(pos=self.pos, rotmat=self.rotmat, home_conf=home_conf, name="cobotta_arm",
                                           enable_cc=False)
        self.end_effector = cbtg.CobottaGripper(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat, name="cobotta_hnd")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # TODO when pose is changed, oih info goes wrong
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk)
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, el0, el1, ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # TODO oiee?

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self._update_end_effector()

    # def hold(self, hnd_name, obj_cmodel, jawwidth=None):
    #     """
    #     the obj_cmodel is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param obj_cmodel:
    #     :return:
    #     """
    #     if hnd_name not in self.hnd_dict:
    #         raise ValueError("Hand name does not exist!")
    #     if jawwidth is not None:
    #         self.hnd_dict[hnd_name].change_jaw_width(jawwidth)
    #     rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_pose_to_tcp(obj_cmodel.get_pos(), obj_cmodel.get_rotmat())
    #     intolist = [self.arm.lnks[0],
    #                 self.arm.lnks[1],
    #                 self.arm.lnks[2],
    #                 self.arm.lnks[3],
    #                 self.arm.lnks[4]]
    #     self.oih_infos.append(self.cc.add_cdobj(obj_cmodel, rel_pos, rel_rotmat, intolist))
    #     return rel_pos, rel_rotmat

    # def get_oih_list(self):
    #     return_list = []
    #     for obj_info in self.oih_infos:
    #         obj_cmodel = obj_info['collision_model']
    #         obj_cmodel.set_pos(obj_info['gl_pos'])
    #         obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
    #         return_list.append(obj_cmodel)
    #     return return_list
    #
    # def release(self, hnd_name, obj_cmodel, jawwidth=None):
    #     """
    #     the obj_cmodel is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param obj_cmodel:
    #     :return:
    #     """
    #     if hnd_name not in self.hnd_dict:
    #         raise ValueError("Hand name does not exist!")
    #     if jawwidth is not None:
    #         self.hnd_dict[hnd_name].change_jaw_width(jawwidth)
    #     for obj_info in self.oih_infos:
    #         if obj_info['collision_model'] is obj_cmodel:
    #             self.cc.delete_cdobj(obj_info)
    #             self.oih_infos.remove(obj_info)
    #             break

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='single_arm_robot_interface_stickmodel'):
        m_col = mc.ModelCollection(name=name)
        self.manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.end_effector.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                         toggle_jnt_frames=toggle_jnt_frames).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        m_col = mc.ModelCollection(name=name)
        self.manipulator.gen_meshmodel(rgb=rgb,
                                       alpha=alpha,
                                       toggle_tcp_frame=toggle_tcp_frame,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       toggle_cdprim=toggle_cdprim,
                                       toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.end_effector.gen_meshmodel(rgb=rgb,
                                        alpha=alpha,
                                        toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_cdprim=toggle_cdprim,
                                        toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import modeling.collision_model as mcm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = Cobotta(enable_cc=True)
    # robot.jaw_to(.02)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # base.run()
    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    robot.show_cdprim()
    robot.unshow_cdprim()
    # base.run()

    robot.goto_given_conf(jnt_values=np.array([0, np.pi / 2, np.pi*11/20, 0, np.pi / 2, 0]))
    robot.show_cdprim()

    box = mcm.gen_box(xyz_lengths=np.array([0.1,.1,.1]),pos=tgt_pos, rgba=np.array([1,1,0,.3]))
    box.attach_to(base)
    tic=time.time()
    result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
    print(result)
    toc=time.time()
    print(toc-tic)
    for pnt in contacts:
        mgm.gen_sphere(pnt).attach_to(base)

    base.run()
