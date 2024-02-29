import os
import math
import numpy as np
import modeling.collision_model as mcm
import basis.robot_math as rm
import robot_sim.manipulators.manipulator_interface as mi


class CobottaArm(mi.ManipulatorInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 home_conf=np.zeros(6),
                 name='cobotta_arm',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name)
        # anchor
        self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "gripper_base.dae"))
        self.jlc.anchor.lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].motion_range = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_rotmat = np.array([-2.617994, 2.617994])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j1.dae"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].motion_range = np.array([0, 0, 0.18])
        self.jlc.jnts[1].loc_rotmat = np.array([-1.047198, 1.745329])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j2.dae"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_rotmat = np.array([-1.047198, 1.745329])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j3.dae"))
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.jnts[2].motion_range = np.array([0, 0, 0.165])
        # fourth joint and link
        self.jlc.jnts[3].motion_range = np.array([-0.012, 0.02, 0.088])
        self.jlc.jnts[3].loc_rotmat = np.array([-1.047198, 1.745329])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j4.dae"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].motion_range = np.array([0, -.02, .0895])
        self.jlc.jnts[4].loc_rotmat = np.array([-1.047198, 1.745329])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j5.dae"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].motion_range = np.array([0, -.0445, 0.042])
        self.jlc.jnts[5].loc_rotmat = np.array([-1.047198, 1.745329])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(os.getcwd(), "meshes", "j6.dae"))
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.finalize()
        # collision detection
        # if enable_cc:
        #     self.enable_cc()

    # def enable_cc(self):
    #     super().enable_cc()
    #     self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
    #     activelist = [self.jlc.lnks[0],
    #                   self.jlc.lnks[1],
    #                   self.jlc.lnks[2],
    #                   self.jlc.lnks[3],
    #                   self.jlc.lnks[4],
    #                   self.jlc.lnks[5],
    #                   self.jlc.lnks[6]]
    #     self.cc.set_active_cdlnks(activelist)
    #     fromlist = [self.jlc.lnks[0],
    #                 self.jlc.lnks[1]]
    #     intolist = [self.jlc.lnks[3],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[2]]
    #     intolist = [self.jlc.lnks[4],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[3]]
    #     intolist = [self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)
    tmp_arm = CobottaArm(enable_cc=True)
    tmp_arm_mesh = tmp_arm.gen_meshmodel()
    tmp_arm_mesh.attach_to(base)
    tmp_arm_mesh.show_cdprimit()
    # manipulator_instance.gen_stickmodel(toggle_joint_frame=True).attach_to(base)
    # tic = time.time()
    # print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # mgm.GeometricModel("./meshes/base.dae").attach_to(base)
    # mgm.gen_frame().attach_to(base)
    base.run()
