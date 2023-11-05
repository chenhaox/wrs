import os
import math
import numpy as np
import modeling.model_collection as mc
import robot_sim.kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp

class YumiGripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='yumi_gripper', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.joints[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.joints[-1]['gl_rotmatq']
        # - lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='base_lft_finger')
        self.lft.joints[1]['pos_in_loc_tcp'] = np.array([0, 0.0065, 0.0837])
        self.lft.joints[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft.joints[1]['end_type'] = 'prismatic'
        self.lft.joints[1]['motion_rng'] = [.0, .025]
        self.lft.joints[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['pos_in_loc_tcp'] = np.zeros(3)
        self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.lft.lnks[0]['rgba'] = [.5, .5, .5, 1]
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.lft.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='rgt_finger')
        self.rgt.joints[1]['pos_in_loc_tcp'] = np.array([0, -0.0065, 0.0837])
        self.rgt.joints[1]['end_type'] = 'prismatic'
        self.rgt.joints[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.rgt.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft.reinitialize(cdmesh_type=cdmesh_type)
        self.rgt.reinitialize(cdmesh_type=cdmesh_type)
        # jaw range
        self.jaw_range = [0.0, .05]
        # jaw center
        self.jaw_center_pos = np.array([0,0,.13])
        # collision detection
        self.all_cdelements=[]
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0, 1])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.lft.lnks[0],
                          self.lft.lnks[1],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cd_elements
        else:
            self.all_cdelements = [self.lft.lnks[0],
                                   self.lft.lnks[1],
                                   self.rgt.lnks[1]]
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.joints[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.joints[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if self.lft.joints[1]['motion_rng'][0] <= -motion_val <= self.lft.joints[1]['motion_rng'][1]:
            self.lft.joints[1]['motion_val'] = motion_val
            self.rgt.joints[1]['motion_val'] = self.lft.joints[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jawwidth):
        if jawwidth > .05:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=-jawwidth / 2.0)

    def get_jaw_width(self):
        return -self.lft.joints[1]['motion_val']*2

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.lft.gen_mesh_model(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_mesh_model(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = YumiGripper(enable_cc=True)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_euler(math.pi/3, math.pi/3, math.pi/3))
    grpr.jaw_to(.02)
    print(grpr.get_jaw_width())
    grpr.gen_stickmodel().attach_to(base)
    grpr.gen_meshmodel(rgba=[0, .5, 0, .5]).attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], math.pi/3))
    grpr.fix_to(pos=np.zeros(3), rotmat=np.eye(3))
    grpr.gen_meshmodel().attach_to(base)

    grpr2 = grpr.copy()
    grpr2.fix_to(pos=np.array([.3, .3, .2]), rotmat=rm.rotmat_from_axangle([0, 1, 0], .01))
    model = grpr2.gen_mesh_model(rgba=[0.5, .5, 0, .5])
    model.attach_to(base)
    grpr2.show_cdprimit()
    grpr2.show_cdmesh()
    base.run()
