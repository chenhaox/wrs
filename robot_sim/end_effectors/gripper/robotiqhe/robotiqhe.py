import os
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp
import modeling.geometric_model as gm
import modeling.collision_model as cm


class RobotiqHE(gp.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type='box',
                 name='robotiqhe',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.jnts[1]['loc_pos'] = coupling_offset_pos
        self.coupling.jnts[1]['gl_rotmat'] = coupling_offset_rotmat
        self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
                                                                self.coupling.jnts[1]['loc_pos'],
                                                                radius=.07, rgba=[.2, .2, .2, 1],
                                                                n_sec=24)
        self.coupling.finalize()
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='base_lft_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([-.025, .0, .11])
        self.lft.jnts[1]['end_type'] = 'prismatic'
        self.lft.jnts[1]['motion_range'] = [0, .025]
        self.lft.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_cvt.stl")
        self.lft.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger1_cvt.stl")
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # - rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([.025, .0, .11])
        self.rgt.jnts[1]['end_type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger2_cvt.stl")
        self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # jaw range
        self.jaw_range = [0.0, 0.05]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .14]) + coupling_offset_pos
        # reinitialize
        self.lft.finalize()
        self.rgt.finalize()
        # collision detection
        self.all_cdelements = []
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
            self.all_cdelements = self.cc.cce_dict
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jawwidth=None):
        self.pos = pos
        self.rotmat = rotmat
        if jawwidth is not None:
            side_jawwidth = (self.jaw_range[1] - jawwidth) / 2.0
            if 0 <= side_jawwidth <= self.jaw_range[1]/2.0:
                self.lft.jnts[1]['motion_value'] = side_jawwidth;
                self.rgt.jnts[1]['motion_value'] = self.lft.jnts[1]['motion_value']
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft.jnts[1]['motion_range'][0] <= motion_val <= self.lft.jnts[1]['motion_range'][1]:
            self.lft.jnts[1]['motion_value'] = motion_val
            self.rgt.jnts[1]['motion_value'] = self.lft.jnts[1]['motion_value']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_value parameter is out of range!")

    def change_jaw_width(self, jaw_width):
        if jaw_width > self.jaw_range[1]:
            raise ValueError("The jawwidth parameter is out of range!")
        self.fk(motion_val=(self.jaw_range[1] - jaw_width) / 2.0)

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
        self.lft.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self.pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(stickmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='robotiqe_mesh_model'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jnt_frames,
                                     rgba=rgba).attach_to(meshmodel)
        self.lft.gen_mesh_model(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_mesh_model(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(meshmodel)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self.pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(meshmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import math

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = RobotiqHE(coupling_offset_pos=np.array([0, 0, 0.0331]),
                     coupling_offset_rotmat=rm.rotmat_from_axangle([1, 0, 0], math.pi / 6), enable_cc=True)
    grpr.change_jaw_width(.05)
    grpr.gen_meshmodel().attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()
