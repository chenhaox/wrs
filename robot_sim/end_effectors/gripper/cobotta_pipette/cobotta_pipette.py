import os
import numpy as np
import modeling.model_collection as mc
import robot_sim.kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class CobottaPipette(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='cobotta_pipette', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.joints[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.joints[-1]['gl_rotmatq']
        self.jlc = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(8), name='base_jlc')
        self.jlc.joints[1]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[1]['end_type'] = 'fixed'
        self.jlc.joints[2]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[2]['end_type'] = 'fixed'
        self.jlc.joints[3]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[3]['end_type'] = 'fixed'
        self.jlc.joints[4]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[4]['end_type'] = 'fixed'
        self.jlc.joints[5]['pos_in_loc_tcp'] = np.array([0, -.007, .0])
        self.jlc.joints[5]['end_type'] = 'prismatic'
        self.jlc.joints[5]['motion_rng'] = [0, .015]
        self.jlc.joints[5]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.joints[6]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[6]['end_type'] = 'fixed'
        self.jlc.joints[7]['pos_in_loc_tcp'] = np.array([0, .014, .0])
        self.jlc.joints[7]['end_type'] = 'prismatic'
        self.jlc.joints[7]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.joints[8]['pos_in_loc_tcp'] = np.array([0, .0, .0])
        self.jlc.joints[8]['end_type'] = 'prismatic'
        self.jlc.joints[8]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "pipette_hand_body.stl")
        self.jlc.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.jlc.lnks[1]['name'] = "cam_front"
        self.jlc.lnks[1]['pos_in_loc_tcp'] = np.array([.008, .04, .08575])
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "camera.stl")
        self.jlc.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.jlc.lnks[2]['name'] = "cam_back"
        self.jlc.lnks[2]['pos_in_loc_tcp'] = np.array([.008, .04, .03575])
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "camera.stl")
        self.jlc.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.jlc.lnks[3]['name'] = "pipette_body"
        self.jlc.lnks[3]['pos_in_loc_tcp'] = np.array([.008, .14275, .06075])
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "pipette_body.stl")
        self.jlc.lnks[3]['rgba'] = [.3, .4, .6, 1]
        self.jlc.lnks[4]['name'] = "pipette_shaft"
        self.jlc.lnks[4]['pos_in_loc_tcp'] = np.array([.008, .14275, .06075])
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "pipette_shaft.stl")
        self.jlc.lnks[4]['rgba'] = [1, 1, 1, 1]
        self.jlc.lnks[5]['name'] = "plunge"
        self.jlc.lnks[5]['pos_in_loc_tcp'] = np.array([0, 0, .0])
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "plunge_presser.stl")
        self.jlc.lnks[5]['rgba'] = [.5, .5, .5, 1]
        self.jlc.lnks[6]['name'] = "plunge_button"
        self.jlc.lnks[6]['pos_in_loc_tcp'] = np.array([.008, .14355, .06075])
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "pipette_plunge.stl")
        self.jlc.lnks[6]['rgba'] = [1, 1, 1, 1]
        self.jlc.lnks[7]['name'] = "ejection"
        self.jlc.lnks[7]['pos_in_loc_tcp'] = np.array([0, 0, .0])
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "ejection_presser.stl")
        self.jlc.lnks[7]['rgba'] = [.5, .5, .5, 1]
        self.jlc.lnks[8]['name'] = "ejection_button"
        self.jlc.lnks[8]['pos_in_loc_tcp'] = np.array([.008, .14355, .06075])
        self.jlc.lnks[8]['mesh_file'] = os.path.join(this_dir, "meshes", "pipette_ejection.stl")
        self.jlc.lnks[8]['rgba'] = [1, 1, 1, 1]
        # jaw range
        self.jaw_range = [0.0, .03]
        # jaw center
        self.jaw_center_pos = np.array([0.008, 0.14305, 0.06075])
        self.jaw_center_rotmat = rm.rotmat_from_axangle([1, 0, 0], -np.pi / 2)
        # reinitialize
        self.jlc.reinitialize()
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 7])
            active_list = [self.jlc.lnks[0],
                           self.jlc.lnks[1],
                           self.jlc.lnks[2],
                           self.jlc.lnks[4],
                           self.jlc.lnks[5],
                           self.jlc.lnks[7]]
            self.cc.set_active_cdlnks(active_list)
            self.all_cdelements = self.cc.all_cd_elements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            side_jawwidth = jaw_width / 2.0
            if self.jaw_range[1] < jaw_width or jaw_width < self.jaw_range[0]:
                self.jlc.joints[5]['motion_val'] = side_jawwidth
                self.jlc.joints[7]['motion_val'] = -jaw_width
                if side_jawwidth <= .007:
                    self.jlc.joints[8]['motion_val'] = .0
                else:
                    self.jlc.joints[8]['motion_val'] = (jaw_width - .014) / 2
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.joints[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.joints[-1]['gl_rotmatq']
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def jaw_to(self, jaw_width):
        print(jaw_width)
        if self.jaw_range[1] < jaw_width or jaw_width < self.jaw_range[0]:
            raise ValueError("The jaw_width parameter is out of range!")
        side_jawwidth = jaw_width / 2.0
        self.jlc.joints[5]['motion_val'] = side_jawwidth
        self.jlc.joints[7]['motion_val'] = -jaw_width
        if side_jawwidth <= .007:
            self.jlc.joints[8]['motion_val'] = .0
        else:
            self.jlc.joints[8]['motion_val'] = (jaw_width - .014) / 2
        self.jlc.fk()

    def get_jaw_width(self):
        return -self.jlc.joints[2]['motion_val']

    def gen_stickmodel(self,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='cbtp_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.jlc.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashed_stick(spos=self.pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(stickmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='cbtp_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.jlc.gen_mesh_model(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashed_stick(spos=self.pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(meshmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = CobottaPipette(enable_cc=True)
    grpr.jaw_to(.0)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    grpr.gen_stickmodel().attach_to(base)
    # grpr.gen_stickmodel(toggle_joint_frame=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()
