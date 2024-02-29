import osimport mathimport numpy as npimport basis.robot_math as rmimport robot_sim._kinematics.jlchain as jlimport robot_sim.manipulators.manipulator_interface as miclass XArm7(mi.ManipulatorInterface):    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='xarm7', enable_cc=True):        super().__init__(pos=pos, rotmat=rotmat, name=name)        this_dir, this_filename = os.path.split(__file__)        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)        # seven joints, n_jnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1        jnt_saferngmargin = math.pi / 18.0        self.jlc.jnts[1]['pos_in_loc_tcp'] = np.array([0, 0, .267])        self.jlc.jnts[1]['motion_range'] = [-math.pi + jnt_saferngmargin, math.pi - jnt_saferngmargin]        self.jlc.jnts[2]['pos_in_loc_tcp'] = np.array([0, 0, 0])        self.jlc.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(-1.5708, 0, 0)        self.jlc.jnts[2]['motion_range'] = [-2.18 + jnt_saferngmargin, 2.18 - jnt_saferngmargin]        self.jlc.jnts[3]['pos_in_loc_tcp'] = np.array([0, -.293, 0])        self.jlc.jnts[3]['gl_rotmat'] = rm.rotmat_from_euler(1.5708, 0, 0)        self.jlc.jnts[3]['motion_range'] = [-math.pi + jnt_saferngmargin, math.pi - jnt_saferngmargin]        self.jlc.jnts[4]['pos_in_loc_tcp'] = np.array([.0525, 0, 0])        self.jlc.jnts[4]['gl_rotmat'] = rm.rotmat_from_euler(1.5708, 0, 0)        self.jlc.jnts[4]['motion_range'] = [-0.11 + jnt_saferngmargin, math.pi - jnt_saferngmargin]        self.jlc.jnts[5]['pos_in_loc_tcp'] = np.array([0.0775, -0.3425, 0])        self.jlc.jnts[5]['gl_rotmat'] = rm.rotmat_from_euler(1.5708, 0, 0)        self.jlc.jnts[5]['motion_range'] = [-math.pi + jnt_saferngmargin, math.pi - jnt_saferngmargin]        self.jlc.jnts[6]['pos_in_loc_tcp'] = np.array([0, 0, 0])        self.jlc.jnts[6]['gl_rotmat'] = rm.rotmat_from_euler(1.5708, 0, 0)        self.jlc.jnts[6]['motion_range'] = [-1.75 + jnt_saferngmargin, math.pi - jnt_saferngmargin]        self.jlc.jnts[7]['pos_in_loc_tcp'] = np.array([0.076, 0.097, 0])        self.jlc.jnts[7]['gl_rotmat'] = rm.rotmat_from_euler(-1.5708, 0, 0)        self.jlc.jnts[7]['motion_range'] = [-math.pi + jnt_saferngmargin, math.pi - jnt_saferngmargin]        # links        self.jlc.lnks[0]['name'] = "link_base"        self.jlc.lnks[0]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[0]['com'] = np.array([-0.021131, -0.0016302, 0.056488])        self.jlc.lnks[0]['mass'] = 0.88556        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "link_base.stl")        self.jlc.lnks[0]['rgba'] = [.5, .5, .5, 1.0]        self.jlc.lnks[1]['name'] = "link1"        self.jlc.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[1]['com'] = np.array([-0.0042142, 0.02821, -0.0087788])        self.jlc.lnks[1]['mass'] = 0.42603        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link1.stl")        self.jlc.lnks[2]['name'] = "link2"        self.jlc.lnks[2]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[2]['com'] = np.array([-3.3178e-5, -0.12849, 0.026337])        self.jlc.lnks[2]['mass'] = 0.56095        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link2.stl")        self.jlc.lnks[2]['rgba'] = [.5, .5, .5, 1.0]        self.jlc.lnks[3]['name'] = "link3"        self.jlc.lnks[3]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[3]['com'] = np.array([0.04223, -0.023258, -0.0096674])        self.jlc.lnks[3]['mass'] = 0.44463        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link3.stl")        self.jlc.lnks[4]['name'] = "link4"        self.jlc.lnks[4]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[4]['com'] = np.array([0.067148, -0.10732, 0.024479])        self.jlc.lnks[4]['mass'] = 0.52387        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link4.stl")        self.jlc.lnks[4]['rgba'] = [.3, .5, .3, 1.0]        self.jlc.lnks[5]['name'] = "link5"        self.jlc.lnks[5]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[5]['com'] = np.array([-0.00023397, 0.036705, -0.080064])        self.jlc.lnks[5]['mass'] = 0.18554        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link5.stl")        self.jlc.lnks[6]['name'] = "link6"        self.jlc.lnks[6]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[6]['com'] = np.array([0.058911, 0.028469, 0.0068428])        self.jlc.lnks[6]['mass'] = 0.31344        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link6.stl")        self.jlc.lnks[6]['rgba'] = [.5, .5, .5, 1.0]        self.jlc.lnks[7]['name'] = "link7"        self.jlc.lnks[7]['pos_in_loc_tcp'] = np.zeros(3)        self.jlc.lnks[7]['com'] = np.array([-1.5846e-5, -0.0046377, -0.012705])        self.jlc.lnks[7]['mass'] = 0.31468        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "link7.stl")        # reinitialization        # self.jlc.setinitvalues(np.array([-math.pi/2, math.pi/3, math.pi/6, 0, 0, 0, 0]))        # self.jlc.setinitvalues(np.array([-math.pi/2, 0, math.pi/3, math.pi/10, 0, 0, 0]))        self.jlc.finalize()        # collision detection        if enable_cc:            self.enable_cc()    def enable_cc(self):        super().enable_cc()        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6, 7])        activelist = [self.jlc.lnks[0],                      self.jlc.lnks[1],                      self.jlc.lnks[2],                      self.jlc.lnks[3],                      self.jlc.lnks[4],                      self.jlc.lnks[5],                      self.jlc.lnks[6],                      self.jlc.lnks[7]]        self.cc.set_active_cdlnks(activelist)        fromlist = [self.jlc.lnks[0],                    self.jlc.lnks[1],                    self.jlc.lnks[2]]        intolist = [self.jlc.lnks[5],                    self.jlc.lnks[6],                    self.jlc.lnks[7]]        self.cc.set_cdpair(fromlist, intolist)if __name__ == '__main__':    import time    import visualization.panda.world as wd    import modeling.geometric_model as gm    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5])    gm.gen_frame().attach_to(base)    manipulator_instance = XArm7(enable_cc=True)    # manipulator_instance.fk([0,0,0,-math.pi/10,0,0,0])    manipulator_meshmodel = manipulator_instance.gen_meshmodel()    manipulator_meshmodel.attach_to(base)    manipulator_instance.gen_stickmodel().attach_to(base)    manipulator_instance.show_cdprimit()    tic = time.time()    print(manipulator_instance.is_collided())    toc = time.time()    print(toc - tic)    manipulator_instance2 = manipulator_instance.copy()    # manipulator_instance2.disable_cc()    manipulator_instance2.fix_to(pos=np.array([0, .3, .3]), rotmat=np.eye(3))    manipulator_instance2.gen_mesh_model().attach_to(base)    manipulator_instance2.show_cdprimit()    base.run()