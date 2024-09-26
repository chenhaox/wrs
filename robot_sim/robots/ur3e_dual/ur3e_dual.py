import os
import math
import time

import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur3e.ur3e as ur
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtq
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri

try:
    from ur_ikfast import URKinematics

    print("IKFast module loaded successfully")
    IS_IKFAST_LOADED = True
except:
    IS_IKFAST_LOADED = False
    print("IKFast module not loaded")


class UR3EDual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3edual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='lft_body_jl')
        self.lft_body.jnts[1]['loc_pos'] = np.array([.365, .345, 1.33])  # left from robot_s view
        self.lft_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot_s view
        self.lft_body.lnks[0]['name'] = "ur3e_dual_base"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3e_dual_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_body.lnks[0]['rgba'] = [.55, .55, .55, 1.0]
        self.lft_body.reinitialize()
        lft_arm_homeconf = np.zeros(6)
        lft_arm_homeconf[0] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[2] = math.pi / 2.0
        lft_arm_homeconf[3] = math.pi
        lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = ur.UR3E(pos=self.lft_body.jnts[-1]['gl_posq'],
                               rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                               homeconf=lft_arm_homeconf,
                               enable_cc=False)
        self.lft_hnd = rtq.RobotiqHE(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                     rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                     coupling_offset_pos=np.array([0.0, 0.0, 0.0331]),
                                     enable_cc=False)
        # rigth side
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_body_jl')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([.365, -.345, 1.33])  # right from robot_s view
        self.rgt_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot_s view
        self.rgt_body.lnks[0]['name'] = "ur3e_dual_base"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['mesh_file'] = None
        self.rgt_body.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.rgt_body.reinitialize()
        rgt_arm_homeconf = np.zeros(6)
        rgt_arm_homeconf[0] = math.pi * 2.0 / 3.0
        rgt_arm_homeconf[1] = -math.pi / 3.0
        rgt_arm_homeconf[2] = -math.pi / 2.0
        rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = ur.UR3E(pos=self.rgt_body.jnts[-1]['gl_posq'],
                               rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                               homeconf=rgt_arm_homeconf,
                               enable_cc=False)
        self.rgt_hnd = rtq.RobotiqHE(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                                     rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                     coupling_offset_pos=np.array([0.0, 0.0, 0.0331]),
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jnt_id = -1
        self.lft_arm.tcp_loc_pos = self.lft_hnd.jaw_center_pos
        self.lft_arm.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat
        # rgt
        self.rgt_arm.tcp_jnt_id = -1
        self.rgt_arm.tcp_loc_pos = self.lft_hnd.jaw_center_pos
        self.rgt_arm.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm  # specify which hand is a gripper installed to
        self.manipulator_dict['lft_hnd'] = self.lft_arm  # specify which hand is a gripper installed to
        self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        self.hnd_dict['lft_hnd'] = self.lft_hnd
        self.hnd_dict['rgt_arm'] = self.rgt_hnd
        self.hnd_dict['lft_arm'] = self.lft_hnd

        self.hnd_raw_dict['rgt_arm'] = self.hnd_raw_dict['rgt_hnd'] = rtq.RobotiqHE(
            coupling_offset_pos=np.array([0.0, 0.0, 0.0331]),
            enable_cc=True)
        self.hnd_raw_dict['lft_arm'] = self.hnd_raw_dict['lft_hnd'] = rtq.RobotiqHE(
            coupling_offset_pos=np.array([0.0, 0.0, 0.0331]),
            enable_cc=True)

        if IS_IKFAST_LOADED:
            self.b2w_homomat_dict = {}
            self.tcp2ee_homomat_dict = {}
            self._ur3e_arm = URKinematics('ur3e')
            self.b2w_homomat_dict['rgt_arm'] = self.b2w_homomat_dict['rgt_hnd'] = np.linalg.inv(
                rm.homomat_from_posrot(self.rgt_body.jnts[-1]['gl_posq'],
                                       self.rgt_body.jnts[-1][
                                           'gl_rotmatq'], ))
            self.b2w_homomat_dict['lft_arm'] = self.b2w_homomat_dict['lft_hnd'] = np.linalg.inv(rm.homomat_from_posrot(self.lft_body.jnts[-1]['gl_posq'],
                                                                                    self.lft_body.jnts[-1][
                                                                                        'gl_rotmatq'], ))
            self.tcp2ee_homomat_dict['rgt_arm'] = self.tcp2ee_homomat_dict['rgt_hnd'] = np.linalg.inv(
                rm.homomat_from_posrot(self.manipulator_dict['rgt_arm'].tcp_loc_pos,
                                       self.manipulator_dict['rgt_arm'].tcp_loc_rotmat))
            self.tcp2ee_homomat_dict['lft_arm'] = self.tcp2ee_homomat_dict['lft_hnd'] = np.linalg.inv(
                rm.homomat_from_posrot(self.manipulator_dict['lft_arm'].tcp_loc_pos,
                                       self.manipulator_dict['lft_arm'].tcp_loc_rotmat))
            self.is_ikfast = True
        else:
            self._ur3e_arm = None
            self.b2w_homomat_dict = None
            self.tcp2ee_homomat_dict = None
            self.is_ikfast = False

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.54, 0.0, 0.39),
                                              x=.54 + radius, y=.6 + radius, z=.39 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.06, 0.0, 0.9),
                                              x=.06 + radius, y=.375 + radius, z=.9 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.18, 0.0, 1.77),
                                              x=.18 + radius, y=.21 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0.2425, 0.345, 1.33),
                                              x=.1225 + radius, y=.06 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.2425, -0.345, 1.33),
                                              x=.1225 + radius, y=.06 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_r0)
        collision_primitive_l1 = CollisionBox(Point3(0.21, 0.405, 1.07),
                                              x=.03 + radius, y=.06 + radius, z=.29 + radius)
        collision_node.addSolid(collision_primitive_l1)
        collision_primitive_r1 = CollisionBox(Point3(0.21, -0.405, 1.07),
                                              x=.03 + radius, y=.06 + radius, z=.29 + radius)
        collision_node.addSolid(collision_primitive_r1)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError
        self.cc.add_cdlnks(self.lft_body, [0])
        self.cc.add_cdlnks(self.lft_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.lft_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.lft_hnd.rgt, [1])
        self.cc.add_cdlnks(self.rgt_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.rgt_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.rgt_hnd.rgt, [1])
        # lnks used for cd with external stationary objects
        activelist = [self.lft_arm.lnks[2],
                      self.lft_arm.lnks[3],
                      self.lft_arm.lnks[4],
                      self.lft_arm.lnks[5],
                      self.lft_arm.lnks[6],
                      self.lft_hnd.lft.lnks[0],
                      self.lft_hnd.lft.lnks[1],
                      self.lft_hnd.rgt.lnks[1],
                      self.rgt_arm.lnks[2],
                      self.rgt_arm.lnks[3],
                      self.rgt_arm.lnks[4],
                      self.rgt_arm.lnks[5],
                      self.rgt_arm.lnks[6],
                      self.rgt_hnd.lft.lnks[0],
                      self.rgt_hnd.lft.lnks[1],
                      self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        fromlist = [self.lft_body.lnks[0],
                    self.lft_arm.lnks[1],
                    self.rgt_arm.lnks[1]]
        intolist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        # lnks used for arm-body collision detection -- extra
        fromlist = [self.lft_body.lnks[0]]  # body
        intolist = [self.lft_arm.lnks[2],
                    self.rgt_arm.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)
        # lnks used for in-arm collision detection
        fromlist = [self.lft_arm.lnks[2]]
        intolist = [self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.rgt_arm.lnks[2]]
        intolist = [self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        # arm-arm collision
        fromlist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1]]
        intolist = [self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'], rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat)
        self.rgt_body.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'], rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat)

    def get_hnd_on_manipulator(self, manipulator_name):
        if manipulator_name == 'rgt_arm' or manipulator_name == 'rgt_hnd':
            return self.rgt_hnd
        elif manipulator_name == 'lft_arm' or manipulator_name == 'lft_hnd':
            return self.lft_hnd
        else:
            raise ValueError("The given jlc does not have a hand!")

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: 1x6 or 1x12 nparray
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='rgt_arm'):
            # inline function for update objects in hand
            if component_name == 'rgt_arm' or component_name == 'rgt_hnd':
                oih_info_list = self.rgt_oih_infos
            elif component_name == 'lft_arm' or component_name == 'lft_hnd':
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.get_hnd_on_manipulator(component_name).fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine length
        if component_name in self.manipulator_dict.keys():
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        elif component_name == 'both_arm':
            if (jnt_values.size != 12):
                raise ValueError("A 1x12 npdarrays must be specified to move both arm!")
            status_lft = update_component('lft_arm', jnt_values[0:6])
            status_rgt = update_component('rgt_arm', jnt_values[6:12])
            return "succ" if status_lft == "succ" and status_rgt == "succ" else "out_of_rng"
        elif component_name == 'all':
            raise NotImplementedError
        else:
            raise ValueError("The given component name is not available!")

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end",
           all_sol=False,
           toggle_debug=False):
        if not IS_IKFAST_LOADED:
            return self.manipulator_dict[component_name].ik(tgt_pos,
                                                            tgt_rotmat,
                                                            seed_jnt_values=seed_jnt_values,
                                                            max_niter=max_niter,
                                                            tcp_jnt_id=tcp_jnt_id,
                                                            tcp_loc_pos=tcp_loc_pos,
                                                            tcp_loc_rotmat=tcp_loc_rotmat,
                                                            local_minima=local_minima,
                                                            toggle_debug=toggle_debug)
        else:

            seed_jnt_values = seed_jnt_values if seed_jnt_values is not None else np.zeros(6)
            w2tcp = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
            b2ee = self.b2w_homomat_dict[component_name].dot(w2tcp.dot(self.tcp2ee_homomat_dict[component_name]))
            return self._ur3e_arm.inverse(b2ee[:3, :],
                                          all_solutions=all_sol,
                                          q_guess=seed_jnt_values)

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            return super().rand_conf(component_name)
        elif component_name == 'both_arm':
            return np.hstack((super().rand_conf('lft_arm'), super().rand_conf('rgt_arm')))
        else:
            raise NotImplementedError

    def init_conf(self):
        """
        override robot_interface.init_conf
        :param component_name:
        :return:
        """
        for component_name in self.manipulator_dict.keys():
            self.fk(component_name, self.manipulator_dict[component_name].homeconf)

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ur3e_dual_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
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
                      name='ur3e_dual_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(mm_collection)
        self.lft_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(mm_collection)
        self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(mm_collection)
        self.rgt_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(mm_collection)
        self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(mm_collection)
        for obj_info in self.lft_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(mm_collection)
        for obj_info in self.rgt_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(mm_collection)
        return mm_collection

    def gen_env_meshmodel(self, toggle_jntscs=False, rgba=None, name='ur3e_dual_env_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(mm_collection)
        return mm_collection

    def gen_arm_meshmodel(self,
                          component_name,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None,
                          toggle_tcpcs=False,
                          toggle_jntscs=False,
                          rgba=None,
                          name='ur3e_dual_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        assert component_name in self.manipulator_dict, 'Error: component name is not recognized'
        arm = self.manipulator_dict[component_name]
        hnd = self.hnd_dict[component_name]
        if component_name == 'lft_arm':
            oih_infos = self.lft_oih_infos
        elif component_name == 'rgt_arm':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError('Error: component name is not recognized')
        arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                          tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat,
                          toggle_tcpcs=toggle_tcpcs,
                          toggle_jntscs=toggle_jntscs,
                          rgba=rgba).attach_to(mm_collection)
        hnd.gen_meshmodel(toggle_tcpcs=False,
                          toggle_jntscs=toggle_jntscs,
                          rgba=rgba).attach_to(mm_collection)
        for obj_info in oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    u3ed = UR3EDual()
    # u3ed.fk(.85)
    u3ed_meshmodel = u3ed.gen_meshmodel(toggle_tcpcs=True)
    u3ed_meshmodel.attach_to(base)
    ur3e = ur.UR3E()
    ur3e.fk(u3ed.get_jnt_values("lft_arm"))
    ur3e.gen_meshmodel().attach_to(base)

    tgt_pos, tgt_rotmat = np.array([1.04097236, 0.33367596, 1.04838618]), np.array(
        [[-8.36449319e-17, -8.66025404e-01, 5.00000000e-01],
         [-8.66025404e-01, -2.50000000e-01, -4.33012702e-01],
         [5.00000000e-01, -4.33012702e-01, -7.50000000e-01]])

    component_name = 'lft_arm'
    w2tcp = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
    ee2tcp = rm.homomat_from_posrot(u3ed.manipulator_dict[component_name].tcp_loc_pos,
                                    u3ed.manipulator_dict[component_name].tcp_loc_rotmat)
    # b2w_homomat = u3ed.b2w_homomat_dict[component_name]
    # w2ee = w2tcp.dot(np.linalg.inv(ee2tcp))
    # b2ee = b2w_homomat.dot(w2ee)
    # print("ur3e", ur3e.get_gl_tcp())
    # print("b2ee", b2ee[:3, 3], b2ee[:3, :3])
    # print(u3ed.get_gl_tcp("lft_arm"))
    pos, rot = np.array([1.04097236 - .01, 0.33367596 - .01, 1.04838618 - .01]), np.array(
        [[-8.36449319e-17, -8.66025404e-01, 5.00000000e-01],
         [-8.66025404e-01, -2.50000000e-01, -4.33012702e-01],
         [5.00000000e-01, -4.33012702e-01, -7.50000000e-01]])
    a = time.time()
    j = u3ed.ik("lft_arm", pos, rot, all_sol=True)
    b = time.time()
    print("time:", (b - a) * 1000)
    if j is not None:
        for j_sgl in j:
            print("jnt_values:", j_sgl)
            u3ed.fk("lft_arm", np.asarray(j_sgl))
            u3ed.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # u3ed_meshmodel.show_cdprimit()
    # u3ed.gen_stickmodel().attach_to(base)
    base.run()
