import copy
import numpy as np
import modeling.model_collection as mc
import robot_sim.kinematics.jlchain as jl
import robot_sim.kinematics.collision_checker as cc
import modeling.geometric_model as gm


class EEInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='aabb', name='end_effector'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cdmesh_type = cdmesh_type  # aabb, convexhull, or triangles
        # joints
        # - coupling - No coupling by default
        self.coupling = jl.JLChain(pos=self.pos, rotmat=self.rotmat, home_conf=np.zeros(0), name='coupling')
        self.coupling.joints[1]['pos_in_loc_tcp'] = np.array([0, 0, .0])
        self.coupling.lnks[0]['name'] = 'coupling_lnk0'
        # toggle on the following part to assign an explicit mesh model to a coupling
        # self.coupling.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "xxx.stl")
        # self.coupling.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.coupling.reinitialize()
        # action center, acting point of the tool
        self.action_center_pos = np.zeros(3)
        self.action_center_rotmat = np.eye(3)
        # collision detection
        self.cc = None
        # cd mesh collection for precise collision checking
        self.cdmesh_collection = mc.ModelCollection()
        # object grasped/held/attached to end_type-effector; oiee = object in end_type-effector
        self.oiee_infos = []

    def update_oiee(self):
        """
        oih = object in hand
        :return:
        author: weiwei
        date: 20230807
        """
        for obj_info in self.oiee_infos:
            gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def hold(self, objcm, **kwargs):
        """
        the objcm is added as a part of the robot_s to the cd checker
        **kwargs is for polyphorism purpose
        :param jawwidth:
        :param objcm:
        :return:
        author: weiwei
        date: 20230811
        """
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.agv.lnks[3],
                    self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return_val =  self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        return return_val

    def is_mesh_collided(self, objcm_list=[], toggle_debug=False):
        for i, cdelement in enumerate(self.all_cdelements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
            iscollided, collided_points = self.cdmesh_collection.cm_list[i].is_mcdwith(objcm_list, True)
            if iscollided:
                if toggle_debug:
                    print(self.cdmesh_collection.cm_list[i].get_homomat())
                    self.cdmesh_collection.cm_list[i].show_cdmesh()
                    for objcm in objcm_list:
                        objcm.show_cdmesh()
                    for point in collided_points:
                        import modeling.geometric_model as gm
                        gm.gen_sphere(point, radius=.001).attach_to(base)
                    print("collided")
                return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        for i, cdelement in enumerate(self.cc.all_cd_elements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ee_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='ee_meshmodel'):
        raise NotImplementedError

    def _toggle_tcpcs(self, parent):
        action_center_gl_pos = self.rotmat.dot(self.action_center_pos) + self.pos
        action_center_gl_rotmat = self.rotmat.dot(self.action_center_rotmat)
        gm.gen_dashed_stick(spos=self.pos,
                            epos=action_center_gl_pos,
                            radius=.0062,
                            rgba=[.5, 0, 1, 1],
                            type="round").attach_to(parent)
        gm.gen_myc_frame(pos=action_center_gl_pos, rotmat=action_center_gl_rotmat).attach_to(parent)

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and pdndp
        :return:
        """
        for cdelement in self.cc.all_cd_elements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.cd_trav.addCollider(child, self_copy.cc.cd_handler)
        return self_copy