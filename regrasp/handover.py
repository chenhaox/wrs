""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240219osaka

"""
from typing import List
import itertools
import pickle
import os
import numpy as np
from tqdm import tqdm
import basis.robot_math as rm
import modeling.collision_model as cm
import regrasp.utils.file_sys as fs
from regrasp.utils.visualize import visualize_grasp_pairs, visualize_handover
from robot_sim.robots.robot_interface import RobotInterface
from robot_sim.end_effectors.gripper.gripper_interface import GripperInterface


def gen_icorotmats_flat(icolevel=1, angles=np.radians([0, 45, 90, 135, 180, 225, 270, 315])):
    """
    generate rotmat3 using icospheres and rotation angle at each origin-vertex vector of the icosphere
    this function generates a flat list of icorotmat instead of a nested one

    :param icolevel, the default value 1 = 42vertices
    :param angles, 8 directions by default
    :return: [rotmat3, ...] a flat list

    author: weiwei
    date: 20191212, osaka
    """

    returnlist = []
    icos = rm.trm.creation.icosphere(icolevel)
    for vert in icos.vertices:
        initvec = np.array([0, 0, 1])
        z = -vert
        if abs(np.dot(z, initvec)) > 9.9:
            initvec = np.array([0, 1, 0])
        x = rm.unit_vector(np.cross(z, initvec))
        y = rm.unit_vector(np.cross(z, x))
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        for angle in angles:
            returnlist.append(np.dot(rm.rotmat_from_axangle(z, angle), temprotmat))
    return returnlist


def gen_icohomomats_flat(icolevel=1,
                         posvec=np.array([0, 0, 0]),
                         angles=np.radians([0, 45, 90, 135, 180, 225, 270, 315])) -> List[np.ndarray]:
    """
    generate homomat(4x4) using icospheres and rotation angle at each origin-vertex vector of the icosphere

    :param icolevel, the default value 1 = 42vertices
    :param pos, zero position by default
    :param angles, 8 directions by default
    :return: [rotmat3, ...] size of the inner list is size of the angles

    author: weiwei
    date: 20191015, osaka
    """

    returnlist = []
    rotmat3list = gen_icorotmats_flat(icolevel=icolevel, angles=angles)
    for npmat3 in rotmat3list:
        npmat4 = np.eye(4)
        npmat4[:3, :3] = npmat3
        npmat4[:3, 3] = posvec
        returnlist.append(npmat4)
    return returnlist


class Handover(object):
    """

    author: hao chen, ruishuang liu, refactored by weiwei
    date: 20191122
    """

    def __init__(self,
                 obj_cm: cm.CollisionModel,
                 rgt_hnd: GripperInterface,
                 lft_hnd: GripperInterface,
                 grasp_info_list_rgt: List[List[np.ndarray]],
                 grasp_info_list_lft: List[List[np.ndarray]],
                 rbt: RobotInterface,
                 save_dir: str or fs.Path = None,
                 ret_dis: float = 0.1, ):
        """

        :param obj_cm: the collision model of the object
        :param rgt_hnd: the right hand
        :param lft_hnd: the left hand
        :param grasp_info_list_rgt: the grasp list of the right hand
        :param grasp_info_list_lft: the grasp list of the left hand
        :param rbt: the robot
        :param save_dir: the directory to save the data
        :param ret_dis: the retraction distance
        :return:

        author: hao, refactored by weiwei
        date: 20191206, 20200104osaka
        """
        assert isinstance(obj_cm, cm.CollisionModel), "Invalid objcm! It should be a CollisionModel!"
        assert isinstance(rgt_hnd, GripperInterface), "Invalid hnd_right! It should be a GripperInterface!"
        assert isinstance(lft_hnd, GripperInterface), "Invalid hnd_left! It should be a GripperInterface!"
        assert isinstance(ret_dis, (int, float)), "Invalid retract_distance! It should be a number!"
        self.obj_cm = obj_cm
        self.rbt = rbt
        assert hasattr(self.rbt, 'is_ikfast') and self.rbt.is_ikfast, "The robot should have ikfast!"
        self.ret_dis = ret_dis  # retraction distance
        self.rgt_hnd = rgt_hnd  # right hand
        self.lft_hnd = lft_hnd  # left hand
        self.grasp_list_rgt = grasp_info_list_rgt  # grasp list right
        self.grasp_list_lft = grasp_info_list_lft  # grasp list left

        self.save_dir = fs.Path(save_dir) if save_dir is not None else fs.workdir_data

        self.grasp = [self.grasp_list_rgt, self.grasp_list_lft]
        self.hnds = [self.rgt_hnd, self.lft_hnd]
        # paramters
        self.fp_list = []  # floating pose list
        self.identity_grasp_pairs = []  # grasp pair list at the identity pose
        self.grasp_list_fp_rgt = {}  # grasp_list_fp_rgt[fpid] = [g0, g1, ...], fpsnestedglist means glist at each floating pose
        self.grasp_list_fp_lft = {}  # grasp_list_fp_lft[fpid] = [g0, g1, ...]
        self.ik_fp_fid_gid_rgt = {}  # fid - feasible id
        self.ik_fp_fid_gid_lft = {}
        self.ik_jnts_fp_gid_rgt = {}
        self.ik_jnts_fp_gid_lft = {}

    def _gen_fp_list(self, pos_list: list, rotmat=None, toggle_debug=False) -> list:
        assert isinstance(pos_list, list), "pos_list must a set of positions indicate the handover positions"
        fp_list = []
        for posvec in pos_list:
            if rotmat is None:
                fp_list += gen_icohomomats_flat(posvec=posvec, angles=np.radians([0, 45, 90, 135, 180, 225, 270]))
            elif isinstance(rotmat, np.ndarray):
                if rotmat.shape == (3, 3):
                    fp_list += [rm.homomat_from_posrot(posvec, rotmat)]
                elif len(rotmat) == 1:
                    fp_list += gen_icohomomats_flat(posvec=posvec, angles=rotmat)
            else:
                raise ValueError("rotmat should be a numpy array or None")
        if toggle_debug:
            for mat in fp_list:
                objtmp = self.obj_cm.copy()
                objtmp.set_homomat(mat)
                objtmp.attach_to(base)
            base.run()
        return fp_list

    def _gen_grasp_pair_identity(self, toggle_debug: bool = False):
        """
        fill up self.identitygplist

        :return:

        author: weiwei
        date: 20191212
        """

        rgt_hnd = self.rgt_hnd
        lft_hnd = self.lft_hnd
        pairidlist = list(itertools.product(range(len(self.grasp_list_rgt)), range(len(self.grasp_list_lft))))
        if toggle_debug:
            debug_gppair = []
        for i in tqdm(range(len(pairidlist)), desc="generating identity gplist...", total=len(pairidlist)):
            # Check whether the hands collide with each or not
            ir, il = pairidlist[i]
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = self.grasp_list_rgt[ir]
            rgt_hnd.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = self.grasp_list_lft[il]
            lft_hnd.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            is_hnd_collided = lft_hnd.is_collided(obstacle_list=rgt_hnd.gen_meshmodel().cm_list)
            # if toggle_debug:
            #     rgt_hnd.gen_meshmodel().attach_to(base)
            #     rgt_hnd.show_cdprimit()
            #     lft_hnd.gen_meshmodel().attach_to(base)
            #     lft_hnd.show_cdprimit()
            #     self.obj_cm.attach_to(base)
            #     base.run()
            if not is_hnd_collided:
                self.identity_grasp_pairs.append(pairidlist[i])
                if toggle_debug:
                    debug_gppair.append([self.grasp_list_rgt[ir], self.grasp_list_lft[il]])
        print("Number of feasible pairs: ", len(self.identity_grasp_pairs))
        if toggle_debug:
            self.obj_cm.attach_to(base)
            visualize_grasp_pairs(debug_gppair, self.rgt_hnd, self.lft_hnd, base)
            base.run()

    def _gen_grasp_list_fp(self, fp_list):
        """
        generate the grasp list for the floating poses

        :return:

        author: hao chen, revised by weiwei
        date: 20191122
        """

        self.grasp_list_fp_rgt = {}
        self.grasp_list_fp_lft = {}
        for posid, icomat4 in tqdm(enumerate(fp_list), desc="generating nested glist at the floating poses...",
                                   total=len(fp_list)):  # for each floating pose
            glist = []
            for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in self.grasp_list_rgt:
                tippos = rm.homomat_transform_points(icomat4, jaw_center_pos)
                homomat = np.dot(icomat4, rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat))
                approach_direction = np.dot(icomat4[:3, :3], jaw_center_rotmat[:, 2])
                glist.append([jaw_width, tippos, homomat, approach_direction])
            self.grasp_list_fp_rgt[posid] = glist
            glist = []
            for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in self.grasp_list_lft:
                tippos = rm.homomat_transform_points(icomat4, jaw_center_pos)
                homomat = np.dot(icomat4, rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat))
                approach_direction = np.dot(icomat4[:3, :3], jaw_center_rotmat[:, 2])
                glist.append([jaw_width, tippos, homomat, approach_direction])
            self.grasp_list_fp_lft[posid] = glist

    def genhvgpsgl(self, pos_list: list, rotmat=None, toggle_debug=False):
        """
        generate the handover grasps using the given position and orientation
        sgl means a single position
        rotmat could either be a single one or multiple (0,90,180,270, default)

        :param pos_list: A list of positions indicating the handover positions
        :param rotmat
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identity_grasp_pairs = []
        self.fp_list = self._gen_fp_list(pos_list, rotmat, toggle_debug)
        self._gen_grasp_pair_identity()
        self._gen_grasp_list_fp(self.fp_list)
        self._check_ik()

        fs.dump_pickle([self.obj_cm.name,
                        self.ret_dis,
                        self.grasp_list_rgt,
                        self.grasp_list_lft,
                        self.fp_list,
                        self.identity_grasp_pairs,
                        self.grasp_pairs_fp,
                        self.grasp_list_fp_rgt,
                        self.grasp_list_fp_lft,
                        self.ik_fp_fid_gid_rgt,
                        self.ik_fp_fid_gid_lft,
                        self.ik_jnts_fp_gid_rgt,
                        self.ik_jnts_fp_gid_lft],
                       self.save_dir.joinpath("data_handover_hndovrinfo.pickle"), reminder=False)

    def genhvgplist(self, hvgplist):
        """
        generate the handover grasps using the given list of homomat

        :param hvgplist, [homomat0, homomat1, ...]
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identity_grasp_pairs = []
        self.fp_list = hvgplist
        self._gen_grasp_pair_identity()
        self._gen_grasp_list_fp(self.fp_list)
        self._check_ik()

        fs.dump_pickle([self.obj_cm.name,
                        self.ret_dis,
                        self.grasp_list_rgt,
                        self.grasp_list_lft,
                        self.fp_list,
                        self.identity_grasp_pairs,
                        self.grasp_pairs_fp,
                        self.grasp_list_fp_rgt,
                        self.grasp_list_fp_lft,
                        self.ik_fp_fid_gid_rgt,
                        self.ik_fp_fid_gid_lft,
                        self.ik_jnts_fp_gid_rgt,
                        self.ik_jnts_fp_gid_lft],
                       self.save_dir.joinpath("data_handover_hndovrinfo.pickle"), reminder=False)

    def gethandover(self):
        """
        io interface to load the previously planned data

        :return:

        author: hao, refactored by weiwei
        date: 20191206, 20191212
        """
        [self.fp_list, self.identity_grasp_pairs,
         self.grasp_list_fp_rgt, self.grasp_list_fp_lft,
         self.ik_fp_fid_gid_rgt, self.ik_fp_fid_gid_lft,
         self.ik_jnts_fp_gid_rgt, self.ik_jnts_fp_gid_lft] = fs.load_pickle(
            self.save_dir.joinpath("data_handover_hndovrinfo.pickle"))

        return self.grasp_list_rgt, self.grasp_list_lft, self.fp_list, \
               self.identity_grasp_pairs, self.grasp_list_fp_rgt, self.grasp_list_fp_lft, \
               self.ik_fp_fid_gid_rgt, self.ik_fp_fid_gid_lft, \
               self.ik_jnts_fp_gid_rgt, self.ik_jnts_fp_gid_lft

    def _check_ik(self, toggle_debug: bool = False):
        # Check the IK of both hand in the handover pose
        ### right hand
        self.ik_fp_fid_gid_rgt = {}  # ik floating pose, feasible id, grasp id
        self.ik_jnts_fp_gid_rgt = {}  # ik joint values, floating pose, grasp id
        self.manipulability_fp_gid_rgt = {}
        self.ik_fp_fid_gid_lft = {}
        self.ik_jnts_fp_gid_lft = {}
        self.manipulability_fp_gid_lft = {}
        for posid in tqdm(self.grasp_list_fp_rgt.keys()):
            armname = 'rgt_arm'
            fp_grasp_list_thispose = self.grasp_list_fp_rgt[posid]  # load grasps at each floating pose
            for i, [_, tippos, homomat, approach_direction] in enumerate(fp_grasp_list_thispose):  # for each grasp
                fp_grasp_center = tippos
                fp_grasp_center_rotmat = homomat[:3, :3]
                approach_direction = -approach_direction
                # minusworldy = np.array([0, -1, 0])
                # if rm.degree_betweenvector(approach_direction, minusworldy) < 90:
                msc_list = self.rbt.ik(component_name=armname,
                                       tgt_pos=fp_grasp_center,
                                       tgt_rotmat=fp_grasp_center_rotmat,
                                       all_sol=True)
                if msc_list is not None:
                    msc_msc_handa_ik_sols = []
                    mpa_mpa_approach = []
                    for msc in msc_list:
                        self.rbt.fk(component_name=armname, jnt_values=msc)
                        is_collided = self.rbt.is_collided()
                        if is_collided:
                            continue
                        manipulability = self.rbt.manipulability(component_name=armname)
                        fp_grasp_center_apporach = fp_grasp_center + approach_direction * self.ret_dis
                        msc_handa = self.rbt.ik(component_name=armname,
                                                tgt_pos=fp_grasp_center_apporach,
                                                tgt_rotmat=fp_grasp_center_rotmat,
                                                seed_jnt_values=msc, )
                        if msc_handa is not None:
                            self.rbt.fk(component_name=armname, jnt_values=msc_handa)
                            is_collided = self.rbt.is_collided()
                            if is_collided:
                                continue
                            manipulability_handa = self.rbt.manipulability(component_name=armname)
                            if posid not in self.ik_fp_fid_gid_rgt:
                                self.ik_fp_fid_gid_rgt[posid] = []
                            if i not in self.ik_fp_fid_gid_rgt[posid]:
                                self.ik_fp_fid_gid_rgt[posid].append(i)
                            msc_msc_handa_ik_sols.append([msc, msc_handa])
                            mpa_mpa_approach.append([manipulability,  manipulability_handa])
                    if posid not in self.ik_jnts_fp_gid_rgt:
                        self.ik_jnts_fp_gid_rgt[posid] = {}
                        self.manipulability_fp_gid_rgt[posid] = {}
                    self.ik_jnts_fp_gid_rgt[posid][i] = msc_msc_handa_ik_sols
                    self.manipulability_fp_gid_rgt[posid][i] = mpa_mpa_approach

        ### left hand
        for posid in tqdm(self.grasp_list_fp_lft.keys()):
            armname = 'lft_arm'
            fp_grasp_list_thispose = self.grasp_list_fp_lft[posid]
            for i, [_, tippos, homomat, approach_direction] in enumerate(fp_grasp_list_thispose):
                fp_grasp_center = tippos
                fp_grasp_center_rotmat = homomat[:3, :3]
                approach_direction = -approach_direction
                # plusworldy = np.array([0, 1, 0])
                # if rm.degree_betweenvector(approach_direction, plusworldy) < 90:
                msc_list = self.rbt.ik(component_name=armname,
                                       tgt_pos=fp_grasp_center,
                                       tgt_rotmat=fp_grasp_center_rotmat,
                                       all_sol=True)
                if msc_list is not None:
                    msc_msc_handa_ik_sols = []
                    mpa_mpa_approach = []
                    for msc in msc_list:
                        self.rbt.fk(component_name=armname, jnt_values=msc)
                        is_collided = self.rbt.is_collided()
                        if is_collided:
                            continue
                        manipulability = self.rbt.manipulability(component_name=armname)
                        fp_grasp_center_apporach = fp_grasp_center + approach_direction * self.ret_dis
                        msc_handa = self.rbt.ik(component_name=armname,
                                                tgt_pos=fp_grasp_center_apporach,
                                                tgt_rotmat=fp_grasp_center_rotmat,
                                                seed_jnt_values=msc, )
                        if msc_handa is not None:
                            self.rbt.fk(component_name=armname, jnt_values=msc_handa)
                            is_collided = self.rbt.is_collided()
                            if is_collided:
                                continue
                            manipulability_handa = self.rbt.manipulability(component_name=armname)
                            if posid not in self.ik_fp_fid_gid_lft:
                                self.ik_fp_fid_gid_lft[posid] = []
                            if i not in self.ik_fp_fid_gid_lft[posid]:
                                self.ik_fp_fid_gid_lft[posid].append(i)
                            msc_msc_handa_ik_sols.append([msc, msc_handa])
                            mpa_mpa_approach.append([manipulability,  manipulability_handa])
                    if posid not in self.ik_jnts_fp_gid_lft:
                        self.ik_jnts_fp_gid_lft[posid] = {}
                        self.manipulability_fp_gid_lft[posid] = {}
                    self.ik_jnts_fp_gid_lft[posid][i] = msc_msc_handa_ik_sols
                    self.manipulability_fp_gid_lft[posid][i] = mpa_mpa_approach

        self.grasp_pairs_fp = {}
        for posid in range(len(self.fp_list)):
            # zz = copy.deepcopy(self.hndovermaster.objcm)
            # zz.reparentTo(base.render)
            # zz.sethomomat(self.gridsfloatingposemat4np[posid])
            # print(self.gridsfloatingposemat4np[posid])
            if posid in self.ik_fp_fid_gid_rgt.keys() and posid in self.ik_fp_fid_gid_lft.keys():
                pass
            else:
                continue
            for i0, i1 in self.identity_grasp_pairs:
                if i0 in self.ik_fp_fid_gid_rgt[posid] and i1 in self.ik_fp_fid_gid_lft[posid]:
                    if posid not in self.grasp_pairs_fp:
                        self.grasp_pairs_fp[posid] = []
                    self.grasp_pairs_fp[posid].append([i0, i1])
                else:
                    continue

        if toggle_debug:
            print("Number of feasible grasps for right hand: ", len(self.ik_fp_fid_gid_rgt))
            print("Number of feasible grasps for left hand: ", len(self.ik_fp_fid_gid_lft))
            visualize_handover(self.identity_grasp_pairs,
                               self.fp_list,
                               self.obj_cm,
                               self.rbt,
                               self.ik_jnts_fp_gid_rgt,
                               self.ik_jnts_fp_gid_lft,
                               base)
            base.run()


if __name__ == "__main__":
    import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rqh
    import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
    import grasping.planning.antipodal as gpa
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])

    obj_name = 'lshape.stl'
    if not isinstance(obj_name, str) and not obj_name.endswith('.stl') and not obj_name.endswith('.dae'):
        raise Exception(
            f"{obj_name} is not a valid object name! Please check the name of the object! It should end with .stl or .dae!")
    # examine if the obj_name is in the model folder
    obj_path = fs.workdir_model.joinpath(obj_name)
    if not obj_path.exists():
        raise Exception(f"{obj_path} does not exist! Please check the name of the object!")
    obj_cm = cm.CollisionModel(str(obj_path), name=obj_name)

    gripper_s = rqh.RobotiqHE()
    rgt_hnd = gripper_s.copy()
    lft_hnd = gripper_s.copy()

    rbt = ur3ed.UR3EDual(enable_cc=True)

    obj_name_no_ext = obj_name.split('.')[0]
    grasp_info_list = gpa.load_pickle_file(obj_name_no_ext, str(fs.workdir_data), file_name='robotiqhe_grasps.pickle')

    hmstr = Handover(obj_cm=obj_cm,
                     rgt_hnd=rgt_hnd,
                     lft_hnd=lft_hnd,
                     grasp_info_list_rgt=grasp_info_list,
                     grasp_info_list_lft=grasp_info_list,
                     rbt=rbt)

    hmstr.genhvgpsgl(pos_list=[np.array([0.700, 0, 1.300])], rotmat=np.radians([0, ]), toggle_debug=False)
