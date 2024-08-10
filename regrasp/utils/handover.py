""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240218osaka

"""
import copy
from typing import List
from tqdm import tqdm
import numpy as np
import modeling.collision_model as cm
import basis.robot_math as rm
from robot_sim.end_effectors.gripper.gripper_interface import GripperInterface


class Handover(object):
    def __init__(self,
                 objcm: cm.CollisionModel,
                 hnd_rgt: GripperInterface,
                 hnd_lft: GripperInterface,
                 grasp_info_list_rgt: List[List[np.ndarray]],
                 grasp_info_list_lft: List[List[np.ndarray]],
                 retract_distance: float = 0.1, ):
        """
        Initialize the handover planner
        """
        assert isinstance(objcm, cm.CollisionModel), "Invalid objcm! It should be a CollisionModel!"
        assert isinstance(hnd_rgt, GripperInterface), "Invalid hnd_right! It should be a GripperInterface!"
        assert isinstance(hnd_lft, GripperInterface), "Invalid hnd_left! It should be a GripperInterface!"
        assert isinstance(retract_distance, (int, float)), "Invalid retract_distance! It should be a number!"
        self._objcm = objcm
        self._hnd_rgt = hnd_rgt.copy()
        self._hnd_lft = hnd_lft.copy()
        self._grasp_info_list_rgt = copy.deepcopy(grasp_info_list_rgt)
        self._grasp_info_list_lft = copy.deepcopy(grasp_info_list_lft)
        self._retract_distance = retract_distance

        # floating object pose list
        self.floating_obj_pose_list = []  # a list of floating object poses (position and rotation 4x4 matrix)
        self.grasp_pair_list_obj_pose = []  # grasp pair list at the identity pose
        self.grasp_pair_list_obj_floating_pose = []  # grasp pair list at the floating pose

        self.ikfid_fpsnestedglist_rgt = {}  # fid - feasible id
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_lft = {}

    def set_objcm(self, objcm: cm.CollisionModel):
        """
        Set the collision model of the object
        :param objcm: the collision model of the object
        :return: None
        """
        assert isinstance(objcm, cm.CollisionModel), "Invalid objcm! It should be a CollisionModel!"
        self._objcm = objcm

    def set_hnd_right(self, hnd_right: GripperInterface):
        """
        Set the right hand
        :param hnd_right: the right hand
        :return: None
        """
        assert isinstance(hnd_right, GripperInterface), "Invalid hnd_right! It should be a GripperInterface!"
        self._hnd_rgt = hnd_right.copy()

    def set_hnd_left(self, hnd_left: GripperInterface):
        """
        Set the left hand
        :param hnd_left: the left hand
        :return: None
        """
        assert isinstance(hnd_left, GripperInterface), "Invalid hnd_left! It should be a GripperInterface!"
        self._hnd_lft = hnd_left.copy()

    def set_retract_distance(self, retract_distance: float):
        """
        Set the retract distance
        :param retract_distance: the retract distance
        :return: None
        """
        assert isinstance(retract_distance, (int, float)), "Invalid retract_distance! It should be a number!"
        self._retract_distance = retract_distance

    def gen_handover_grasp_psoe(self,
                                handover_pose: np.ndarray,
                                toggle_debug: bool = False, ):
        """
        generate a handover grasp pose using the given object handover pose and the handover direction
        :param handover_pose: a handover pose of the object (position and rotation 4x4 matrix)
        :param toggle_debug: whether to toggle the debug mode
        """
        obj_pos = handover_pose[:3, 3]
        obj_rotmat = handover_pose[:3, :3]
        # get the handover direction

    def gen_grasp_list_floating_obj_psoe(self,
                                          floating_obj_pose_list: List[np.ndarray], ):
        """
        Generate a list of grasps for a floating object
        :param floating_obj_pose: the pose of the floating object (position and rotation 4x4 matrix)
        """
        self.fpsnestedglist_rgt = {}
        self.fpsnestedglist_lft = {}
        for posid, floating_pose in tqdm(enumerate(floating_obj_pose_list),
                                         desc="generating nested glist at the floating poses...",
                                         total=len(floating_obj_pose_list)):  # for each floating pose
            glist = []
            for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in self._grasp_info_list_rgt:
                floating_rotmat = floating_pose[:3, :3]
                jaw_center_homomat = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
                homomat = np.dot(floating_pose, jaw_center_homomat)
                approach_direction = jaw_center_rotmat[:, 2]
                approach_direction = np.dot(floating_rotmat, approach_direction)
                glist.append([jaw_width, homomat, approach_direction])

                # robot ee pose
                init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
                # robot ee rot
                init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)

            self.fpsnestedglist_rgt[posid] = glist
            glist = []
            for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in self._grasp_info_list_lft:
                tippos = rm.homotransformpoint(icomat4, fc)
                homomat = np.dot(icomat4, homomat)
                approach_direction = np.dot(icomat4[:3, :3], approach_direction)
                glist.append([jawwidth, tippos, homomat, approach_direction])
            self.fpsnestedglist_lft[posid] = glist


def plan_handover(self, objcm, hnd_s, hnd_g, angle_between_contact_normals, openning_direction, max_samples,
                  min_dist_between_sampled_contact_points, contact_offset):
    """
    Plan the handover
    :param objcm: the collision model of the object
    :param hnd_s: the source hand
    :param hnd_g: the goal hand
    :param angle_between_contact_normals: the angle between the contact normals of the two hands
    :param openning_direction: the openning direction of the hands
    :param max_samples: the maximum number of samples to plan the handover
    :param min_dist_between_sampled_contact_points: the minimum distance between the sampled contact points
    :param contact_offset: the offset of the contact points
    :return: a list of handover info, each handover info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    """
    raise NotImplementedError


def write_pickle_file(self, obj_name, handover_info_list, root, file_name):
    """
    Write the handover info list to a pickle file
    :param obj_name: the name of the object
    :param handover_info_list: a list of handover info, each handover info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    :param root: the root directory to save the pickle file
    :param file_name: the file name of the pickle file
    :return: None
    """
    raise NotImplementedError


def read_pickle_file(self, path):
    pass


if __name__ == '__main__':
    pass
