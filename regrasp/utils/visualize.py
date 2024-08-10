""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240218osaka

"""
from typing import List, Optional, Dict

import numpy as np
from direct.gui.DirectLabel import DirectLabel

import modeling.geometric_model as gm
import modeling.model_collection as mc
import visualization.panda.world as wd
from robot_sim.end_effectors.gripper.gripper_interface import GripperInterface
from robot_sim.robots.robot_interface import RobotInterface


def visualize_grasps(grasp_info_list: List[List[np.ndarray]],
                     gripper: GripperInterface,
                     base: Optional[wd.World] = None,
                     toggle_show_all: bool = False):
    """
    Visualize the grasps in the grasp_info_list
    :param grasp_info_list: a list of grasp info, each grasp info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    :param gripper: the gripper to visualize the grasps
    :param base: the world to visualize the grasps
    :param toggle_show_all: whether to show all the grasps at the same time
    :return: None
    """
    assert isinstance(grasp_info_list, list) and isinstance(grasp_info_list[0], list) and len(
        grasp_info_list[0]) == 5, "Invalid grasp_info_list!"
    assert isinstance(gripper, GripperInterface), "Invalid gripper!"
    if base is None:
        try:
            hasattr(base, 'run')
        except Exception as e:
            raise Exception(f"Invalid base: {e}")

    if toggle_show_all:
        for grasp_info in grasp_info_list:
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            gripper.gen_meshmodel().attach_to(base)
    else:
        counter = [0]
        vis_node: List[Optional[gm.GeometricModel]] = [None]

        def _show_gripper(task):
            """
            Show the gripper in the base
            """
            if base.inputmgr.keymap['space']:  # press space to switch to the next grasp
                base.inputmgr.keymap['space'] = False
                if vis_node[0] is not None:
                    vis_node[0].detach()  # detach the previous gripper
                counter[0] = (counter[0] + 1) % len(grasp_info_list)
                grasp_info = grasp_info_list[counter[0]]
                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
                gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
                vis_node[0] = gripper.gen_meshmodel()
                vis_node[0].attach_to(base)
            return task.again

        taskMgr.doMethodLater(0.1, _show_gripper, "show_gripper")  # show the gripper in the base


def visualize_grasps_fp(grasp_info_fid_gp: Dict[int, List[np.ndarray]],
                        fp_list: List[np.ndarray],
                        obj: gm.GeometricModel,
                        gripper: GripperInterface,
                        base: Optional[wd.World] = None, ):
    """
    Visualize the grasps in the grasp_info_list
    :param grasp_info_list: a list of grasp info, each grasp info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    :param gripper: the gripper to visualize the grasps
    :param base: the world to visualize the grasps
    :param toggle_show_all: whether to show all the grasps at the same time
    :return: None
    """
    assert isinstance(grasp_info_fid_gp, dict), "Invalid grasp_info_fid_gp!"
    assert isinstance(gripper, GripperInterface), "Invalid gripper!"
    if base is None:
        try:
            hasattr(base, 'run')
        except Exception as e:
            raise Exception(f"Invalid base: {e}")
    counter = [0, 0]
    vis_node: List[Optional[gm.GeometricModel]] = [None, None]
    gkeys = list(grasp_info_fid_gp.keys())

    def _show_gripper(task):
        """
        Show the gripper in the base
        """
        # if base.inputmgr.keymap['space']:  # press space to switch to the next grasp
        #     base.inputmgr.keymap['space'] = False
        if vis_node[0] is not None:
            vis_node[0].detach()  # detach the previous gripper
        if vis_node[1] is not None:
            vis_node[1].detach()  # detach the previous object
        idx = counter[1] // len(grasp_info_fid_gp[gkeys[0]])
        grasp_info = grasp_info_fid_gp[gkeys[idx]][counter[0]]
        jaw_width, jaw_center_pos, jaw_center_homomat, approach_direction = grasp_info
        gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_homomat[:3, :3], jaw_width)
        vis_node[0] = gripper.gen_meshmodel()
        vis_node[0].attach_to(base)
        vis_node[1] = obj.copy()
        vis_node[1].set_homomat(fp_list[idx])
        vis_node[1].attach_to(base)
        # update counter
        counter[0] = (counter[0] + 1) % len(grasp_info_fid_gp[gkeys[idx]])
        counter[1] = counter[1] + 1
        return task.again

    taskMgr.doMethodLater(0.1, _show_gripper, "show_gripper")  # show the gripper in the base


def visualize_grasp_pairs(grasp_pairs: List,
                          rgt_gripper: GripperInterface,
                          lft_gripper: GripperInterface,
                          base: Optional[wd.World] = None,
                          toggle_show_all: bool = False):
    """
    Visualize the grasps in the grasp_info_list
    :param grasp_info_list: a list of grasp info, each grasp info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    :param gripper: the gripper to visualize the grasps
    :param base: the world to visualize the grasps
    :param toggle_show_all: whether to show all the grasps at the same time
    :return: None
    """
    assert isinstance(rgt_gripper, GripperInterface), "Invalid gripper!"
    assert isinstance(lft_gripper, GripperInterface), "Invalid gripper!"
    if base is None:
        try:
            hasattr(base, 'run')
        except Exception as e:
            raise Exception(f"Invalid base: {e}")

    counter = [0]
    vis_node: List[Optional[gm.GeometricModel]] = [None, None]

    def _show_gripper(task):
        """
        Show the gripper in the base
        """
        # if base.inputmgr.keymap['space']:  # press space to switch to the next grasp
        #     base.inputmgr.keymap['space'] = False
        if vis_node[0] is not None:
            vis_node[0].detach()  # detach the previous gripper
        if vis_node[1] is not None:
            vis_node[1].detach()  # detach the previous gripper

        counter[0] = (counter[0] + 1) % len(grasp_pairs)
        grasp_info_rgt, grasp_info_lft = grasp_pairs[counter[0]]
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_rgt
        rgt_gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        vis_node[0] = rgt_gripper.gen_meshmodel()
        vis_node[0].attach_to(base)
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_lft
        lft_gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        vis_node[1] = lft_gripper.gen_meshmodel()
        vis_node[1].attach_to(base)

        return task.again

    taskMgr.doMethodLater(0.1, _show_gripper, "show_gripper_pair")  # show the gripper in the base


def visualize_handover(grasp_pairs_fp: List,
                       fp_list: List[np.ndarray],
                       obj: gm.GeometricModel,
                       rbt: RobotInterface,
                       ikjnt_fp_rgt: Dict[int, Dict[int, List]],
                       ikjnt_fp_lft: Dict[int, Dict[int, List]],
                       base: Optional[wd.World] = None,
                       toggle_show_all: bool = False):
    """
    Visualize the grasps in the grasp_info_list
    :param grasp_info_list: a list of grasp info, each grasp info is a list of 5 np.ndarray (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat)
        jaw_width: the width of the gripper
        jaw_center_pos: the position of the gripper
        jaw_center_rotmat: the rotation matrix of the gripper
        hnd_pos: the position of the hand
        hnd_rotmat: the rotation matrix of the hand
    :param gripper: the gripper to visualize the grasps
    :param base: the world to visualize the grasps
    :param toggle_show_all: whether to show all the grasps at the same time
    :return: None
    """
    assert isinstance(rbt, RobotInterface), "Invalid robot!"
    if base is None:
        try:
            hasattr(base, 'run')
        except Exception as e:
            raise Exception(f"Invalid base: {e}")

    counter = [0, 0]

    vis_node: List[Optional[gm.GeometricModel]] = [None, None]

    fp_id_txt = DirectLabel(text="Floating Pose ID: ", text_scale=0.1,
                            parent=base.a2dTopCenter, pos=(0, 0, -0.1), frameColor=(1, 1, 1, 1))
    gp_id_txt = DirectLabel(text="Grasping Pair ID: ", text_scale=0.1,
                            parent=base.a2dTopCenter, pos=(0, 0, -0.2), frameColor=(1, 1, 1, 1))
    note_txt = DirectLabel(text="Press 'Space' to Continue", text_scale=0.1,
                            parent=base.a2dBottomCenter, pos=(0, 0, 0.1), frameColor=(1, 1, 1, 1))
    rbt.gen_env_meshmodel().attach_to(base)
    def _show_gripper(task):
        """
        Show the gripper in the base
        """
        if base.inputmgr.keymap['space']:  # press space to switch to the next grasp
            base.inputmgr.keymap['space'] = False
            if counter[1] >= len(grasp_pairs_fp[counter[0]]):
                counter[0] = (counter[0] + 1) % len(fp_list)
                for i in range(len(fp_list)):
                    if counter[0] not in grasp_pairs_fp:
                        counter[0] = (counter[0] + 1) % len(fp_list)
                    else:
                        break
                counter[1] = 0
                # counter[0] = (counter[0] + 1) % len(fp_list)

            if vis_node[0] is not None:
                vis_node[0].detach()  # detach the previous gripper
            if vis_node[1] is not None:
                vis_node[1].detach()  # detach the previous gripper
            fp_id_txt["text"] = f"Floating Pose ID: {counter[0]} / {len(fp_list)}"
            gp_id_txt["text"] = f"Grasping Pair ID: {counter[1]} / {len(grasp_pairs_fp[counter[0]])}"
            rgt_id, lft_id = grasp_pairs_fp[counter[0]][counter[1]]
            counter[1] = counter[1] + 1
            jnt_rgt_list = ikjnt_fp_rgt[counter[0]].get(rgt_id, [])
            jnt_lft_list = ikjnt_fp_lft[counter[0]].get(lft_id, [])
            mm_collection = mc.ModelCollection()
            for jnt_rgt, jnt_rgt_approach in jnt_rgt_list:
                rbt.fk("rgt_arm", jnt_rgt)
                rbt.gen_arm_meshmodel("rgt_arm").attach_to(mm_collection)
            for jnt_lft, jnt_lft_approach in jnt_lft_list:
                rbt.fk("lft_arm", jnt_lft)
                rbt.gen_arm_meshmodel("lft_arm").attach_to(mm_collection)
            vis_node[0] = mm_collection
            vis_node[0].attach_to(base)
            vis_node[1] = obj.copy()
            vis_node[1].set_homomat(fp_list[counter[0]])
            vis_node[1].attach_to(base)
        return task.again

    taskMgr.doMethodLater(0.1, _show_gripper, "show_gripper_pair")  # show the gripper in the base
