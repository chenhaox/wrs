"""
Created on 2024/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
from collections import namedtuple

HandoverData = namedtuple('HandoverData', ['obj_name',
                                           'retract_dist',
                                           'grasp_list_rgt',
                                           'grasp_list_lft',
                                           'floating_pose_list',
                                           'identity_grasp_pairs',
                                           'grasp_pairs_fp',
                                           'grasp_list_fp_rgt',
                                           'grasp_list_fp_lft',
                                           'ik_fp_fid_gid_rgt',
                                           'ik_fp_fid_gid_lft',
                                           'ik_jnts_fp_gid_rgt',
                                           'ik_jnts_fp_gid_lft',
                                           'manipulability_fp_gid_rgt',
                                           'manipulability_fp_gid_lft',])
