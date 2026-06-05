import numpy as np

import wrs.basis.robot_math as rm

BOX_RGBA = np.array([.05, .24, .56, .72])
BOX1_CENTER = np.array([0.807, -0.245, 0.8])
BOX2_CENTER = np.array([0.232, 0.32, 0.72])
BOX1_PLACE = np.array([0.323, -0.36, 1.0])
BOX2_PLACE = np.array([-0.094, -0.149, 0.9])

OBJECT_UP_DOWN_ROTMAT = rm.rotmat_from_euler(0, np.pi, 0)

U_PICK_POSE = (BOX2_CENTER + np.array([0.1, 0.0, 0.06]), OBJECT_UP_DOWN_ROTMAT)
U625_PICK_POSE = (BOX1_CENTER + np.array([0.2, 0.1, 0.03]), np.eye(3))
U_PLACE_POSE = (BOX2_PLACE, OBJECT_UP_DOWN_ROTMAT)
U625_PLACE_POSE = (BOX1_PLACE, np.eye(3))
U_RGBA = np.array([.02, .58, .72, .96])
U625_RGBA = np.array([.95, .48, .08, .96])
PICK_LIFT_HEIGHT = 0.15
PLACE_APPROACH_DISTANCE = 0.1
RRT_TCP_Z_AXIS_MAX_ANGLE = np.radians(60)
RRT_TCP_Z_AXIS_WORLD = -rm.const.z_ax

RACK_BASE_OFFSET = np.array([.2, 0.2, -0.3])
RACK_BASE_POS = np.array([0.7, 0.2, 0.7]) + RACK_BASE_OFFSET
RACK_YAW = np.radians(45)
RACK_ROT = rm.rotmat_from_axangle(rm.const.z_ax, RACK_YAW)
RACK_VERTICAL_FRAME_HEIGHT = 0.4
RACK_VERTICAL_FRAME_X_LENGTH = .1
RACK_VERTICAL_FRAME_Y_LENGTH = .1
RACK_VERTICAL_FRAME_XY = np.array([RACK_VERTICAL_FRAME_X_LENGTH, RACK_VERTICAL_FRAME_Y_LENGTH])
RACK_HORIZONTAL_FRAME_X_LENGTH = .24
RACK_HORIZONTAL_FRAME_Y_LENGTH = .2
RACK_HORIZONTAL_FRAME_THICKNESS = .08
RACK_VERTICAL_FRAME_RGB = np.array([.60, .62, .62])
RACK_HORIZONTAL_FRAME_RGB = np.array([.05, .16, .32])
RACK_VERTICAL_FRAME_ALPHA = .58
RACK_HORIZONTAL_FRAME_ALPHA = .86
RACK_ARM_Y_OFFSET_REFERENCE_FRAME_Y_LENGTH = .70
RACK_ARM_Y_OFFSET = 0.258485281374
RACK_LFT_ARM_LOC_ROTMAT = rm.rotmat_from_euler(-0.5 * np.pi / 3.0, 0, np.pi/4)
RACK_RGT_ARM_LOC_ROTMAT = (rm.rotmat_from_euler(0.5 * np.pi / 3.0, 0, -np.pi/4) @
                           rm.rotmat_from_euler(0, 0, np.pi))

UR3_DUAL_LFT_HOME_CONF = np.zeros(6)
UR3_DUAL_RGT_HOME_CONF = np.zeros(6)

BOX1_PART_OFFSETS = {
    "1": np.array([0.0, -0.017, 0.02]),
    "2": np.array([0.0, 0.013, 0.02]),
    "3": np.array([-0.013, 0.0, 0.02]),
    "4": np.array([0.013, 0.0, 0.02]),
    "5": np.zeros(3),
}
BOX2_PART_OFFSETS = {
    "1": np.array([0.007, 0.0, 0.02]),
    "2": np.array([-0.01, 0.0, 0.02]),
    "3": np.array([0.0, -0.017, 0.02]),
    "4": np.array([0.0, 0.005, 0.02]),
    "5": np.zeros(3),
}

U_PLACE_POSITIONS = [BOX2_PLACE,
                     BOX2_PLACE + np.array([0.0, 0.1, 0.0]),
                     BOX2_PLACE + np.array([0.0, 0.2, 0.0])]
U_GRASP_POSITIONS = [BOX2_CENTER + np.array([0.1, 0.0, 0.0]),
                     BOX2_CENTER + np.array([-0.1, 0.2, 0.0]),
                     BOX2_CENTER + np.array([0.1, -0.2, 0.0]),
                     BOX2_CENTER + np.array([-0.1, -0.2, 0.0])]
U625_PLACE_POSITIONS = [BOX1_PLACE,
                        BOX1_PLACE + np.array([-0.1, 0.0, 0.0]),
                        BOX1_PLACE + np.array([-0.2, 0.0, 0.0])]
U625_GRASP_POSITIONS = [BOX1_CENTER + np.array([0.2, 0.1, 0.0]),
                        BOX1_CENTER + np.array([-0.2, 0.1, 0.0]),
                        BOX1_CENTER + np.array([0.2, -0.1, 0.0]),
                        BOX1_CENTER + np.array([-0.2, -0.1, 0.0])]
U_PLACE_POSES = [(pos, U_PLACE_POSE[1]) for pos in U_PLACE_POSITIONS]
U625_PLACE_POSES = [(pos, U625_PLACE_POSE[1]) for pos in U625_PLACE_POSITIONS]
U_PICK_POSE_OFFSET = U_PICK_POSE[0] - U_GRASP_POSITIONS[0]
U625_PICK_POSE_OFFSET = U625_PICK_POSE[0] - U625_GRASP_POSITIONS[0]
U_PICK_POSE_CANDIDATES = [(pos + U_PICK_POSE_OFFSET, U_PICK_POSE[1]) for pos in U_GRASP_POSITIONS]
U625_PICK_POSE_CANDIDATES = [(pos + U625_PICK_POSE_OFFSET, U625_PICK_POSE[1]) for pos in U625_GRASP_POSITIONS]
PICK_ROTATIONAL_SYMMETRY_ANGLE_COUNT = 1

# pick_pose is the object's world pose before picking. The actual gripper TCP
# target is pick_pose composed with the grasp selected from grasp_pickle.
# place_poses are the candidate object world poses after placing.
OBJECT_SPECS = {
    "u": {
        "mesh": "u.STL",
        "pick_pose": U_PICK_POSE,
        "pick_pose_candidates": U_PICK_POSE_CANDIDATES,
        "rotational_symmetry_axis": rm.const.z_ax,
        "rotational_symmetry_angle_count": PICK_ROTATIONAL_SYMMETRY_ANGLE_COUNT,
        "rgba": U_RGBA,
        "grasp_pickle": "U1_dh50.pickle",
        "grasp_key": "u",
        "grasp_positions": U_GRASP_POSITIONS,
        "place_poses": U_PLACE_POSES,
    },
    "U625": {
        "mesh": "U625.STL",
        "pick_pose": U625_PICK_POSE,
        "pick_pose_candidates": U625_PICK_POSE_CANDIDATES,
        "rgba": U625_RGBA,
        "grasp_pickle": "U625_dh50.pickle",
        "grasp_key": "U625",
        "grasp_positions": U625_GRASP_POSITIONS,
        "place_poses": U625_PLACE_POSES,
    },
}

DUAL_PICK_PLACE_SPECS = {
    "lft_arm": {
        "object_name": "u",
        "place_index": 2,
    },
    "rgt_arm": {
        "object_name": "U625",
        "place_index": 0,
    },
}
