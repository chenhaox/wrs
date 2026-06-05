from dataclasses import dataclass

import numpy as np


@dataclass
class FrameState:
    conf_dict: dict
    jaw_width_dict: dict
    payload_mode_dict: dict


@dataclass
class PickPlaceTask:
    arm_name: str
    object_name: str
    grasp_index: int
    pick_pose_index: int
    symmetry_angle: float
    place_pose_index: int
    place_symmetry_angle: float
    pick_conf: np.ndarray
    lift_conf: np.ndarray
    pre_place_conf: np.ndarray
    place_conf: np.ndarray
    jaw_width: float
    pick_pose: tuple
    lift_pose: tuple
    pre_place_pose: tuple
    place_pose: tuple
    payload_rel_pose: tuple
    pick_solution_type: str
    lift_solution_type: str
    pre_place_solution_type: str
    place_solution_type: str


class PickPlacePlanningError(RuntimeError):

    def __init__(self, message, conf_dict=None, arm_name=None, object_name=None, grasp_indices=None):
        super().__init__(message)
        self.conf_dict = {} if conf_dict is None else conf_dict
        self.arm_name = arm_name
        self.object_name = object_name
        self.grasp_indices = [] if grasp_indices is None else list(grasp_indices)


class MultiArmPlanningError(RuntimeError):

    def __init__(self, message, debug_info=None):
        super().__init__(message)
        self.debug_info = {} if debug_info is None else debug_info
