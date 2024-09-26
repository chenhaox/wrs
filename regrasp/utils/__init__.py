"""
Created on 2024/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
import copy
from typing import Literal

from .file_sys import *
from .data_structure import *
from .visualize import *

import modeling.collision_model as cm

MODEL_PATH = Path(__file__).resolve().parent.parent.joinpath('model')


def load_handover_data(data_path: str or Path) -> HandoverData:
    data_path = Path(data_path)
    if not data_path.exists():
        raise Exception(f"{data_path} does not exist! Please check the path of the data!")
    data = load_pickle(data_path)
    return HandoverData(*data)


def get_obj_cm_by_name(obj_name: str):
    model_path = MODEL_PATH.joinpath(obj_name)
    if not model_path.exists():
        raise Exception(f"{model_path} does not exist! Please check the path of the model!")
    return cm.CollisionModel(str(model_path), name=obj_name)


def get_rbt_by_name(r_name: Literal['ur3e', 'yumi', 'ur3',]):
    if r_name == 'ur3e':
        import robot_sim.robots.ur3e_dual.ur3e_dual as ur3e
        return ur3e.UR3EDual()
    elif r_name == 'yumi':
        import robot_sim.robots.yumi.yumi as yumi
        return yumi.Yumi()
    elif r_name == 'ur3':
        import robot_sim.robots.ur3_dual.ur3_dual as ur3
        return ur3.UR3Dual()
    else:
        raise Exception(f"{r_name} is not a valid robot name! Please check the name of the robot!")


def get_gripper_by_name(g_name: Literal['robotiqhe']):
    if g_name == 'robotiqhe':
        import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rqh
        return rqh.RobotiqHE()
    else:
        raise Exception(f"{g_name} is not a valid gripper name! Please check the name of the gripper!")


class JJState:
    def __init__(self,
                 jnt_val: Optional[np.ndarray] = None,
                 jawwidth: Optional[float] = None, ):
        self.jnt_val = jnt_val
        self.jawwidth = jawwidth


class JJOState:
    def __init__(self, ):
        self.jjo_state_dict = {}
        self.obj_homomat = None

    def set_state(self,
                  component_name: str,
                  jnt_val: Optional[np.ndarray] = None,
                  jawwidth: Optional[float] = None,
                  obj_homomat: Optional[np.ndarray] = None):
        self.jjo_state_dict[component_name] = JJState(jnt_val,
                                                      jawwidth, )
        self.obj_homomat = obj_homomat

    def __getitem__(self, item: str) -> JJState:
        assert item in self.jjo_state_dict.keys(), f"Component {item} is not in the state!"
        assert isinstance(self.jjo_state_dict[item], JJState), f"State of {item} is not a JJState object!"
        return self.jjo_state_dict[item]


def get_jjostate_rbt_obj(rbt: 'RobotInterface' = None, obj: cm.CollisionModel = None):
    is_rbt_valid = rbt is not None
    is_obj_valid = obj is not None
    component_name_list = [component_name for component_name in rbt.manipulator_dict.keys() if "arm" in component_name]
    state = JJOState()
    for name in component_name_list:
        state.set_state(name,
                        jnt_val=rbt.get_jnt_values(name) if is_rbt_valid else None,
                        jawwidth=rbt.get_jawwidth(name) if is_rbt_valid else None,
                        obj_homomat=obj.get_homomat() if is_obj_valid else None)
    return state


def get_jjostate_rbt_objhomomat(rbt: 'RobotInterface' = None, obj_homomat: np.ndarray = None):
    is_rbt_valid = rbt is not None
    component_name_list = [component_name for component_name in rbt.manipulator_dict.keys() if "arm" in component_name]
    state = JJOState()
    for name in component_name_list:
        state.set_state(name,
                        jnt_val=rbt.get_jnt_values(name) if is_rbt_valid else None,
                        jawwidth=rbt.get_jawwidth(name) if is_rbt_valid else None,
                        obj_homomat=obj_homomat)
    return state


class RbtTrajData:
    def __init__(self, ):
        self.jjo_state_list = []

    def add_state(self, state: JJOState, ):
        self.jjo_state_list.append(state)

    def copy_last_state(self) -> JJOState:
        if len(self.jjo_state_list) == 0:
            raise Exception("No state in the list!")
        return copy.deepcopy(self.jjo_state_list[-1])

    def __len__(self) -> int:
        return len(self.jjo_state_list)

    def __getitem__(self, index) -> JJOState:
        assert isinstance(index, int), f"Index {index} is not an integer!"
        assert len(self.jjo_state_list) > 0, "No state in the list!"
        assert 0 <= index < len(self.jjo_state_list), f"Index {index} is out of range!"
        assert isinstance(self.jjo_state_list[index], JJOState), f"State {index} is not a JJOState object!"
        return self.jjo_state_list[index]
