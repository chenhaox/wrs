"""
Created on 2024/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
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


def get_obj_cm(obj_name: str):
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
