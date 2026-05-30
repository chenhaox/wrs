import os  #文件路径/标准库
import math
import numpy as np
import modeling.model_collection as mc  #自定义模型集合简称mc
import modeling.collision_model as cm   #自定义碰撞检测模型简称cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl   #自定义关节模块
import basis.robot_math as rm   #自定义机器人工具包
import robot_sim.end_effectors.gripper.gripper_interface as gp   #自定义末端夹爪类
import trimesh
import visualization.panda.world as wd
import modeling.geometric_model as gm

base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
stl_path = r'C:\Users\86159\Desktop\ROPG\wrs\wrs-fujikoshi-main\robot_sim\end_effectors\gripper\dh50\meshes\shell assembly_lft.STL'
# stl_path = r'C:\Users\86159\Desktop\ROPG\wrs\wrs-fujikoshi-main\robot_sim\end_effectors\gripper\dh60\meshes\fingertip1.stl'

model = cm.CollisionModel(stl_path)
model.attach_to(base)
gm.gen_frame().attach_to(base)

base.run()