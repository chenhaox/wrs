import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.ur7e._ur7e_common import UR7EBase, UR7EDualBase


class UR7E(UR7EBase):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur7e", enable_cc=True):
        mount_pos = np.array([0.1375, 0.0125, 0.0])
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         name=name,
                         enable_cc=enable_cc,
                         arm_home_conf=np.array([0, -rm.pi / 2, rm.pi / 2, -rm.pi / 2, 0, 0]),
                         arm_loc_pos=mount_pos + np.array([0, 0, 0.015]),
                         arm_loc_rotmat=np.eye(3),
                         fixture_specs=[
                             {
                                 "mesh_path": "wholetable.STL",
                                 "name": name + "_table",
                                 "loc_pos": np.zeros(3),
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, 1.0]),
                             },
                             {
                                 "mesh_path": "connnect.STL",
                                 "name": name + "_connector",
                                 "loc_pos": mount_pos,
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, 1.0]),
                             },
                         ])


class DualUR7E(UR7EDualBase):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="dual_ur7e", enable_cc=True):
        lower_mount_pos = np.array([0.6125, 0.5875, 0.0])
        upper_mount_pos = np.array([0.6125, -0.5875, 0.0])
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         name=name,
                         enable_cc=enable_cc,
                         active_arm_name="upper_arm",
                         arm_specs=[
                             {
                                 "name": "lower_arm",
                                 "loc_pos": lower_mount_pos + np.array([0, 0, 0.015]),
                                 "loc_rotmat": np.eye(3),
                                 "home_conf": np.array([0, -rm.pi / 2, rm.pi / 2, -rm.pi / 2, 0, 0]),
                             },
                             {
                                 "name": "upper_arm",
                                 "loc_pos": upper_mount_pos + np.array([0, 0, 0.015]),
                                 "loc_rotmat": np.eye(3),
                                 "home_conf": np.array(
                                     [rm.pi / 2, -rm.pi / 2, rm.pi / 2, -rm.pi, -rm.pi / 2, 0]),
                             },
                         ],
                         fixture_specs=[
                             {
                                 "mesh_path": "wholetable.STL",
                                 "name": name + "_table",
                                 "loc_pos": np.zeros(3),
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, 1.0]),
                             },
                             {
                                 "mesh_path": "connnect.STL",
                                 "name": name + "_lower_connector",
                                 "loc_pos": lower_mount_pos,
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, 1.0]),
                             },
                             {
                                 "mesh_path": "connnect.STL",
                                 "name": name + "_upper_connector",
                                 "loc_pos": upper_mount_pos,
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, 1.0]),
                             },
                         ])


if __name__ == "__main__":
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = DualUR7E(enable_cc=True)

    robot.fk("upper_arm", np.array(
        [np.radians(110), 0, 0, 0, 0, 0]
    ))
    robot.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=True, alpha=.7).attach_to(base)
    robot.show_cdprim()
    robot.is_collided()
    print(robot.is_collided())
    base.run()
