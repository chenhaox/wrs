import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.ur7e._ur7e_common import UR7EBase


class UR7E(UR7EBase):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur7e", enable_cc=True):
        mount_pos = np.array([0.6125, -0.5875, 0.0])
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         name=name,
                         enable_cc=enable_cc,
                         arm_home_conf=np.array([rm.pi / 2, -rm.pi / 2, rm.pi / 2, -rm.pi, -rm.pi / 2, 0]),
                         arm_loc_pos=mount_pos + np.array([0, 0, 0.015]),
                         arm_loc_rotmat=np.eye(3),
                         fixture_specs=[
                             {
                                 "mesh_path": "connnect.STL",
                                 "name": name + "_connector",
                                 "loc_pos": mount_pos,
                                 "cdprim_type": mcm.const.CDPrimType.AABB,
                                 "rgba": np.array([.35, .35, .35, .3]),
                             }
                         ])


if __name__ == "__main__":
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = UR7E(enable_cc=True)
    robot.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=True, alpha=.7).attach_to(base)
    print(robot.is_collided())
    base.run()
