import numpy as np

from wrs.robot_sim.robots.ur7e._ur7e_common import UR7EBase


class UR7E(UR7EBase):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur7e", enable_cc=True):
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         name=name,
                         enable_cc=enable_cc,
                         arm_home_conf=np.zeros(6),
                         arm_loc_pos=np.zeros(3),
                         arm_loc_rotmat=np.eye(3))


if __name__ == "__main__":
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = UR7E(enable_cc=True)
    robot.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=True, alpha=.7).attach_to(base)
    print(robot.is_collided())
    base.run()
