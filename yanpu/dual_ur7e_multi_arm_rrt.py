import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wrs import wd, mgm
from wrs.motion.probabilistic import multi_arm_rrt_connect as marrtc
from wrs.robot_sim.robots.ur7e.ur7e_table import DualUR7E


class AnimationData:

    def __init__(self, mesh_list, hold_frames=12):
        self.mesh_list = mesh_list
        self.hold_frames = hold_frames
        self.counter = 0
        self.hold_counter = 0
        self.current_mesh = None


def update(animation_data, task):
    if animation_data.current_mesh is not None:
        animation_data.current_mesh.detach()
    animation_data.current_mesh = animation_data.mesh_list[animation_data.counter]
    animation_data.current_mesh.attach_to(base)
    if animation_data.counter == len(animation_data.mesh_list) - 1:
        animation_data.hold_counter += 1
        if animation_data.hold_counter >= animation_data.hold_frames:
            animation_data.counter = 0
            animation_data.hold_counter = 0
    else:
        animation_data.counter += 1
    return task.again


def main(toggle_visual=True):
    global base
    base = wd.World(cam_pos=[2.2, -1.8, 1.4], lookat_pos=[0.35, -0.25, 0.25])
    mgm.gen_frame().attach_to(base)

    robot = DualUR7E(enable_cc=True)
    lower_start = robot.get_jnt_values("lower_arm")
    upper_start = robot.get_jnt_values("upper_arm")
    lower_goal = np.array([1.386, -2.861, 0.690, -2.460, 0.926, 1.341])
    upper_goal = np.array([2.159, -1.026, 0.127, -3.985, -2.814, -0.569])

    planner = marrtc.MultiArmRRTConnect(robot)
    planner.add_arm(name="upper_arm", start_conf=upper_start, goal_conf=upper_goal)
    planner.add_arm(name="lower_arm", start_conf=lower_start, goal_conf=lower_goal)
    mot_data = planner.plan(ext_dist=.1,
                            max_time=20.0,
                            smoothing_n_iter=0,
                            coordination_ext_dist=.04,
                            moving_tcp_clearance=1.0,
                            toggle_dbg=True)
    if mot_data is None:
        raise RuntimeError("Failed to plan a collision-free multi-arm schedule.")

    print(f"Planned {len(mot_data)} synchronized states.")
    print("Schedule states:", planner.schedule_state_list)
    mot_data.apply(len(mot_data) - 1)
    tcp_dist = np.linalg.norm(robot.manipulator_dict["lower_arm"].gl_tcp_pos -
                              robot.manipulator_dict["upper_arm"].gl_tcp_pos)
    print(f"Goal TCP distance: {tcp_dist:.3f}m")

    if toggle_visual:
        mesh_list = []
        for i in range(len(mot_data)):
            mot_data.apply(i)
            mesh_list.append(robot.gen_meshmodel(alpha=.9,
                                                toggle_tcp_frame=True,
                                                toggle_flange_frame=False))
        mot_data.apply(0)
        animation_data = AnimationData(mesh_list=mesh_list)
        taskMgr.doMethodLater(0.06,
                              update,
                              "dual_ur7e_multi_arm_rrt_update",
                              extraArgs=[animation_data],
                              appendTask=True)
        base.run()
    return mot_data


if __name__ == "__main__":
    main(toggle_visual=True)
