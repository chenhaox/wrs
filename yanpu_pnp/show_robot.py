import argparse

from trac_ik import TracIK as _TracIK

import numpy as np

from wrs import wd, mgm

from yanpu_pnp import config as cfgutils
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene


def run(cfg):
    rack_pos = cfgutils.array(cfg["rack"]["base_pos"])
    base = wd.World(cam_pos=[2.0, -1.8, 1.55], lookat_pos=rack_pos + np.array([0.0, 0.0, 0.55]))
    mgm.gen_frame().attach_to(base)
    mgm.gen_frame(pos=rack_pos,
                  rotmat=cfgutils.rack_rotmat(cfg),
                  ax_length=.18,
                  ax_radius=.006).attach_to(base)

    object_specs = cfgutils.object_specs(cfg)
    robot = robot_factory.build_robot(cfg, enable_cc=True)
    robot.gen_meshmodel(alpha=.9,
                        toggle_tcp_frame=True,
                        toggle_jnt_frames=True,
                        toggle_flange_frame=True,
                        toggle_cdprim=True).attach_to(base)
    obstacle_list, payload_dict = scene.build_scene(base, cfg, object_specs, attach_visuals=True)

    vertical = cfg["rack"]["vertical_frame"]
    horizontal = cfg["rack"]["horizontal_frame"]
    print("Rack base pos:", cfgutils.array(cfg["rack"]["base_pos"]))
    print("Rack yaw deg:", cfg["rack"]["yaw_deg"])
    print("Arm distance:", cfg["rack"]["arm_distance"])
    print("Left arm mount euler deg:", cfg["rack"]["left_arm_mount_euler_deg"])
    print("Vertical frame xyz:", np.array([vertical["x_length"], vertical["y_length"], vertical["height"]]))
    print("Horizontal frame xyz:", np.array([horizontal["x_length"], horizontal["y_length"], horizontal["thickness"]]))
    print("Left mount pos:", robot.lft_mount_pos)
    print("Right mount pos:", robot.rgt_mount_pos)
    print("Left TCP:", robot.lft_arm.gl_tcp_pos)
    print("Right TCP:", robot.rgt_arm.gl_tcp_pos)
    print("Obstacle count:", len(obstacle_list))
    print("Payloads:", list(payload_dict))
    print("Self/environment collision:", robot.is_collided())
    base.run()


def main():
    parser = argparse.ArgumentParser(description="Show UR7E + DH50 robot and scene from YAML config.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run(cfgutils.load_config(args.config))


if __name__ == "__main__":
    main()
