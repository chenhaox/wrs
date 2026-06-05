import argparse
import math

from trac_ik import TracIK as _TracIK

import numpy as np
from panda3d.core import TextNode

from wrs import wd, mgm

from yanpu_pnp import config as cfgutils
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import search_config as search_cfgutils


def iter_left_arm_mount_euler_samples(search_cfg):
    for euler_deg in search_cfgutils.left_arm_mount_euler_sample_list(search_cfg):
        yield np.array(euler_deg, dtype=float)


def attach_world_label(base, text, pos, scale=.035, rgba=(0.02, 0.02, 0.02, 1.0)):
    text_node = TextNode("arm_mount_sample_label")
    text_node.setText(text)
    text_node.setAlign(TextNode.ACenter)
    text_node.setTextColor(*rgba)
    label = base.render.attachNewNode(text_node)
    label.setPos(float(pos[0]), float(pos[1]), float(pos[2]))
    label.setScale(scale)
    label.setBillboardPointEye()
    return label


def grid_base_positions(reference_pos, count, columns, spacing_x, spacing_y):
    rows = int(math.ceil(count / columns))
    centered_cols = (columns - 1) * 0.5
    centered_rows = (rows - 1) * 0.5
    for index in range(count):
        row = index // columns
        col = index % columns
        yield reference_pos + np.array([
            (col - centered_cols) * spacing_x,
            (centered_rows - row) * spacing_y,
            0.0,
        ])


def _is_current_euler(euler_deg, cfg):
    return np.allclose(euler_deg, cfgutils.array(cfg["rack"]["left_arm_mount_euler_deg"]), atol=1e-9)


def attach_mount_direction_arrows(base, mount_pos, mount_rotmat, x_rgb, y_rgb, length=.24):
    x_tip = mount_pos + mount_rotmat[:, 0] * length
    y_tip = mount_pos + mount_rotmat[:, 1] * length
    mgm.gen_arrow(spos=mount_pos,
                  epos=x_tip,
                  rgb=x_rgb,
                  alpha=.9,
                  stick_radius=.007).attach_to(base)
    mgm.gen_arrow(spos=mount_pos,
                  epos=y_tip,
                  rgb=y_rgb,
                  alpha=.76,
                  stick_radius=.005).attach_to(base)


def attach_mount_candidate(base, base_cfg, euler_deg, rack_base_pos, index, args):
    cfg = cfgutils.apply_search_candidate(base_cfg, {
        "rack_base_pos": rack_base_pos,
        "left_arm_mount_euler_deg": euler_deg,
    })
    robot = robot_factory.build_robot(cfg, enable_cc=False)
    is_current = _is_current_euler(euler_deg, base_cfg)

    robot.gen_meshmodel(alpha=args.alpha,
                        toggle_tcp_frame=True,
                        toggle_jnt_frames=False,
                        toggle_flange_frame=False,
                        toggle_cdprim=args.show_cdprim).attach_to(base)
    mgm.gen_frame(pos=cfgutils.array(cfg["rack"]["base_pos"]),
                  rotmat=cfgutils.rack_rotmat(cfg),
                  ax_length=.16,
                  ax_radius=.004).attach_to(base)
    mgm.gen_frame(pos=robot.lft_mount_pos,
                  rotmat=robot.lft_mount_rotmat,
                  ax_length=.12,
                  ax_radius=.003).attach_to(base)
    mgm.gen_frame(pos=robot.rgt_mount_pos,
                  rotmat=robot.rgt_mount_rotmat,
                  ax_length=.12,
                  ax_radius=.003).attach_to(base)
    if not args.hide_mount_arrows:
        attach_mount_direction_arrows(base,
                                      robot.lft_mount_pos,
                                      robot.lft_mount_rotmat,
                                      x_rgb=np.array([.02, .72, .24]),
                                      y_rgb=np.array([.86, .74, .06]))
        attach_mount_direction_arrows(base,
                                      robot.rgt_mount_pos,
                                      robot.rgt_mount_rotmat,
                                      x_rgb=np.array([.94, .08, .06]),
                                      y_rgb=np.array([1.0, .46, .02]))
    mgm.gen_sphere(pos=robot.lft_mount_pos,
                   radius=.025,
                   rgb=np.array([.02, .75, .22]),
                   alpha=.85).attach_to(base)
    mgm.gen_sphere(pos=robot.rgt_mount_pos,
                   radius=.025,
                   rgb=np.array([.92, .08, .06]),
                   alpha=.85).attach_to(base)

    label_pos = rack_base_pos + np.array([0.0, 0.0,
                                          float(cfg["rack"]["vertical_frame"]["height"]) + .28])
    label = f"#{index}\nroll={euler_deg[0]:.0f}, pitch={euler_deg[1]:.0f}, yaw={euler_deg[2]:.0f}"
    if is_current:
        label = f"current\n{label}"
    attach_world_label(base,
                       label,
                       label_pos,
                       scale=.034 if is_current else .028,
                       rgba=(.72, .04, .02, 1.0) if is_current else (.02, .02, .02, .92))

    print(f"#{index}: left_arm_mount_euler_deg={euler_deg.tolist()}, "
          f"rack_base_pos={np.round(rack_base_pos, 4).tolist()}, "
          f"lft_mount={np.round(robot.lft_mount_pos, 4).tolist()}, "
          f"rgt_mount={np.round(robot.rgt_mount_pos, 4).tolist()}, "
          f"lft_x={np.round(robot.lft_mount_rotmat[:, 0], 4).tolist()}, "
          f"lft_y={np.round(robot.lft_mount_rotmat[:, 1], 4).tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize all left_arm_mount_euler_deg samples from YAML.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--spacing-x", type=float, default=1.15)
    parser.add_argument("--spacing-y", type=float, default=1.35)
    parser.add_argument("--alpha", type=float, default=.38)
    parser.add_argument("--max-mounts", type=int, default=None)
    parser.add_argument("--show-cdprim", action="store_true", help="Also draw robot collision primitives.")
    parser.add_argument("--hide-mount-arrows", action="store_true",
                        help="Hide thick local +X/+Y arrows at each arm mount.")
    args = parser.parse_args()

    if args.columns <= 0:
        raise ValueError("--columns must be positive")

    cfg = cfgutils.load_config(args.config)
    search_cfg = search_cfgutils.load_search_config(args.search_config)
    euler_samples = list(iter_left_arm_mount_euler_samples(search_cfg))
    if args.max_mounts is not None:
        euler_samples = euler_samples[:args.max_mounts]
    if not euler_samples:
        raise RuntimeError("No left_arm_mount_euler_deg samples found in search config.")

    reference_pos = cfgutils.array(cfg["rack"]["base_pos"])
    rack_base_positions = list(grid_base_positions(reference_pos,
                                                  len(euler_samples),
                                                  args.columns,
                                                  args.spacing_x,
                                                  args.spacing_y))
    lookat_pos = np.mean(np.asarray(rack_base_positions), axis=0) + np.array([0.0, 0.0, .35])
    rows = int(math.ceil(len(euler_samples) / args.columns))
    cam_pos = [
        lookat_pos[0] + max(1.8, args.spacing_x * args.columns * .42),
        lookat_pos[1] - max(2.0, args.spacing_y * rows * .7),
        lookat_pos[2] + 1.65,
    ]
    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos)
    mgm.gen_frame(ax_length=.25, ax_radius=.006).attach_to(base)

    for index, (euler_deg, rack_base_pos) in enumerate(zip(euler_samples, rack_base_positions), start=1):
        attach_mount_candidate(base, cfg, euler_deg, rack_base_pos, index, args)

    arm_spec = search_cfgutils.left_arm_mount_euler_samples(search_cfg)
    print(f"Arm mount samples: {len(euler_samples)}")
    print("Roll values:", arm_spec["roll"])
    print("Pitch values:", arm_spec["pitch"])
    print("Yaw values:", arm_spec["yaw"])
    print("Current left_arm_mount_euler_deg:", cfg["rack"]["left_arm_mount_euler_deg"])
    base.run()


if __name__ == "__main__":
    main()
