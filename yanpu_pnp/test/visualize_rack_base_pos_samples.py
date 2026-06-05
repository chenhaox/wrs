import argparse
import itertools

from trac_ik import TracIK as _TracIK

import numpy as np
from panda3d.core import TextNode

from wrs import wd, mgm

from yanpu_pnp import config as cfgutils
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene
from yanpu_pnp import search_config as search_cfgutils


def iter_rack_base_pos_samples(search_cfg):
    spec = search_cfgutils.base_pos_samples(search_cfg)
    for x, y, z in itertools.product(spec["x"], spec["y"], spec["z"]):
        yield np.array([x, y, z], dtype=float)


def _axis_limits(points):
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs


def attach_world_label(base, text, pos, scale=.026, rgba=(0.02, 0.02, 0.02, 1.0)):
    text_node = TextNode("rack_sample_label")
    text_node.setText(text)
    text_node.setAlign(TextNode.ACenter)
    text_node.setTextColor(*rgba)
    label = base.render.attachNewNode(text_node)
    label.setPos(float(pos[0]), float(pos[1]), float(pos[2]))
    label.setScale(scale)
    label.setBillboardPointEye()
    return label


def attach_sample_grid(base, points, current_pos, show_labels=True):
    mins, maxs = _axis_limits(points)
    center = (mins + maxs) * 0.5
    mgm.gen_frame(pos=center, ax_length=.18, ax_radius=.004).attach_to(base)

    for point in points:
        is_current = np.linalg.norm(point - current_pos) < 1e-9
        rgb = np.array([1.0, .12, .06]) if is_current else np.array([.0, .42, 1.0])
        radius = .022 if is_current else .014
        alpha = .95 if is_current else .58
        mgm.gen_sphere(pos=point, radius=radius, rgb=rgb, alpha=alpha).attach_to(base)
        if show_labels:
            label = f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"
            if is_current:
                label = f"current\n{label}"
            attach_world_label(base,
                               label,
                               point + np.array([0.0, 0.0, .035]),
                               scale=.024 if is_current else .018,
                               rgba=(.72, .04, .02, 1.0) if is_current else (.03, .12, .22, .82))

    for x in sorted(set(points[:, 0])):
        for y in sorted(set(points[:, 1])):
            layer_points = points[(np.isclose(points[:, 0], x)) & (np.isclose(points[:, 1], y))]
            layer_points = layer_points[np.argsort(layer_points[:, 2])]
            for start, end in zip(layer_points[:-1], layer_points[1:]):
                mgm.gen_stick(spos=start,
                              epos=end,
                              radius=.0025,
                              rgb=np.array([.18, .32, .48]),
                              alpha=.22).attach_to(base)


def attach_current_setup(base, cfg, show_scene=True, show_robot=True):
    object_specs = cfgutils.object_specs(cfg)
    if show_scene:
        scene.build_scene(base, cfg, object_specs, attach_visuals=True)
    if show_robot:
        robot = robot_factory.build_robot(cfg, enable_cc=True)
        robot.gen_meshmodel(alpha=.36,
                            toggle_tcp_frame=True,
                            toggle_jnt_frames=False,
                            toggle_flange_frame=False,
                            toggle_cdprim=True).attach_to(base)
        mgm.gen_frame(pos=cfgutils.array(cfg["rack"]["base_pos"]),
                      rotmat=cfgutils.rack_rotmat(cfg),
                      ax_length=.16,
                      ax_radius=.005).attach_to(base)
        print("Current left mount:", robot.lft_mount_pos)
        print("Current right mount:", robot.rgt_mount_pos)


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled rack base_pos positions from YAML search config.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--no-scene", action="store_true", help="Only draw rack base_pos samples, without environment.")
    parser.add_argument("--no-robot", action="store_true", help="Do not draw the current rack/robot.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide 3D numeric labels for sampled points.")
    args = parser.parse_args()

    cfg = cfgutils.load_config(args.config)
    search_cfg = search_cfgutils.load_search_config(args.search_config)
    base_pos_spec = search_cfgutils.base_pos_samples(search_cfg)
    points = np.array(list(iter_rack_base_pos_samples(search_cfg)))
    current_pos = cfgutils.array(cfg["rack"]["base_pos"])
    lookat_pos = points.mean(axis=0) if len(points) else current_pos
    base = wd.World(cam_pos=[lookat_pos[0] + 1.2, lookat_pos[1] - 1.6, lookat_pos[2] + 1.0],
                    lookat_pos=lookat_pos)
    mgm.gen_frame().attach_to(base)
    attach_current_setup(base, cfg, show_scene=not args.no_scene, show_robot=not args.no_robot)
    attach_sample_grid(base, points, current_pos, show_labels=not args.hide_labels)
    print(f"Rack base_pos samples: {len(points)}")
    print("Sample x values:", base_pos_spec["x"])
    print("Sample y values:", base_pos_spec["y"])
    print("Sample z values:", base_pos_spec["z"])
    print("Current rack base_pos:", current_pos)
    print("Sample min:", points.min(axis=0))
    print("Sample max:", points.max(axis=0))
    base.run()


if __name__ == "__main__":
    main()
