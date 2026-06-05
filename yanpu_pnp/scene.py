import os

import numpy as np

from wrs import mgm
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.end_effectors.grippers.dh50.dh50 import Dh50

from yanpu_pnp import config as cfgutils
from yanpu_pnp import grasping


def make_collision_model(mesh_name,
                         pos=np.zeros(3),
                         rotmat=np.eye(3),
                         rgba=None,
                         ex_radius=.001,
                         attach_to=None,
                         mesh_dir=None):
    mesh_path = os.path.join(mesh_dir, mesh_name)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    cmodel = mcm.CollisionModel(mesh_path,
                                cdprim_type=mcm.const.CDPrimType.AABB,
                                ex_radius=ex_radius)
    cmodel.pos = np.asarray(pos, dtype=float)
    cmodel.rotmat = rotmat
    if rgba is not None:
        cmodel.rgba = np.asarray(rgba, dtype=float)
    if attach_to is not None:
        cmodel.attach_to(attach_to)
    return cmodel


def _rotmat_from_scene_spec(spec):
    if "rot_euler_deg" in spec:
        return rm.rotmat_from_euler(*np.radians(np.asarray(spec["rot_euler_deg"], dtype=float)))
    return np.eye(3)


def attach_position_markers(base, positions, rgb, alpha, radius):
    if base is None:
        return
    for pos in positions:
        mgm.gen_sphere(pos=np.asarray(pos),
                       radius=radius,
                       rgb=np.asarray(rgb),
                       alpha=alpha).attach_to(base)


def attach_grasp_previews(base, model_dir, spec):
    if base is None:
        return
    obj_pos, obj_rotmat = spec["pick_pose"]
    grasp_info_list = grasping.load_grasp_info_list(model_dir, spec)
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
        gl_jaw_center_pos = obj_pos + obj_rotmat @ jaw_center_pos
        gl_jaw_center_rotmat = obj_rotmat @ jaw_center_rotmat
        gripper = Dh50()
        gripper.grip_at_by_pose(jaw_center_pos=gl_jaw_center_pos,
                                jaw_center_rotmat=gl_jaw_center_rotmat,
                                jaw_width=jaw_width)
        gripper.gen_meshmodel(rgb=spec["rgba"][:3],
                              alpha=.24,
                              toggle_tcp_frame=True).attach_to(base)


def attach_grasp_debug_previews(base, model_dir, spec, grasp_indices):
    if base is None:
        return
    obj_pos, obj_rotmat = spec["pick_pose"]
    grasp_info_list = grasping.load_grasp_info_list(model_dir, spec)
    for grasp_index in grasp_indices:
        grasp_info = grasp_info_list[grasp_index]
        jaw_width, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
        gl_jaw_center_pos = obj_pos + obj_rotmat @ jaw_center_pos
        gl_jaw_center_rotmat = obj_rotmat @ jaw_center_rotmat
        mgm.gen_sphere(pos=gl_jaw_center_pos,
                       radius=.022,
                       rgb=np.array([1.0, .08, .04]),
                       alpha=.82).attach_to(base)
        gripper = Dh50()
        gripper.grip_at_by_pose(jaw_center_pos=gl_jaw_center_pos,
                                jaw_center_rotmat=gl_jaw_center_rotmat,
                                jaw_width=jaw_width)
        gripper.gen_meshmodel(rgb=np.array([1.0, .08, .04]),
                              alpha=.52,
                              toggle_tcp_frame=True).attach_to(base)


def attach_place_object_previews(base, cfg, spec, alpha=.18):
    if base is None:
        return
    rgba = np.asarray(spec["rgba"], dtype=float).copy()
    rgba[3] = alpha
    for place_pose in spec["place_poses"]:
        make_collision_model(spec["mesh"],
                             pos=place_pose[0],
                             rotmat=place_pose[1],
                             rgba=rgba,
                             attach_to=base,
                             mesh_dir=cfgutils.model_dir(cfg))


def build_box_stack(base, cfg, stack_spec):
    part_list = []
    center = np.asarray(stack_spec["center"], dtype=float)
    rotmat = _rotmat_from_scene_spec(stack_spec)
    mesh_prefix = stack_spec.get("mesh_prefix", "")
    rgba = np.asarray(cfg["scene"]["box_rgba"], dtype=float)
    for suffix, offset in stack_spec["part_offsets"].items():
        part_list.append(make_collision_model(f"{mesh_prefix}{suffix}.STL",
                                              pos=center + np.asarray(offset, dtype=float),
                                              rotmat=rotmat,
                                              rgba=rgba,
                                              ex_radius=0.0,
                                              attach_to=base,
                                              mesh_dir=cfgutils.mesh_dir(cfg)))
    return part_list


def build_scene(base, cfg, object_specs=None, attach_visuals=True):
    if object_specs is None:
        object_specs = cfgutils.object_specs(cfg)
    static_obstacles = []
    for spec in cfg["scene"].get("static_obstacles", []):
        static_obstacles.append(make_collision_model(spec["mesh"],
                                                     pos=np.asarray(spec["pos"], dtype=float),
                                                     rotmat=_rotmat_from_scene_spec(spec),
                                                     rgba=np.asarray(spec["rgba"], dtype=float),
                                                     attach_to=base if attach_visuals else None,
                                                     mesh_dir=cfgutils.mesh_dir(cfg)))
    for spec in cfg["scene"].get("display_only", []):
        make_collision_model(spec["mesh"],
                             pos=np.asarray(spec["pos"], dtype=float),
                             rotmat=_rotmat_from_scene_spec(spec),
                             rgba=np.asarray(spec["rgba"], dtype=float),
                             attach_to=base if attach_visuals else None,
                             mesh_dir=cfgutils.mesh_dir(cfg))
    for stack_spec in cfg["scene"].get("box_stacks", []):
        static_obstacles += build_box_stack(base if attach_visuals else None, cfg, stack_spec)

    payload_dict = {}
    for object_name, spec in object_specs.items():
        payload_dict[object_name] = make_collision_model(spec["mesh"],
                                                         pos=spec["pick_pose"][0],
                                                         rotmat=spec["pick_pose"][1],
                                                         rgba=spec["rgba"],
                                                         attach_to=base if attach_visuals else None,
                                                         mesh_dir=cfgutils.model_dir(cfg))
        if attach_visuals:
            attach_position_markers(base,
                                    spec["grasp_positions"],
                                    rgb=spec["rgba"][:3],
                                    alpha=.42,
                                    radius=.012)
            attach_position_markers(base,
                                    [pose[0] for pose in spec["place_poses"]],
                                    rgb=spec["rgba"][:3],
                                    alpha=.16,
                                    radius=.018)
            attach_place_object_previews(base, cfg, spec, alpha=.16)
            attach_grasp_previews(base, cfgutils.model_dir(cfg), spec)
    return static_obstacles, payload_dict


def make_planning_obstacle_list(robot, obstacle_list):
    planning_obstacle_list = list(obstacle_list) if obstacle_list is not None else []
    for frame_cmodel in robot.frame_collision_models:
        if all(frame_cmodel is not obstacle for obstacle in planning_obstacle_list):
            planning_obstacle_list.append(frame_cmodel)
    return planning_obstacle_list

