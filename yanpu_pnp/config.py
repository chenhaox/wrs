import copy
import os

import numpy as np
import yaml

import wrs.basis.robot_math as rm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "yanpu_pnp", "config", "default.yaml")


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def clone_config(cfg):
    return copy.deepcopy(cfg)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def array(value, dtype=float):
    return np.asarray(value, dtype=dtype)


def deg_array(value):
    return np.radians(array(value, dtype=float))


def rotmat_from_spec(spec):
    if spec is None or spec.get("identity", False):
        return np.eye(3)
    if "euler_deg" in spec:
        return rm.rotmat_from_euler(*deg_array(spec["euler_deg"]))
    if "euler_rad" in spec:
        return rm.rotmat_from_euler(*array(spec["euler_rad"]))
    if "axangle_deg" in spec:
        axis = array(spec["axangle_deg"]["axis"])
        angle = np.radians(float(spec["axangle_deg"]["angle"]))
        return rm.rotmat_from_axangle(axis / np.linalg.norm(axis), angle)
    if "matrix" in spec:
        return array(spec["matrix"]).reshape(3, 3)
    raise ValueError(f"Unsupported rotation spec: {spec}")


def rotmat_from_euler_deg(euler_deg):
    return rm.rotmat_from_euler(*deg_array(euler_deg))


def rack_rotmat(cfg):
    return rm.rotmat_from_axangle(rm.const.z_ax, np.radians(float(cfg["rack"]["yaw_deg"])))


def symmetric_arm_mount_rotmats(cfg):
    left_euler = array(cfg["rack"]["left_arm_mount_euler_deg"])
    lft = rotmat_from_euler_deg(left_euler)
    rgt_euler = np.array([-left_euler[0], left_euler[1], -left_euler[2]])
    rgt = rotmat_from_euler_deg(rgt_euler) @ rm.rotmat_from_axangle(rm.const.z_ax, np.pi)
    return lft, rgt


def rack_kwargs(cfg):
    rack = cfg["rack"]
    vertical = rack["vertical_frame"]
    horizontal = rack["horizontal_frame"]
    lft_rotmat, rgt_rotmat = symmetric_arm_mount_rotmats(cfg)
    arm_distance = float(rack["arm_distance"])
    return {
        "body_root_pos": array(rack["base_pos"]),
        "body_root_rotmat": rack_rotmat(cfg),
        "vertical_frame_height": float(vertical["height"]),
        "vertical_frame_xy": array([vertical["x_length"], vertical["y_length"]]),
        "vertical_frame_x_length": float(vertical["x_length"]),
        "vertical_frame_y_length": float(vertical["y_length"]),
        "horizontal_frame_thickness": float(horizontal["thickness"]),
        "horizontal_frame_x_length": float(horizontal["x_length"]),
        "horizontal_frame_y_length": float(horizontal["y_length"]),
        "vertical_frame_rgb": array(vertical["rgb"]),
        "horizontal_frame_rgb": array(horizontal["rgb"]),
        "vertical_frame_alpha": float(vertical["alpha"]),
        "horizontal_frame_alpha": float(horizontal["alpha"]),
        # DualUR7EDH50 expects a per-side offset; YAML exposes the absolute distance.
        "arm_y_offset": arm_distance * 0.5,
        "arm_y_offset_reference_frame_y_length": 0.0,
        "lft_arm_loc_rotmat": lft_rotmat,
        "rgt_arm_loc_rotmat": rgt_rotmat,
        "lft_home_conf": array(rack["home_conf"]["lft_arm"]),
        "rgt_home_conf": array(rack["home_conf"]["rgt_arm"]),
    }


def object_specs(cfg):
    specs = {}
    for name, obj in cfg["objects"].items():
        pick_rotmat = rotmat_from_spec(obj.get("pick_rotation"))
        place_rotmat = rotmat_from_spec(obj.get("place_rotation"))
        pick_pose_candidates = [(array(pos), pick_rotmat.copy()) for pos in obj["pick_pose_candidates"]]
        place_poses = [(array(pos), place_rotmat.copy()) for pos in obj["place_positions"]]
        spec = {
            "mesh": obj["mesh"],
            "rgba": array(obj["rgba"]),
            "grasp_pickle": obj["grasp_pickle"],
            "grasp_key": obj["grasp_key"],
            "pick_pose": pick_pose_candidates[0],
            "pick_pose_candidates": pick_pose_candidates,
            "grasp_positions": [array(pos) for pos in obj.get("marker_positions", obj["pick_pose_candidates"])],
            "place_poses": place_poses,
        }
        if "rotational_symmetry" in obj:
            symmetry = obj["rotational_symmetry"]
            spec["rotational_symmetry_axis"] = array(symmetry.get("axis", [0, 0, 1]))
            spec["rotational_symmetry_angle_count"] = int(symmetry.get("angle_count", 1))
        specs[name] = spec
    return specs


def task_specs(cfg):
    return copy.deepcopy(cfg["tasks"])


def mesh_dir(cfg):
    return resolve_path(cfg["paths"]["mesh_dir"])


def model_dir(cfg):
    return resolve_path(cfg["paths"]["model_dir"])


def camera_kwargs(cfg):
    camera = cfg.get("camera", {})
    return {
        "cam_pos": camera.get("cam_pos", [1.8, 1.6, 1.35]),
        "lookat_pos": camera.get("lookat_pos", [0.35, 0.0, 0.95]),
    }


def apply_search_candidate(cfg, candidate):
    new_cfg = clone_config(cfg)
    rack = new_cfg["rack"]
    if "rack_base_pos" in candidate:
        rack["base_pos"] = list(candidate["rack_base_pos"])
    if "rack_yaw_deg" in candidate:
        rack["yaw_deg"] = float(candidate["rack_yaw_deg"])
    if "vertical_frame_height" in candidate:
        rack["vertical_frame"]["height"] = float(candidate["vertical_frame_height"])
    if "horizontal_frame_y_length" in candidate:
        rack["horizontal_frame"]["y_length"] = float(candidate["horizontal_frame_y_length"])
    if "arm_distance" in candidate:
        rack["arm_distance"] = float(candidate["arm_distance"])
    if "left_arm_mount_euler_deg" in candidate:
        rack["left_arm_mount_euler_deg"] = list(candidate["left_arm_mount_euler_deg"])
    return new_cfg
