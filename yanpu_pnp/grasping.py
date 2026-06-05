import os
import pickle

import numpy as np

import wrs.basis.robot_math as rm


def stable_seed(*values):
    seed = 17
    for value in values:
        for char in str(value):
            seed = (seed * 31 + ord(char)) % (2 ** 32)
    return seed


def load_grasp_info_list(model_dir, spec):
    pickle_path = os.path.join(model_dir, spec["grasp_pickle"])
    with open(pickle_path, "rb") as f:
        grasp_info_dict = pickle.load(f)
    return grasp_info_dict[spec["grasp_key"]]


def grasp_tcp_pose(object_pose, grasp_info):
    obj_pos, obj_rotmat = object_pose
    _, jaw_center_pos, jaw_center_rotmat, _, _ = grasp_info
    return obj_pos + obj_rotmat @ jaw_center_pos, obj_rotmat @ jaw_center_rotmat


def candidate_grasp_indices(task_spec, grasp_info_list):
    if "grasp_indices" in task_spec:
        return list(task_spec["grasp_indices"])
    return list(range(len(grasp_info_list)))


def candidate_pick_poses(task_spec, object_spec):
    if "pick_pose_candidates" in task_spec:
        return list(task_spec["pick_pose_candidates"])
    if "pick_pose_candidates" in object_spec:
        return list(object_spec["pick_pose_candidates"])
    return [object_spec["pick_pose"]]


def candidate_place_pose_items(task_spec, object_spec):
    if "place_pose_candidates" in task_spec:
        return list(enumerate(task_spec["place_pose_candidates"]))
    place_poses = object_spec["place_poses"]
    if "place_indices" in task_spec:
        return [(int(index), place_poses[int(index)]) for index in task_spec["place_indices"]]
    if "place_index" in task_spec:
        index = int(task_spec["place_index"])
        return [(index, place_poses[index])]
    return list(enumerate(place_poses))


def _angles_from_count(angle_count):
    if angle_count <= 1:
        return [0.0]
    return [float(angle) for angle in np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)]


def _candidate_symmetry_angles(task_spec, object_spec, prefix):
    rad_key = f"{prefix}_symmetry_angles"
    deg_key = f"{prefix}_symmetry_angles_deg"
    count_key = f"{prefix}_symmetry_angle_count"
    for spec in (task_spec, object_spec):
        if rad_key in spec:
            return [float(angle) for angle in spec[rad_key]]
        if deg_key in spec:
            return [float(np.radians(angle)) for angle in spec[deg_key]]
        if count_key in spec:
            return _angles_from_count(int(spec[count_key]))
    if "symmetry_angles" in task_spec:
        return [float(angle) for angle in task_spec["symmetry_angles"]]
    if "symmetry_angles" in object_spec:
        return [float(angle) for angle in object_spec["symmetry_angles"]]
    angle_count = int(task_spec.get("rotational_symmetry_angle_count",
                                    object_spec.get("rotational_symmetry_angle_count", 1)))
    return _angles_from_count(angle_count)


def candidate_pick_symmetry_angles(task_spec, object_spec):
    return _candidate_symmetry_angles(task_spec, object_spec, "pick")


def candidate_place_symmetry_angles(task_spec, object_spec):
    return _candidate_symmetry_angles(task_spec, object_spec, "place")


def candidate_symmetry_angles(task_spec, object_spec):
    return candidate_pick_symmetry_angles(task_spec, object_spec)


def apply_pose_symmetry(pose, object_spec, angle):
    pos, rotmat = pose
    pos = np.asarray(pos, dtype=float)
    rotmat = np.asarray(rotmat, dtype=float)
    if abs(angle) <= 1e-9:
        return pos.copy(), rotmat.copy()
    axis = np.asarray(object_spec.get("rotational_symmetry_axis", rm.const.z_ax), dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-9:
        return pos.copy(), rotmat.copy()
    return pos.copy(), rotmat @ rm.rotmat_from_axangle(axis / axis_norm, angle)
