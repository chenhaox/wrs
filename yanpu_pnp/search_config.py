import os
import itertools
import copy

import yaml

from yanpu_pnp.config import PROJECT_ROOT


DEFAULT_SEARCH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "yanpu_pnp", "config", "rack_search.yaml")


def load_search_config(path=DEFAULT_SEARCH_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


_TASK_SYMMETRY_KEYS = (
    "pick_symmetry_angle_count",
    "place_symmetry_angle_count",
    "pick_symmetry_angles",
    "place_symmetry_angles",
    "pick_symmetry_angles_deg",
    "place_symmetry_angles_deg",
)


def _copy_value(value):
    return copy.deepcopy(value)


def task_overrides(search_cfg):
    return dict(search_cfg.get("task_overrides", {}))


def _merged_task_override_for_arm(overrides, arm_name):
    merged = {}
    for key in _TASK_SYMMETRY_KEYS:
        if key in overrides:
            merged[key] = _copy_value(overrides[key])
    arm_overrides = overrides.get(arm_name, {})
    if arm_overrides is None:
        arm_overrides = {}
    for key in _TASK_SYMMETRY_KEYS:
        if key in arm_overrides:
            merged[key] = _copy_value(arm_overrides[key])
    return merged


def _clear_prefix_symmetry(task_spec, prefix):
    task_spec.pop(f"{prefix}_symmetry_angle_count", None)
    task_spec.pop(f"{prefix}_symmetry_angles", None)
    task_spec.pop(f"{prefix}_symmetry_angles_deg", None)


def apply_task_overrides(task_specs, search_cfg):
    overrides = task_overrides(search_cfg)
    if not overrides:
        return task_specs
    for arm_name, task_spec in task_specs.items():
        arm_overrides = _merged_task_override_for_arm(overrides, arm_name)
        for prefix in ("pick", "place"):
            prefix_keys = [key for key in arm_overrides if key.startswith(f"{prefix}_symmetry_")]
            if prefix_keys:
                _clear_prefix_symmetry(task_spec, prefix)
                for key in prefix_keys:
                    task_spec[key] = _copy_value(arm_overrides[key])
    return task_specs


def sample_values(spec):
    if isinstance(spec, (list, tuple)):
        return [float(value) for value in spec]
    if "values" in spec:
        return [float(value) for value in spec["values"]]
    start = float(spec.get("min", spec.get("start")))
    stop = float(spec.get("max", spec.get("stop")))
    step = float(spec["step"])
    if step <= 0:
        raise ValueError(f"step must be positive: {spec}")
    values = []
    value = start
    eps = abs(step) * 1e-9
    while value <= stop + eps:
        values.append(round(value, 10))
        value += step
    return values


def base_pos_samples(search_cfg):
    base_pos = search_cfg["grid"]["base_pos"]
    return {
        "x": sample_values(base_pos["x"]),
        "y": sample_values(base_pos["y"]),
        "z": sample_values(base_pos["z"]),
    }


def left_arm_mount_euler_samples(search_cfg):
    euler = search_cfg["grid"]["left_arm_mount_euler_deg"]
    return {
        "roll": sample_values(euler["roll"]),
        "pitch": sample_values(euler["pitch"]),
        "yaw": sample_values(euler["yaw"]),
    }


def left_arm_mount_euler_sample_list(search_cfg, collapse_zero_roll_yaw=True):
    spec = left_arm_mount_euler_samples(search_cfg)
    euler_list = []
    for roll, pitch in itertools.product(spec["roll"], spec["pitch"]):
        yaw_values = spec["yaw"]
        if collapse_zero_roll_yaw and abs(roll) < 1e-9:
            yaw_values = [yaw_values[0]]
        for yaw in yaw_values:
            euler_list.append([float(roll), float(pitch), float(yaw)])
    return euler_list
