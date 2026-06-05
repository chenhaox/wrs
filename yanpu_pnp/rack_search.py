import argparse
import itertools
import traceback
import os
import yaml

import numpy as np

from yanpu_pnp import config as cfgutils
from yanpu_pnp import ik_backends
from yanpu_pnp import ik
from yanpu_pnp import planner
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene
from yanpu_pnp import search_config as search_cfgutils


PLAN_SCORE_METRIC_DESCRIPTION = "100 / sum(planned path point counts per arm); higher is better"
IK_SCORE_METRIC_DESCRIPTION = "100 / joint-space L2 delta cost (IK-only fallback); higher is better"


def _rounded_key_value(value):
    return round(float(value), 10)


def mount_group_key(arm_distance, left_arm_mount_euler_deg):
    return (
        _rounded_key_value(arm_distance),
        tuple(_rounded_key_value(value) for value in left_arm_mount_euler_deg),
    )


def candidate_mount_group_key(candidate):
    return mount_group_key(candidate["arm_distance"], candidate["left_arm_mount_euler_deg"])


def _blacklist_spec(search_cfg):
    spec = search_cfg.get("mount_group_blacklist", {})
    if spec is None:
        spec = {}
    if isinstance(spec, str):
        spec = {"enabled": True, "path": spec}
    return {
        "enabled": bool(spec.get("enabled", False)),
        "path": spec.get("path", "yanpu_pnp/config/rack_mount_blacklist.yaml"),
    }


def load_mount_group_blacklist(search_cfg):
    spec = _blacklist_spec(search_cfg)
    if not spec["enabled"]:
        return {
            "enabled": False,
            "path": None,
            "entries": [],
            "keys": set(),
        }
    path = cfgutils.resolve_path(spec["path"])
    if not os.path.exists(path):
        return {
            "enabled": True,
            "path": path,
            "entries": [],
            "keys": set(),
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("mount_groups", [])
    keys = {
        mount_group_key(entry["arm_distance"], entry["left_arm_mount_euler_deg"])
        for entry in entries
    }
    return {
        "enabled": True,
        "path": path,
        "entries": entries,
        "keys": keys,
    }


def raw_mount_group_samples(search_cfg):
    grid = search_cfg["grid"]
    arm_eulers = search_cfgutils.left_arm_mount_euler_sample_list(search_cfg)
    groups = []
    for group_index, (arm_distance, left_euler) in enumerate(
            itertools.product(search_cfgutils.sample_values(grid["arm_distance"]), arm_eulers),
            start=1):
        groups.append({
            "mount_group_index": int(group_index),
            "mount_group_key": mount_group_key(arm_distance, left_euler),
            "arm_distance": float(arm_distance),
            "left_arm_mount_euler_deg": list(left_euler),
        })
    return groups


def mount_group_samples(search_cfg, include_blacklisted=False):
    groups = raw_mount_group_samples(search_cfg)
    if include_blacklisted:
        return groups
    blacklist = load_mount_group_blacklist(search_cfg)
    if not blacklist["keys"]:
        return groups
    return [
        group for group in groups
        if group["mount_group_key"] not in blacklist["keys"]
    ]


def grid_counts(search_cfg):
    grid = search_cfg["grid"]
    base_pos_spec = search_cfgutils.base_pos_samples(search_cfg)
    yaw_values = search_cfgutils.sample_values(grid["yaw_deg"])
    mount_groups = mount_group_samples(search_cfg)
    raw_mount_groups = raw_mount_group_samples(search_cfg)
    blacklist = load_mount_group_blacklist(search_cfg)
    base_pos_count = len(base_pos_spec["x"]) * len(base_pos_spec["y"]) * len(base_pos_spec["z"])
    placement_count_per_group = base_pos_count * len(yaw_values)
    return {
        "base_pos_count": int(base_pos_count),
        "yaw_count": int(len(yaw_values)),
        "raw_mount_group_count": int(len(raw_mount_groups)),
        "mount_group_count": int(len(mount_groups)),
        "blacklisted_mount_group_count": int(len(raw_mount_groups) - len(mount_groups)),
        "blacklist_enabled": bool(blacklist["enabled"]),
        "blacklist_path": blacklist["path"],
        "placement_count_per_mount_group": int(placement_count_per_group),
        "candidate_count": int(len(mount_groups) * placement_count_per_group),
    }


def _make_candidate(base_pos,
                    yaw_deg,
                    arm_distance,
                    left_euler,
                    mount_group=None,
                    placement_index=None,
                    placement_count=None):
    candidate = {
        "rack_base_pos": list(base_pos),
        "rack_yaw_deg": float(yaw_deg),
        "arm_distance": float(arm_distance),
        "left_arm_mount_euler_deg": list(left_euler),
    }
    if mount_group is not None:
        candidate["mount_group"] = {
            "index": int(mount_group["mount_group_index"]),
            "key": list(mount_group["mount_group_key"]),
            "arm_distance": float(mount_group["arm_distance"]),
            "left_arm_mount_euler_deg": list(mount_group["left_arm_mount_euler_deg"]),
            "placement_index": None if placement_index is None else int(placement_index),
            "placement_count": None if placement_count is None else int(placement_count),
        }
    return candidate


def iter_search_candidates(search_cfg, max_candidates=None, group_by_mount=False, include_mount_group=False):
    grid = search_cfg["grid"]
    base_pos_spec = search_cfgutils.base_pos_samples(search_cfg)
    base_positions = itertools.product(base_pos_spec["x"], base_pos_spec["y"], base_pos_spec["z"])
    yaw_values = search_cfgutils.sample_values(grid["yaw_deg"])
    count = 0
    if group_by_mount:
        base_positions = list(base_positions)
        placement_count = len(base_positions) * len(yaw_values)
        for mount_group in mount_group_samples(search_cfg):
            placement_index = 0
            for base_pos, yaw_deg in itertools.product(base_positions, yaw_values):
                placement_index += 1
                yield _make_candidate(base_pos,
                                      yaw_deg,
                                      mount_group["arm_distance"],
                                      mount_group["left_arm_mount_euler_deg"],
                                      mount_group=mount_group if include_mount_group else None,
                                      placement_index=placement_index,
                                      placement_count=placement_count)
                count += 1
                if max_candidates is not None and count >= max_candidates:
                    return
    else:
        for base_pos, yaw_deg, mount_group in itertools.product(
                base_positions,
                yaw_values,
                mount_group_samples(search_cfg)):
            yield _make_candidate(base_pos,
                                  yaw_deg,
                                  mount_group["arm_distance"],
                                  mount_group["left_arm_mount_euler_deg"],
                                  mount_group=mount_group if include_mount_group else None)
            count += 1
            if max_candidates is not None and count >= max_candidates:
                return


def resolve_ik_backend(base_cfg, search_cfg=None, override=None):
    return ik_backends.resolve_backend(base_cfg=base_cfg, search_cfg=search_cfg, override=override)


def _joint_delta_cost(task_list):
    cost = 0.0
    for task in task_list:
        cost += np.linalg.norm(task.lift_conf - task.pick_conf)
        cost += np.linalg.norm(task.pre_place_conf - task.lift_conf)
        cost += np.linalg.norm(task.place_conf - task.pre_place_conf)
    return float(cost)


def _inverse_length_score(length):
    if length is None:
        return float("-inf")
    length = float(length)
    if length <= 0:
        return float("inf")
    return float(100.0 / length)


def _task_summary(task):
    return {
        "arm_name": task.arm_name,
        "object_name": task.object_name,
        "grasp_index": int(task.grasp_index),
        "pick_pose_index": int(task.pick_pose_index),
        "pick_symmetry_deg": float(np.degrees(task.symmetry_angle)),
        "place_pose_index": int(task.place_pose_index),
        "place_symmetry_deg": float(np.degrees(task.place_symmetry_angle)),
        "jaw_width": float(task.jaw_width),
        "joint_delta": {
            "pick_to_lift": float(np.linalg.norm(task.lift_conf - task.pick_conf)),
            "lift_to_pre_place": float(np.linalg.norm(task.pre_place_conf - task.lift_conf)),
            "pre_place_to_place": float(np.linalg.norm(task.place_conf - task.pre_place_conf)),
        },
    }


def evaluate_candidate(base_cfg, candidate, plan_motion=False, task_specs=None, ik_backend=None):
    cfg = cfgutils.apply_search_candidate(base_cfg, candidate)
    backend = resolve_ik_backend(cfg, override=ik_backend)
    ik_backends.set_backend(cfg, backend)
    object_specs = cfgutils.object_specs(cfg)
    if task_specs is None:
        task_specs = cfgutils.task_specs(cfg)
    robot = robot_factory.build_robot(cfg, enable_cc=True, ik_backend=backend)
    obstacle_list, _ = scene.build_scene(None, cfg, object_specs, attach_visuals=False)
    try:
        task_list = ik.build_pick_place_tasks(robot,
                                             cfg,
                                             obstacle_list,
                                             object_specs=object_specs,
                                             task_specs=task_specs)
        frame_count = None
        plan_info = {}
        path_point_count = None
        joint_delta_cost = _joint_delta_cost(task_list)
        if plan_motion:
            frame_list, plan_info = planner.build_frame_list_with_plan_info(robot, cfg, obstacle_list, task_list)
            frame_count = len(frame_list)
            path_point_count = int(plan_info["planned_arm_path_point_count_sum"])
            score = _inverse_length_score(path_point_count)
            score_metric = PLAN_SCORE_METRIC_DESCRIPTION
        else:
            score = _inverse_length_score(joint_delta_cost)
            score_metric = IK_SCORE_METRIC_DESCRIPTION
        return {
            "ok": True,
            "candidate": candidate,
            "score": score,
            "score_metric": score_metric,
            "joint_delta_cost": joint_delta_cost,
            "path_point_count": path_point_count,
            "plan_info": plan_info,
            "frame_count": frame_count,
            "reason": "ok",
            "ik_backend": backend,
            "tasks": [_task_summary(task) for task in task_list],
        }
    except Exception as error:
        return {
            "ok": False,
            "candidate": candidate,
            "score": float("-inf"),
            "score_metric": PLAN_SCORE_METRIC_DESCRIPTION if plan_motion else IK_SCORE_METRIC_DESCRIPTION,
            "frame_count": None,
            "reason": str(error),
            "ik_backend": backend,
            "traceback": traceback.format_exc(limit=3),
        }


def search(cfg, search_cfg, max_candidates=None, plan_motion=None, ik_backend=None):
    if plan_motion is None:
        plan_motion = bool(search_cfg.get("plan_motion", False))
    backend = resolve_ik_backend(cfg, search_cfg=search_cfg, override=ik_backend)
    task_specs = search_cfgutils.apply_task_overrides(cfgutils.task_specs(cfg), search_cfg)
    print(f"IK backend: {backend}")
    best_result = None
    ok_count = 0
    fail_count = 0
    for idx, candidate in enumerate(iter_search_candidates(search_cfg, max_candidates=max_candidates), start=1):
        print(f"Search candidate #{idx}: {candidate}")
        result = evaluate_candidate(cfg,
                                    candidate,
                                    plan_motion=plan_motion,
                                    task_specs=task_specs,
                                    ik_backend=backend)
        if result["ok"]:
            ok_count += 1
            print(f"  ok score={result['score']:.3f}, "
                  f"path_point_count={result.get('path_point_count')}, "
                  f"frame_count={result['frame_count']}")
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
                print("  new best")
        else:
            fail_count += 1
            print(f"  failed: {result['reason']}")
    print(f"Search finished: ok={ok_count}, failed={fail_count}.")
    if best_result is not None:
        print("Best candidate:", best_result["candidate"])
        print(f"Best score: {best_result['score']:.3f}")
    return best_result


def main():
    parser = argparse.ArgumentParser(description="Search rack parameters for dual UR7E + DH50 pick-and-place.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--plan-motion", dest="plan_motion", action="store_true", default=None,
                        help="Also run multi-arm RRT for each IK-feasible candidate.")
    parser.add_argument("--no-plan-motion", dest="plan_motion", action="store_false")
    parser.add_argument("--ik-backend", default=None,
                        help="Override IK backend for this run, e.g. tracik or ikfast.")
    args = parser.parse_args()
    cfg = cfgutils.load_config(args.config)
    search_cfg = search_cfgutils.load_search_config(args.search_config)
    search(cfg,
           search_cfg,
           max_candidates=args.max_candidates,
           plan_motion=args.plan_motion,
           ik_backend=args.ik_backend)


if __name__ == "__main__":
    main()
