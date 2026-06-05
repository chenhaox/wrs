import os
from datetime import datetime

import numpy as np
import yaml

from yanpu_pnp import config as cfgutils


DEFAULT_RESULT_PATH = os.path.join(cfgutils.PROJECT_ROOT,
                                   "yanpu_pnp",
                                   "results",
                                   "last_rack_search_result.yaml")
DEFAULT_ALL_RESULTS_PATH = os.path.join(cfgutils.PROJECT_ROOT,
                                        "yanpu_pnp",
                                        "results",
                                        "all_rack_search_results.yaml")


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def resolve_result_path(path):
    if os.path.isabs(path):
        return path
    return cfgutils.resolve_path(path)


def build_result_record(base_cfg,
                        search_cfg,
                        args,
                        task_specs,
                        result,
                        candidate_index,
                        completed_candidates,
                        elapsed,
                        worker_elapsed=None):
    candidate = result["candidate"]
    applied_cfg = cfgutils.apply_search_candidate(base_cfg, candidate)
    record = {
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "base_config": base_cfg.get("_config_path", cfgutils.DEFAULT_CONFIG_PATH),
            "search_config": search_cfg.get("_config_path"),
        },
        "base_config_snapshot": base_cfg,
        "search_config_snapshot": search_cfg,
        "search": {
            "candidate_index": int(candidate_index),
            "completed_candidates": None if completed_candidates is None else int(completed_candidates),
            "score": float(result["score"]),
            "score_metric": result.get("score_metric"),
            "joint_delta_cost": result.get("joint_delta_cost"),
            "path_point_count": result.get("path_point_count"),
            "frame_count": result["frame_count"],
            "elapsed": float(elapsed),
            "worker_elapsed": None if worker_elapsed is None else float(worker_elapsed),
            "plan_motion": bool(args.plan_motion),
            "ik_backend": getattr(args, "ik_backend", result.get("ik_backend")),
            "workers": int(args.workers),
            "group_by_mount": getattr(args, "group_by_mount", None),
            "start_index": int(args.start_index),
            "max_candidates": args.max_candidates,
        },
        "candidate": candidate,
        "one_pickplace": pickplace_selection_from_args(args),
        "task_overrides": search_cfg.get("task_overrides", {}),
        "task_specs": task_specs,
        "tasks": result.get("tasks", []),
        "applied_config": applied_cfg,
    }
    return to_builtin(record)


def pickplace_selection_from_args(args):
    return {
        "lft_arm": {
            "pick_index": getattr(args, "lft_pick_index", None),
            "place_index": getattr(args, "lft_place_index", None),
            "grasp_index": getattr(args, "lft_grasp_index", None),
        },
        "rgt_arm": {
            "pick_index": getattr(args, "rgt_pick_index", None),
            "place_index": getattr(args, "rgt_place_index", None),
            "grasp_index": getattr(args, "rgt_grasp_index", None),
        },
    }


def _candidate_mount_key(candidate):
    return (
        float(candidate["arm_distance"]),
        tuple(float(value) for value in candidate["left_arm_mount_euler_deg"]),
    )


def _placement_summary(result):
    candidate = result["candidate"]
    return {
        "candidate_index": result["candidate_index"],
        "rack_base_pos": candidate["rack_base_pos"],
        "rack_yaw_deg": candidate["rack_yaw_deg"],
        "score": result["score"],
        "score_metric": result.get("score_metric"),
        "path_point_count": result.get("path_point_count"),
        "joint_delta_cost": result.get("joint_delta_cost"),
        "frame_count": result.get("frame_count"),
        "worker_elapsed": result.get("worker_elapsed"),
        "search_elapsed_when_found": result.get("search_elapsed_when_found"),
    }


def _best_summary(result):
    candidate = result["candidate"]
    return {
        "candidate_index": result["candidate_index"],
        "arm_distance": candidate["arm_distance"],
        "left_arm_mount_euler_deg": candidate["left_arm_mount_euler_deg"],
        "rack_base_pos": candidate["rack_base_pos"],
        "rack_yaw_deg": candidate["rack_yaw_deg"],
        "score": result["score"],
        "score_metric": result.get("score_metric"),
        "path_point_count": result.get("path_point_count"),
        "joint_delta_cost": result.get("joint_delta_cost"),
        "frame_count": result.get("frame_count"),
    }


def build_mount_results(results):
    group_map = {}
    for result in results:
        candidate = result["candidate"]
        key = _candidate_mount_key(candidate)
        group = group_map.get(key)
        if group is None:
            mount_group = candidate.get("mount_group", {})
            group = {
                "mount_group_index": mount_group.get("index"),
                "arm_distance": candidate["arm_distance"],
                "left_arm_mount_euler_deg": candidate["left_arm_mount_euler_deg"],
                "feasible_count": 0,
                "best": None,
                "feasible_placements": [],
            }
            group_map[key] = group
        placement = _placement_summary(result)
        group["feasible_count"] += 1
        group["feasible_placements"].append(placement)
        if group["best"] is None or placement["score"] > group["best"]["score"]:
            group["best"] = dict(placement)
    groups = list(group_map.values())
    for group in groups:
        group["feasible_placements"].sort(key=lambda item: (-item["score"], item["candidate_index"]))
    groups.sort(key=lambda item: (
        item["mount_group_index"] is None,
        -item["best"]["score"] if item["best"] is not None else float("inf"),
        item["arm_distance"],
        item["left_arm_mount_euler_deg"],
    ))
    return groups


def write_result(path,
                 base_cfg,
                 search_cfg,
                 args,
                 task_specs,
                 result,
                 candidate_index,
                 completed_candidates,
                 elapsed,
                 worker_elapsed=None):
    result_path = resolve_result_path(path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    record = build_result_record(base_cfg,
                                 search_cfg,
                                 args,
                                 task_specs,
                                 result,
                                 candidate_index,
                                 completed_candidates,
                                 elapsed,
                                 worker_elapsed=worker_elapsed)
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    return result_path


def build_all_results_record(base_cfg,
                             search_cfg,
                             args,
                             task_specs,
                             results,
                             checked_count,
                             failed_count,
                             elapsed,
                             mount_summary=None):
    mount_results = build_mount_results(results)
    if mount_summary is None:
        mount_summary = {
            "total_mount_groups": None,
            "completed_mount_groups": None,
            "feasible_mount_groups": len(mount_results),
            "infeasible_mount_groups": None,
            "unfinished_mount_groups": None,
        }
    return to_builtin({
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "base_config": base_cfg.get("_config_path", cfgutils.DEFAULT_CONFIG_PATH),
            "search_config": search_cfg.get("_config_path"),
        },
        "base_config_snapshot": base_cfg,
        "search_config_snapshot": search_cfg,
        "search": {
            "checked_candidates": int(checked_count),
            "feasible_candidates": len(results),
            "total_mount_groups": mount_summary.get("total_mount_groups"),
            "completed_mount_groups": mount_summary.get("completed_mount_groups"),
            "feasible_mount_groups": mount_summary.get("feasible_mount_groups", len(mount_results)),
            "infeasible_mount_groups": mount_summary.get("infeasible_mount_groups"),
            "unfinished_mount_groups": mount_summary.get("unfinished_mount_groups"),
            "failed_candidates": int(failed_count),
            "elapsed": float(elapsed),
            "plan_motion": bool(args.plan_motion),
            "ik_backend": getattr(args, "ik_backend", None),
            "workers": int(args.workers),
            "worker_maxtasks_per_child": getattr(args, "worker_maxtasks_per_child", None),
            "group_by_mount": getattr(args, "group_by_mount", None),
            "start_index": int(args.start_index),
            "max_candidates": args.max_candidates,
        },
        "pickplace_selection": pickplace_selection_from_args(args),
        "task_overrides": search_cfg.get("task_overrides", {}),
        "task_specs": task_specs,
        "best_result": None if not results else _best_summary(results[0]),
        "mount_results": mount_results,
        "results": results,
    })


def write_all_results(path,
                      base_cfg,
                      search_cfg,
                      args,
                      task_specs,
                      results,
                      checked_count,
                      failed_count,
                      elapsed,
                      mount_summary=None):
    result_path = resolve_result_path(path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    record = build_all_results_record(base_cfg,
                                      search_cfg,
                                      args,
                                      task_specs,
                                      results,
                                      checked_count,
                                      failed_count,
                                      elapsed,
                                      mount_summary=mount_summary)
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    return result_path


def load_result(path=DEFAULT_RESULT_PATH):
    result_path = resolve_result_path(path)
    with open(result_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_from_result(record):
    if "results" in record:
        if not record["results"]:
            raise ValueError("No feasible result is stored in this all-results record.")
        record = record["results"][0] | {"source": record.get("source", {})}
    if "applied_config" in record:
        return record["applied_config"]
    if "base_config_snapshot" in record:
        return cfgutils.apply_search_candidate(record["base_config_snapshot"], record["candidate"])
    source = record.get("source", {})
    base_config_path = source.get("base_config", cfgutils.DEFAULT_CONFIG_PATH)
    base_cfg = cfgutils.load_config(cfgutils.resolve_path(base_config_path))
    return cfgutils.apply_search_candidate(base_cfg, record["candidate"])
