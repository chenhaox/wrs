import argparse
import contextlib
import copy
import gc
import io
import multiprocessing as mp
import time
import traceback

from trac_ik import TracIK as _TracIK

import numpy as np

from yanpu_pnp import config as cfgutils
from yanpu_pnp import console
from yanpu_pnp import grasping
from yanpu_pnp import ik
from yanpu_pnp import ik_backends
from yanpu_pnp import planner
from yanpu_pnp import rack_search
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene
from yanpu_pnp import search_config as search_cfgutils
from yanpu_pnp import search_result


_WORKER_BASE_CFG = None
_WORKER_TASK_SPECS = None
_WORKER_PLAN_MOTION = False
_WORKER_VERBOSE = False
_WORKER_IK_BACKEND = None
PLAN_SCORE_METRIC_DESCRIPTION = "100 / sum(planned path point counts per arm); higher is better"
IK_SCORE_METRIC_DESCRIPTION = "100 / joint-space L2 delta cost (IK-only fallback); higher is better"


class _NullStream:

    def write(self, _text):
        return 0

    def flush(self):
        return None


def _init_worker(base_cfg, task_specs, plan_motion, verbose, ik_backend):
    global _WORKER_BASE_CFG, _WORKER_TASK_SPECS, _WORKER_PLAN_MOTION, _WORKER_VERBOSE, _WORKER_IK_BACKEND
    _WORKER_BASE_CFG = base_cfg
    _WORKER_TASK_SPECS = task_specs
    _WORKER_PLAN_MOTION = plan_motion
    _WORKER_VERBOSE = verbose
    _WORKER_IK_BACKEND = ik_backend


def _evaluate_candidate_payload(payload):
    index, candidate = payload
    tic = time.perf_counter()
    stream = io.StringIO() if _WORKER_VERBOSE else _NullStream()
    try:
        with contextlib.redirect_stdout(stream):
            result = evaluate_coverage_candidate(_WORKER_BASE_CFG,
                                                 candidate,
                                                 _WORKER_TASK_SPECS,
                                                 plan_motion=_WORKER_PLAN_MOTION,
                                                 ik_backend=_WORKER_IK_BACKEND)
    except Exception as error:
        result = {
            "ok": False,
            "candidate": candidate,
            "score": float("-inf"),
            "score_metric": PLAN_SCORE_METRIC_DESCRIPTION
            if _WORKER_PLAN_MOTION else IK_SCORE_METRIC_DESCRIPTION,
            "frame_count": None,
            "reason": str(error),
        }
    finally:
        gc.collect()
    raw_log = stream.getvalue() if _WORKER_VERBOSE else ""
    return index, result, raw_log, time.perf_counter() - tic


def _select_index(seq, index, label):
    if index < 0 or index >= len(seq):
        raise IndexError(f"{label} index {index} out of range [0, {len(seq) - 1}]")
    return seq[index]


def make_task_specs(cfg,
                    object_specs,
                    lft_pick_index=None,
                    lft_place_index=None,
                    rgt_pick_index=None,
                    rgt_place_index=None,
                    lft_grasp_index=None,
                    rgt_grasp_index=None):
    task_specs = cfgutils.task_specs(cfg)
    selection = {
        "lft_arm": (lft_pick_index, lft_place_index, lft_grasp_index),
        "rgt_arm": (rgt_pick_index, rgt_place_index, rgt_grasp_index),
    }
    for arm_name, (pick_index, place_index, grasp_index) in selection.items():
        task_spec = task_specs[arm_name]
        object_name = task_spec["object_name"]
        object_spec = object_specs[object_name]
        if pick_index is not None:
            pick_pose = _select_index(object_spec["pick_pose_candidates"],
                                      pick_index,
                                      f"{arm_name} {object_name} pick_pose")
            task_spec["pick_pose_candidates"] = [pick_pose]
        if place_index is not None:
            _select_index(object_spec["place_poses"], place_index, f"{arm_name} {object_name} place_pose")
            task_spec["place_indices"] = [place_index]
            task_spec.pop("place_index", None)
        if grasp_index is not None:
            task_spec["grasp_indices"] = [int(grasp_index)]
    return task_specs


def iter_indexed_candidates(search_cfg, args):
    checked_count = 0
    for index, candidate in enumerate(
            rack_search.iter_search_candidates(search_cfg,
                                               group_by_mount=args.group_by_mount,
                                               include_mount_group=True),
            start=1):
        if index < args.start_index:
            continue
        if args.max_candidates is not None and checked_count >= args.max_candidates:
            break
        checked_count += 1
        yield index, candidate


def build_task_search_info(task_specs, object_specs):
    info = {}
    for arm_name, task_spec in task_specs.items():
        object_name = task_spec["object_name"]
        object_spec = object_specs[object_name]
        pick_pose_count = len(grasping.candidate_pick_poses(task_spec, object_spec))
        place_pose_count = len(grasping.candidate_place_pose_items(task_spec, object_spec))
        pick_symmetry_count = len(grasping.candidate_pick_symmetry_angles(task_spec, object_spec))
        place_symmetry_count = len(grasping.candidate_place_symmetry_angles(task_spec, object_spec))
        info[arm_name] = {
            "object_name": object_name,
            "pick_pose_count": int(pick_pose_count),
            "place_pose_count": int(place_pose_count),
            "pick_symmetry_count": int(pick_symmetry_count),
            "place_symmetry_count": int(place_symmetry_count),
            "pose_symmetry_tuple_count": int(pick_pose_count * place_pose_count *
                                             pick_symmetry_count * place_symmetry_count),
        }
    return search_result.to_builtin(info)


def _arm_option_maps(task_specs, object_specs):
    pick_options = {}
    place_options = {}
    for arm_name, task_spec in task_specs.items():
        object_spec = object_specs[task_spec["object_name"]]
        pick_options[arm_name] = list(enumerate(grasping.candidate_pick_poses(task_spec, object_spec)))
        place_options[arm_name] = grasping.candidate_place_pose_items(task_spec, object_spec)
    return pick_options, place_options


def _task_specs_for_arm_case(task_specs, object_specs, arm_name, pick_index, place_index):
    task_spec = copy.deepcopy(task_specs[arm_name])
    object_spec = object_specs[task_spec["object_name"]]
    candidate_pick_poses = grasping.candidate_pick_poses(task_specs[arm_name], object_spec)
    task_spec["pick_pose_candidates"] = [_select_index(candidate_pick_poses,
                                                       int(pick_index),
                                                       f"{arm_name} pick_pose")]
    task_spec["place_indices"] = [int(place_index)]
    task_spec.pop("place_index", None)
    return {arm_name: task_spec}


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


def _evaluate_case_on_context(robot, cfg, obstacle_list, object_specs, task_specs, candidate, plan_motion, ik_backend):
    try:
        task_list = ik.build_pick_place_tasks(robot,
                                             cfg,
                                             obstacle_list,
                                             object_specs=object_specs,
                                             task_specs=task_specs)
        joint_delta_cost = _joint_delta_cost(task_list)
        frame_count = None
        plan_info = {}
        path_point_count = None
        if plan_motion:
            frame_list, plan_info = planner.build_frame_list_with_plan_info(robot,
                                                                            cfg,
                                                                            obstacle_list,
                                                                            task_list)
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
            "ik_backend": ik_backend,
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
            "ik_backend": ik_backend,
            "traceback": traceback.format_exc(limit=3),
        }


def _annotate_arm_case_tasks(result, arm_name, pick_index, place_index):
    for task in result.get("tasks", []):
        if task["arm_name"] == arm_name:
            task["pick_pose_index"] = int(pick_index)
            task["place_pose_index"] = int(place_index)


def _case_record(result, arm_name, pick_index, place_index):
    return {
        "arm_name": arm_name,
        "pick_pose_index": int(pick_index),
        "place_pose_index": int(place_index),
        "score": float(result["score"]),
        "score_metric": result.get("score_metric"),
        "joint_delta_cost": float(result.get("joint_delta_cost", 0.0)),
        "path_point_count": result.get("path_point_count"),
        "plan_info": result.get("plan_info", {}),
        "frame_count": result["frame_count"],
        "tasks": result.get("tasks", []),
    }


def _select_representative_task(arm_result):
    if arm_result is None:
        return None
    best_case = max(arm_result["case_results"], key=lambda item: item["score"])
    for task in best_case.get("tasks", []):
        if task["arm_name"] == arm_result["arm_name"]:
            return task
    return None


def _evaluate_arm_pick_coverage(robot,
                                cfg,
                                obstacle_list,
                                object_specs,
                                task_specs,
                                candidate,
                                arm_name,
                                pick_options,
                                place_options,
                                plan_motion,
                                ik_backend):
    failure_samples = []
    best_arm_result = None
    for place_index, _place_pose in place_options:
        case_results = []
        for pick_index, _pick_pose in pick_options:
            case_specs = _task_specs_for_arm_case(task_specs,
                                                  object_specs,
                                                  arm_name,
                                                  pick_index,
                                                  place_index)
            result = _evaluate_case_on_context(robot,
                                               cfg,
                                               obstacle_list,
                                               object_specs,
                                               case_specs,
                                               candidate,
                                               plan_motion,
                                               ik_backend)
            gc.collect()
            if not result["ok"]:
                failure_samples.append({
                    "arm_name": arm_name,
                    "fixed_place_pose_index": int(place_index),
                    "pick_pose_index": int(pick_index),
                    "reason": result["reason"],
                })
                break
            _annotate_arm_case_tasks(result, arm_name, pick_index, place_index)
            case_results.append(_case_record(result, arm_name, pick_index, place_index))
        if len(case_results) == len(pick_options):
            total_joint_delta_cost = sum(case["joint_delta_cost"] for case in case_results)
            total_path_point_count = None
            if plan_motion:
                total_path_point_count = sum(int(case["path_point_count"]) for case in case_results)
                total_score = _inverse_length_score(total_path_point_count)
            else:
                total_score = _inverse_length_score(total_joint_delta_cost)
            max_case_score = max((case["score"] for case in case_results), default=0.0)
            arm_result = {
                "ok": True,
                "arm_name": arm_name,
                "object_name": task_specs[arm_name]["object_name"],
                "fixed_place_pose_index": int(place_index),
                "required_pick_pose_count": int(len(pick_options)),
                "covered_pick_pose_count": int(len(case_results)),
                "total_score": float(total_score),
                "max_case_score": float(max_case_score),
                "total_joint_delta_cost": float(total_joint_delta_cost),
                "total_path_point_count": total_path_point_count,
                "score_metric": PLAN_SCORE_METRIC_DESCRIPTION
                if plan_motion else IK_SCORE_METRIC_DESCRIPTION,
                "case_results": case_results,
            }
            if best_arm_result is None or arm_result["total_score"] > best_arm_result["total_score"]:
                best_arm_result = arm_result
    if best_arm_result is not None:
        return best_arm_result
    return {
        "ok": False,
        "arm_name": arm_name,
        "object_name": task_specs[arm_name]["object_name"],
        "required_pick_pose_count": int(len(pick_options)),
        "covered_pick_pose_count": 0,
        "failure_samples": failure_samples[:8],
    }


def evaluate_coverage_candidate(base_cfg, candidate, task_specs, plan_motion, ik_backend):
    cfg = cfgutils.apply_search_candidate(base_cfg, candidate)
    backend = rack_search.resolve_ik_backend(cfg, override=ik_backend)
    ik_backends.set_backend(cfg, backend)
    object_specs = cfgutils.object_specs(cfg)
    pick_options, place_options = _arm_option_maps(task_specs, object_specs)
    robot = robot_factory.build_robot(cfg, enable_cc=True, ik_backend=backend)
    obstacle_list, _ = scene.build_scene(None, cfg, object_specs, attach_visuals=False)
    arm_results = {}
    failure_samples = []
    for arm_name in task_specs:
        arm_result = _evaluate_arm_pick_coverage(robot,
                                                 cfg,
                                                 obstacle_list,
                                                 object_specs,
                                                 task_specs,
                                                 candidate,
                                                 arm_name,
                                                 pick_options[arm_name],
                                                 place_options[arm_name],
                                                 plan_motion,
                                                 backend)
        if not arm_result["ok"]:
            failure_samples.extend(arm_result.get("failure_samples", []))
            return {
                "ok": False,
                "candidate": candidate,
                "score": float("-inf"),
                "score_metric": PLAN_SCORE_METRIC_DESCRIPTION if plan_motion else IK_SCORE_METRIC_DESCRIPTION,
                "frame_count": None,
                "reason": (f"{arm_name} has no single place pose covering all pick poses; "
                           f"required {arm_result['required_pick_pose_count']} pick poses."),
                "ik_backend": backend,
                "coverage": {
                    "mode": "per_arm_all_pick_poses_one_fixed_place_pose",
                    "required_pick_pose_count_by_arm": {
                        name: int(len(pick_options[name])) for name in task_specs
                    },
                    "covered_pick_pose_count_by_arm": {
                        name: int(arm_results.get(name, {}).get("covered_pick_pose_count", 0))
                        for name in task_specs
                    } | {arm_name: 0},
                    "pick_symmetry_policy": "free_per_pick_pose",
                    "place_symmetry_policy": "free_per_pick_pose",
                    "failure_samples": failure_samples[:8],
                },
            }
        arm_results[arm_name] = arm_result
    representative_tasks = [
        task for task in (_select_representative_task(arm_results[arm_name]) for arm_name in task_specs)
        if task is not None
    ]
    total_joint_delta_cost = sum(arm_result["total_joint_delta_cost"] for arm_result in arm_results.values())
    total_path_point_count = None
    if plan_motion:
        total_path_point_count = sum(int(arm_result["total_path_point_count"]) for arm_result in arm_results.values())
        total_score = _inverse_length_score(total_path_point_count)
        score_metric = PLAN_SCORE_METRIC_DESCRIPTION
    else:
        total_score = _inverse_length_score(total_joint_delta_cost)
        score_metric = IK_SCORE_METRIC_DESCRIPTION
    max_case_score = max((arm_result["max_case_score"] for arm_result in arm_results.values()), default=0.0)
    frame_count = None
    if plan_motion:
        frame_count = sum(
            0 if case["frame_count"] is None else int(case["frame_count"])
            for arm_result in arm_results.values()
            for case in arm_result["case_results"])
    return {
        "ok": True,
        "candidate": candidate,
        "score": float(total_score),
        "score_metric": score_metric,
        "joint_delta_cost": float(total_joint_delta_cost),
        "path_point_count": total_path_point_count,
        "frame_count": frame_count,
        "reason": "ok",
        "ik_backend": backend,
        "tasks": representative_tasks,
        "coverage": {
            "mode": "per_arm_all_pick_poses_one_fixed_place_pose",
            "fixed_place_pose_assignment": {
                arm_name: int(arm_result["fixed_place_pose_index"])
                for arm_name, arm_result in arm_results.items()
            },
            "pick_symmetry_policy": "free_per_pick_pose",
            "place_symmetry_policy": "free_per_pick_pose",
            "required_pick_pose_count_by_arm": {
                arm_name: int(arm_result["required_pick_pose_count"])
                for arm_name, arm_result in arm_results.items()
            },
            "covered_pick_pose_count_by_arm": {
                arm_name: int(arm_result["covered_pick_pose_count"])
                for arm_name, arm_result in arm_results.items()
            },
            "arm_results": arm_results,
            "total_score": float(total_score),
            "max_case_score": float(max_case_score),
            "total_joint_delta_cost": float(total_joint_delta_cost),
            "total_path_point_count": total_path_point_count,
            "path_point_count_by_arm": {
                arm_name: arm_result["total_path_point_count"]
                for arm_name, arm_result in arm_results.items()
            },
            "joint_delta_cost_by_arm": {
                arm_name: arm_result["total_joint_delta_cost"]
                for arm_name, arm_result in arm_results.items()
            },
            "score_metric": score_metric,
        },
    }


def summarize_task_specs(task_specs, task_search_info):
    console.section("Pick/place search setup")
    for arm_name, task_spec in task_specs.items():
        search_info = task_search_info[arm_name]
        console.key_value(f"{arm_name} object", task_spec["object_name"])
        console.key_value(f"{arm_name} pick_pose_mode",
                          "all object pick_pose_candidates"
                          if "pick_pose_candidates" not in task_spec else len(task_spec["pick_pose_candidates"]))
        console.key_value(f"{arm_name} place_indices", task_spec.get("place_indices", "all object place_poses"))
        console.key_value(f"{arm_name} search counts",
                          f"pick_poses={search_info['pick_pose_count']}, "
                          f"place_poses={search_info['place_pose_count']}, "
                          f"pick_symmetries={search_info['pick_symmetry_count']}, "
                          f"place_symmetries={search_info['place_symmetry_count']}, "
                          f"pose/symmetry tuples={search_info['pose_symmetry_tuple_count']}")
        if "grasp_indices" in task_spec:
            console.key_value(f"{arm_name} grasp_indices", task_spec["grasp_indices"])


def print_runtime_header(base_cfg, search_cfg, args):
    console.section("Runtime")
    console.key_value("TracIK loaded before Panda3D/WRS",
                      f"{bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__})",
                      color=console.Fore.GREEN if _TracIK else console.Fore.RED)
    console.key_value("terminal color", console.color_package_hint(), color=console.Fore.CYAN)
    console.key_value("config", base_cfg.get("_config_path", "<memory>"))
    console.key_value("search_config", search_cfg.get("_config_path", "<memory>"))
    console.key_value("candidate source", "rack_search.yaml grid")
    console.key_value("task overrides", search_cfgutils.task_overrides(search_cfg) or "<none>")
    console.key_value("mode", "IK + RRT motion" if args.plan_motion else "IK only")
    console.key_value("ik_backend", args.ik_backend)
    console.key_value("workers", args.workers)
    console.key_value("worker_maxtasks_per_child", args.worker_maxtasks_per_child)
    console.key_value("group_by_mount", args.group_by_mount)
    grid_counts = rack_search.grid_counts(search_cfg)
    console.key_value("mount_group_blacklist_enabled", grid_counts["blacklist_enabled"])
    console.key_value("mount_group_blacklist_path", grid_counts["blacklist_path"])
    console.key_value("raw_mount_groups", grid_counts["raw_mount_group_count"])
    console.key_value("blacklisted_mount_groups", grid_counts["blacklisted_mount_group_count"],
                      color=console.Fore.YELLOW)
    console.key_value("mount_groups", grid_counts["mount_group_count"])
    console.key_value("placements_per_mount_group", grid_counts["placement_count_per_mount_group"])
    console.key_value("total_grid_candidates", grid_counts["candidate_count"])
    console.key_value("start_index", args.start_index)
    console.key_value("max_candidates", args.max_candidates)
    console.key_value("result_path", args.result_path)
    console.key_value("score_metric",
                      PLAN_SCORE_METRIC_DESCRIPTION if args.plan_motion else IK_SCORE_METRIC_DESCRIPTION)
    ik_seed_note = "ignored by multi-solution IK backend" if ik_backends.is_multi_solution_backend(args.ik_backend) else "used for pick-stage random seeds"
    console.key_value("pnp.ik_seed_count", f"{base_cfg['pnp'].get('ik_seed_count', 80)} ({ik_seed_note})")
    console.key_value("pnp.ik_max_branch_count", base_cfg["pnp"].get("ik_max_branch_count", 8))


def compact_result(index,
                   result,
                   worker_elapsed,
                   completed_candidates,
                   search_elapsed_when_found,
                   task_search_info):
    return search_result.to_builtin({
        "candidate_index": int(index),
        "completed_candidates_when_found": int(completed_candidates),
        "score": float(result["score"]),
        "score_metric": result.get("score_metric"),
        "joint_delta_cost": result.get("joint_delta_cost"),
        "path_point_count": result.get("path_point_count"),
        "frame_count": result["frame_count"],
        "worker_elapsed": float(worker_elapsed),
        "search_elapsed_when_found": float(search_elapsed_when_found),
        "ik_backend": result.get("ik_backend"),
        "candidate": result["candidate"],
        "task_search_info": task_search_info,
        "coverage": result.get("coverage", {}),
        "tasks": result.get("tasks", []),
    })


def _fmt_joint_delta(task):
    joint_delta = task.get("joint_delta", {})
    return (f"pick->lift={joint_delta.get('pick_to_lift', 0.0):.3f}, "
            f"lift->pre-place={joint_delta.get('lift_to_pre_place', 0.0):.3f}, "
            f"pre-place->place={joint_delta.get('pre_place_to_place', 0.0):.3f}")


def print_feasible(index, result, worker_elapsed, task_search_info, checked_count=None, search_elapsed=None):
    candidate = result["candidate"]
    mount_group = candidate.get("mount_group", {})
    coverage = result.get("coverage", {})
    console.separator()
    console.success(f"[Candidate #{index}] feasible.")
    console.key_value("completed_candidates", checked_count if checked_count is not None else "<unknown>",
                      color=console.Fore.GREEN)
    console.key_value("score", f"{result['score']:.3f}", color=console.Fore.GREEN)
    console.key_value("worker_elapsed", f"{worker_elapsed:.2f}s", color=console.Fore.GREEN)
    if search_elapsed is not None:
        console.key_value("search_elapsed", f"{search_elapsed:.2f}s", color=console.Fore.GREEN)
    console.key_value("ik_backend", result.get("ik_backend", "<unknown>"), color=console.Fore.GREEN)
    console.key_value("frame_count", result["frame_count"], color=console.Fore.GREEN)
    console.key_value("path_point_count", result.get("path_point_count"), color=console.Fore.GREEN)
    console.key_value("joint_delta_cost", result.get("joint_delta_cost"), color=console.Fore.GREEN)
    console.key_value("score_metric", result.get("score_metric"),
                      color=console.Fore.GREEN)
    if mount_group:
        console.key_value("mount_group",
                          f"{mount_group.get('index')} "
                          f"placement {mount_group.get('placement_index')}/"
                          f"{mount_group.get('placement_count')}",
                          color=console.Fore.CYAN)
    if coverage:
        console.key_value("coverage_mode", coverage.get("mode"), color=console.Fore.GREEN)
        console.key_value("fixed_place_pose_assignment",
                          coverage.get("fixed_place_pose_assignment"),
                          color=console.Fore.GREEN)
        console.key_value("pick_symmetry_policy", coverage.get("pick_symmetry_policy"), color=console.Fore.GREEN)
        console.key_value("place_symmetry_policy", coverage.get("place_symmetry_policy"), color=console.Fore.GREEN)
        console.key_value("pick_pose_coverage_by_arm",
                          f"{coverage.get('covered_pick_pose_count_by_arm')}/"
                          f"{coverage.get('required_pick_pose_count_by_arm')}",
                          color=console.Fore.GREEN)
        console.key_value("coverage_total_score", f"{coverage.get('total_score', result['score']):.3f}",
                          color=console.Fore.GREEN)
        console.key_value("coverage_max_case_score", f"{coverage.get('max_case_score', 0.0):.3f}",
                          color=console.Fore.GREEN)
        console.key_value("coverage_total_path_point_count",
                          coverage.get("total_path_point_count"),
                          color=console.Fore.GREEN)
        console.key_value("path_point_count_by_arm",
                          coverage.get("path_point_count_by_arm"),
                          color=console.Fore.GREEN)
    console.key_value("rack.base_pos", candidate["rack_base_pos"], color=console.Fore.CYAN)
    console.key_value("rack.yaw_deg", candidate["rack_yaw_deg"], color=console.Fore.CYAN)
    console.key_value("rack.arm_distance", candidate["arm_distance"], color=console.Fore.CYAN)
    console.key_value("rack.left_arm_mount_euler_deg",
                      candidate["left_arm_mount_euler_deg"],
                      color=console.Fore.CYAN)
    for task in result.get("tasks", []):
        search_info = task_search_info.get(task["arm_name"], {})
        console.info(f"  {task['arm_name']}: object={task['object_name']}, "
                     f"pick_pose=#{task['pick_pose_index']}, "
                     f"pick_sym={task['pick_symmetry_deg']:.1f}deg, "
                     f"grasp=#{task['grasp_index']}, "
                     f"place_pose=#{task['place_pose_index']}, "
                     f"place_sym={task['place_symmetry_deg']:.1f}deg, "
                     f"jaw_width={task['jaw_width']:.4f}")
        console.dim(f"    search_space: pick_poses={search_info.get('pick_pose_count')}, "
                    f"place_poses={search_info.get('place_pose_count')}, "
                    f"pick_symmetries={search_info.get('pick_symmetry_count')}, "
                    f"place_symmetries={search_info.get('place_symmetry_count')}")
        console.dim(f"    joint_delta: {_fmt_joint_delta(task)}")


def print_top_results(feasible_results, limit):
    if not feasible_results or limit <= 0:
        return
    console.section(f"Top {min(limit, len(feasible_results))} feasible results")
    for rank, item in enumerate(feasible_results[:limit], start=1):
        candidate = item["candidate"]
        console.success(f"#{rank}: candidate #{item['candidate_index']}, "
                        f"score={item['score']:.3f}, "
                        f"path_points={item.get('path_point_count')}, "
                        f"worker_elapsed={item['worker_elapsed']:.2f}s, "
                        f"search_elapsed={item.get('search_elapsed_when_found', 0.0):.2f}s, "
                        f"backend={item.get('ik_backend')}")
        coverage = item.get("coverage", {})
        if coverage:
            console.dim(f"  coverage: fixed_place_pose_assignment="
                        f"{coverage.get('fixed_place_pose_assignment')}, "
                        f"pick_symmetry_policy={coverage.get('pick_symmetry_policy')}, "
                        f"place_symmetry_policy={coverage.get('place_symmetry_policy')}, "
                        f"pick_pose_coverage_by_arm={coverage.get('covered_pick_pose_count_by_arm')}/"
                        f"{coverage.get('required_pick_pose_count_by_arm')}, "
                        f"path_point_count_by_arm={coverage.get('path_point_count_by_arm')}, "
                        f"max_case_score={coverage.get('max_case_score', 0.0):.3f}")
        mount_group = candidate.get("mount_group", {})
        if mount_group:
            console.dim(f"  mount_group={mount_group.get('index')}, "
                        f"placement={mount_group.get('placement_index')}/"
                        f"{mount_group.get('placement_count')}")
        console.dim(f"  rack.base_pos={candidate['rack_base_pos']}, "
                    f"yaw={candidate['rack_yaw_deg']}, "
                    f"arm_distance={candidate['arm_distance']}, "
                    f"left_mount={candidate['left_arm_mount_euler_deg']}")
        for task in item.get("tasks", []):
            search_info = item.get("task_search_info", {}).get(task["arm_name"], {})
            console.dim(f"  {task['arm_name']}: {task['object_name']}, "
                        f"pick=#{task['pick_pose_index']}, "
                        f"pick_sym={task['pick_symmetry_deg']:.1f}deg, "
                        f"grasp=#{task['grasp_index']}, "
                        f"place=#{task['place_pose_index']}, "
                        f"place_sym={task['place_symmetry_deg']:.1f}deg, "
                        f"counts(pick/place/sym)={search_info.get('pick_pose_count')}/"
                        f"{search_info.get('place_pose_count')}/"
                        f"{search_info.get('pick_symmetry_count')}x"
                        f"{search_info.get('place_symmetry_count')}, "
                        f"{_fmt_joint_delta(task)}")


def _candidate_mount_key(candidate):
    return (
        float(candidate["arm_distance"]),
        tuple(float(value) for value in candidate["left_arm_mount_euler_deg"]),
    )


def _build_mount_group_stats(candidate_items):
    stats = {}
    for _index, candidate in candidate_items:
        key = _candidate_mount_key(candidate)
        group = stats.get(key)
        if group is None:
            mount_group = candidate.get("mount_group", {})
            group = {
                "mount_group_index": mount_group.get("index"),
                "arm_distance": candidate["arm_distance"],
                "left_arm_mount_euler_deg": candidate["left_arm_mount_euler_deg"],
                "expected": 0,
                "completed": 0,
                "feasible": 0,
                "best_score": float("-inf"),
                "best_candidate_index": None,
                "best_rack_base_pos": None,
                "best_rack_yaw_deg": None,
                "best_path_point_count": None,
                "best_score_metric": None,
                "reported": False,
            }
            stats[key] = group
        group["expected"] += 1
    return stats


def _mount_group_counters(group_stats):
    completed_groups = [
        group for group in group_stats.values()
        if group["completed"] >= group["expected"]
    ]
    feasible_mount_count = sum(1 for group in completed_groups if group["feasible"] > 0)
    infeasible_mount_count = sum(1 for group in completed_groups if group["feasible"] == 0)
    return {
        "total_mount_groups": int(len(group_stats)),
        "completed_mount_groups": int(len(completed_groups)),
        "feasible_mount_groups": int(feasible_mount_count),
        "infeasible_mount_groups": int(infeasible_mount_count),
        "unfinished_mount_groups": int(len(group_stats) - len(completed_groups)),
    }


def _update_mount_group_stats(group_stats, candidate_index, result):
    candidate = result["candidate"]
    group = group_stats[_candidate_mount_key(candidate)]
    group["completed"] += 1
    if result["ok"]:
        group["feasible"] += 1
        if result["score"] > group["best_score"]:
            group["best_score"] = float(result["score"])
            group["best_candidate_index"] = int(candidate_index)
            group["best_rack_base_pos"] = candidate["rack_base_pos"]
            group["best_rack_yaw_deg"] = candidate["rack_yaw_deg"]
            group["best_path_point_count"] = result.get("path_point_count")
            group["best_score_metric"] = result.get("score_metric")
    if group["completed"] >= group["expected"] and not group["reported"]:
        group["reported"] = True
        print_mount_group_finished(group, _mount_group_counters(group_stats))


def print_mount_group_finished(group, counters):
    console.section(f"Mount group {group['mount_group_index']} finished")
    console.key_value("arm_distance", group["arm_distance"], color=console.Fore.CYAN)
    console.key_value("left_arm_mount_euler_deg", group["left_arm_mount_euler_deg"], color=console.Fore.CYAN)
    console.key_value("checked_in_group", group["completed"])
    console.key_value("feasible_in_group", group["feasible"], color=console.Fore.GREEN)
    console.key_value("infeasible_in_group", group["completed"] - group["feasible"], color=console.Fore.YELLOW)
    console.key_value("mount_status",
                      "feasible" if group["feasible"] > 0 else "infeasible",
                      color=console.Fore.GREEN if group["feasible"] > 0 else console.Fore.YELLOW)
    console.key_value("completed_mount_num",
                      f"{counters['completed_mount_groups']}/{counters['total_mount_groups']}")
    console.key_value("feasible_mount_num", counters["feasible_mount_groups"], color=console.Fore.GREEN)
    console.key_value("infeasible_mount_num", counters["infeasible_mount_groups"], color=console.Fore.YELLOW)
    if group["feasible"] > 0:
        console.key_value("best_candidate_index", group["best_candidate_index"], color=console.Fore.GREEN)
        console.key_value("best_score", f"{group['best_score']:.3f}", color=console.Fore.GREEN)
        console.key_value("best_rack_base_pos", group["best_rack_base_pos"], color=console.Fore.GREEN)
        console.key_value("best_rack_yaw_deg", group["best_rack_yaw_deg"], color=console.Fore.GREEN)
        console.key_value("best_path_point_count", group["best_path_point_count"], color=console.Fore.GREEN)
        console.key_value("best_score_metric", group["best_score_metric"], color=console.Fore.GREEN)


def search_all_results(base_cfg, search_cfg, task_specs, task_search_info, args):
    tic = time.perf_counter()
    checked_count = 0
    failed_count = 0
    feasible_results = []
    worker_count = max(1, int(args.workers))
    candidate_items = list(iter_indexed_candidates(search_cfg, args))
    mount_group_stats = _build_mount_group_stats(candidate_items)

    if worker_count <= 1:
        for index, candidate in candidate_items:
            checked_count += 1
            worker_tic = time.perf_counter()
            result, raw_log = evaluate_candidate(base_cfg,
                                                 candidate,
                                                 task_specs,
                                                 args.plan_motion,
                                                 args.verbose,
                                                 args.ik_backend)
            worker_elapsed = time.perf_counter() - worker_tic
            if args.verbose and raw_log:
                print(raw_log.rstrip())
            if result["ok"]:
                search_elapsed = time.perf_counter() - tic
                feasible_results.append(compact_result(index,
                                                       result,
                                                       worker_elapsed,
                                                       checked_count,
                                                       search_elapsed,
                                                       task_search_info))
                print_feasible(index,
                               result,
                               worker_elapsed,
                               task_search_info,
                               checked_count=checked_count,
                               search_elapsed=search_elapsed)
            else:
                failed_count += 1
                if args.print_failures:
                    console.warning(f"[Candidate #{index}] failed in {worker_elapsed:.2f}s: {result['reason']}")
            _update_mount_group_stats(mount_group_stats, index, result)
            maybe_print_progress(checked_count,
                                 feasible_results,
                                 failed_count,
                                 args.progress_interval,
                                 search_elapsed=time.perf_counter() - tic)
    else:
        console.info("Parallel all-results search evaluates every candidate in the requested range.")
        ctx = mp.get_context("spawn")
        maxtasks = None if args.worker_maxtasks_per_child <= 0 else int(args.worker_maxtasks_per_child)
        with ctx.Pool(processes=worker_count,
                      initializer=_init_worker,
                      initargs=(base_cfg, task_specs, args.plan_motion, args.verbose, args.ik_backend),
                      maxtasksperchild=maxtasks) as pool:
            for index, result, raw_log, worker_elapsed in pool.imap_unordered(
                    _evaluate_candidate_payload,
                    candidate_items,
                    chunksize=1):
                checked_count += 1
                if args.verbose and raw_log:
                    print(raw_log.rstrip())
                if result["ok"]:
                    search_elapsed = time.perf_counter() - tic
                    feasible_results.append(compact_result(index,
                                                           result,
                                                           worker_elapsed,
                                                           checked_count,
                                                           search_elapsed,
                                                           task_search_info))
                    print_feasible(index,
                                   result,
                                   worker_elapsed,
                                   task_search_info,
                                   checked_count=checked_count,
                                   search_elapsed=search_elapsed)
                else:
                    failed_count += 1
                    if args.print_failures:
                        console.warning(f"[Candidate #{index}] failed in {worker_elapsed:.2f}s: {result['reason']}")
                _update_mount_group_stats(mount_group_stats, index, result)
                maybe_print_progress(checked_count,
                                     feasible_results,
                                     failed_count,
                                     args.progress_interval,
                                     search_elapsed=time.perf_counter() - tic)

    feasible_results.sort(key=lambda item: (-item["score"], item["candidate_index"]))
    elapsed = time.perf_counter() - tic
    mount_summary = _mount_group_counters(mount_group_stats)
    result_path = search_result.write_all_results(args.result_path,
                                                  base_cfg,
                                                  search_cfg,
                                                  args,
                                                  task_specs,
                                                  feasible_results,
                                                  checked_count=checked_count,
                                                  failed_count=failed_count,
                                                  elapsed=elapsed,
                                                  mount_summary=mount_summary)
    console.section("All-results search summary")
    console.key_value("elapsed", f"{elapsed:.2f}s")
    console.key_value("checked", checked_count)
    console.key_value("feasible", len(feasible_results), color=console.Fore.GREEN)
    console.key_value("failed", failed_count, color=console.Fore.YELLOW)
    console.key_value("completed_mount_num",
                      f"{mount_summary['completed_mount_groups']}/{mount_summary['total_mount_groups']}")
    console.key_value("feasible_mount_num", mount_summary["feasible_mount_groups"], color=console.Fore.GREEN)
    console.key_value("infeasible_mount_num", mount_summary["infeasible_mount_groups"], color=console.Fore.YELLOW)
    if checked_count > 0:
        console.key_value("avg_wall_time_per_completed_candidate", f"{elapsed / checked_count:.2f}s")
    console.key_value("result_yaml", result_path, color=console.Fore.GREEN)
    if feasible_results:
        best = feasible_results[0]
        console.success(f"Best result: candidate #{best['candidate_index']}, score={best['score']:.3f}.")
        print_top_results(feasible_results, args.summary_top)
    return feasible_results


def evaluate_candidate(base_cfg, candidate, task_specs, plan_motion, verbose, ik_backend):
    if verbose:
        return evaluate_coverage_candidate(base_cfg,
                                           candidate,
                                           task_specs,
                                           plan_motion=plan_motion,
                                           ik_backend=ik_backend), ""
    stream = _NullStream()
    with contextlib.redirect_stdout(stream):
        result = evaluate_coverage_candidate(base_cfg,
                                             candidate,
                                             task_specs,
                                             plan_motion=plan_motion,
                                             ik_backend=ik_backend)
    return result, ""


def maybe_print_progress(checked_count, feasible_results, failed_count, progress_interval, search_elapsed=None):
    if progress_interval <= 0 or checked_count % progress_interval != 0:
        return
    console.section("Progress")
    console.key_value("checked", checked_count)
    console.key_value("feasible", len(feasible_results), color=console.Fore.GREEN)
    console.key_value("failed", failed_count, color=console.Fore.YELLOW)
    if search_elapsed is not None:
        console.key_value("elapsed", f"{search_elapsed:.2f}s")
        console.key_value("avg_wall_time_per_completed_candidate", f"{search_elapsed / checked_count:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Search all feasible rack candidates from rack_search.yaml; pick poses are all searched by default.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--lft-pick-index", type=int, default=None,
                        help="Restrict left arm to one pick pose; omit to search all pick poses.")
    parser.add_argument("--rgt-pick-index", type=int, default=None,
                        help="Restrict right arm to one pick pose; omit to search all pick poses.")
    parser.add_argument("--lft-place-index", type=int, default=None,
                        help="Restrict left arm to one place pose; omit to use configured place_indices.")
    parser.add_argument("--rgt-place-index", type=int, default=None,
                        help="Restrict right arm to one place pose; omit to use configured place_indices.")
    parser.add_argument("--lft-grasp-index", type=int, default=None)
    parser.add_argument("--rgt-grasp-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--worker-maxtasks-per-child", type=int, default=5,
                        help="Restart each worker after this many rack candidates; use 0 to disable recycling.")
    parser.add_argument("--group-by-mount", dest="group_by_mount", action="store_true", default=True,
                        help="Evaluate candidates grouped by arm_distance and left_arm_mount_euler_deg.")
    parser.add_argument("--no-group-by-mount", dest="group_by_mount", action="store_false",
                        help="Use the raw Cartesian grid order instead of mount-group order.")
    parser.add_argument("--plan-motion", dest="plan_motion", action="store_true", default=None)
    parser.add_argument("--no-plan-motion", dest="plan_motion", action="store_false",
                        help="Disable RRT planning and use IK-only fallback scoring.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("--print-failures", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--ik-backend", default=None,
                        help="Override rack_search.yaml ik_backend, e.g. ikfast or tracik.")
    parser.add_argument("--summary-top", type=int, default=10,
                        help="Print this many best feasible results after the full search.")
    parser.add_argument("--result-path", default=search_result.DEFAULT_ALL_RESULTS_PATH)
    args = parser.parse_args()
    _ = _TracIK

    base_cfg = cfgutils.load_config(args.config)
    search_cfg = search_cfgutils.load_search_config(args.search_config)
    if args.plan_motion is None:
        args.plan_motion = bool(search_cfg.get("plan_motion", False))
    args.ik_backend = rack_search.resolve_ik_backend(base_cfg, search_cfg, args.ik_backend)
    ik_backends.set_backend(base_cfg, args.ik_backend)
    object_specs = cfgutils.object_specs(base_cfg)
    task_specs = make_task_specs(base_cfg,
                                 object_specs,
                                 lft_pick_index=args.lft_pick_index,
                                 lft_place_index=args.lft_place_index,
                                 rgt_pick_index=args.rgt_pick_index,
                                 rgt_place_index=args.rgt_place_index,
                                 lft_grasp_index=args.lft_grasp_index,
                                 rgt_grasp_index=args.rgt_grasp_index)
    task_specs = search_cfgutils.apply_task_overrides(task_specs, search_cfg)
    task_search_info = build_task_search_info(task_specs, object_specs)

    print_runtime_header(base_cfg, search_cfg, args)
    summarize_task_specs(task_specs, task_search_info)
    search_all_results(base_cfg, search_cfg, task_specs, task_search_info, args)


if __name__ == "__main__":
    main()
