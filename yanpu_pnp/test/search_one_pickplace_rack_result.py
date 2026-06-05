import argparse
import contextlib
import io
import multiprocessing as mp
import time

from trac_ik import TracIK as _TracIK

from yanpu_pnp import config as cfgutils
from yanpu_pnp import console
from yanpu_pnp import ik_backends
from yanpu_pnp import rack_search
from yanpu_pnp import search_result
from yanpu_pnp import search_config as search_cfgutils


_WORKER_BASE_CFG = None
_WORKER_TASK_SPECS = None
_WORKER_PLAN_MOTION = False
_WORKER_VERBOSE = False
_WORKER_IK_BACKEND = None


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
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            result = rack_search.evaluate_candidate(_WORKER_BASE_CFG,
                                                    candidate,
                                                    plan_motion=_WORKER_PLAN_MOTION,
                                                    task_specs=_WORKER_TASK_SPECS,
                                                    ik_backend=_WORKER_IK_BACKEND)
    except Exception as error:
        result = {
            "ok": False,
            "candidate": candidate,
            "score": float("inf"),
            "frame_count": None,
            "reason": str(error),
        }
    raw_log = stream.getvalue() if _WORKER_VERBOSE else ""
    return index, result, raw_log, time.perf_counter() - tic


def _select_index(seq, index, label):
    if index < 0 or index >= len(seq):
        raise IndexError(f"{label} index {index} out of range [0, {len(seq) - 1}]")
    return seq[index]


def make_one_pickplace_task_specs(cfg,
                                  object_specs,
                                  lft_pick_index,
                                  lft_place_index,
                                  rgt_pick_index,
                                  rgt_place_index,
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
        pick_pose = _select_index(object_spec["pick_pose_candidates"],
                                  pick_index,
                                  f"{arm_name} {object_name} pick_pose")
        _select_index(object_spec["place_poses"], place_index, f"{arm_name} {object_name} place_pose")
        task_spec["pick_pose_candidates"] = [pick_pose]
        task_spec["place_indices"] = [place_index]
        task_spec.pop("place_index", None)
        if grasp_index is not None:
            task_spec["grasp_indices"] = [int(grasp_index)]
    return task_specs


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
    console.key_value("start_index", args.start_index)
    console.key_value("max_candidates", args.max_candidates)
    console.key_value("result_path", args.result_path)
    console.key_value("pnp.ik_seed_count", base_cfg["pnp"].get("ik_seed_count", 80))
    console.key_value("pnp.ik_max_branch_count", base_cfg["pnp"].get("ik_max_branch_count", 8))


def print_one_pose_setup(task_specs):
    console.section("One-pick-one-place setup")
    for arm_name, task_spec in task_specs.items():
        console.key_value(f"{arm_name} object", task_spec["object_name"])
        console.key_value(f"{arm_name} pick_pose_count", len(task_spec["pick_pose_candidates"]))
        console.key_value(f"{arm_name} place_indices", task_spec["place_indices"])
        console.key_value(f"{arm_name} pick_symmetry_angle_count", task_spec["pick_symmetry_angle_count"])
        console.key_value(f"{arm_name} place_symmetry_angle_count", task_spec["place_symmetry_angle_count"])
        if "grasp_indices" in task_spec:
            console.key_value(f"{arm_name} grasp_indices", task_spec["grasp_indices"])


def print_candidate(index, candidate):
    console.separator()
    console.section(f"Rack candidate #{index}")
    console.key_value("rack.base_pos", candidate["rack_base_pos"])
    console.key_value("rack.yaw_deg", candidate["rack_yaw_deg"])
    console.key_value("rack.arm_distance", candidate["arm_distance"])
    console.key_value("rack.left_arm_mount_euler_deg", candidate["left_arm_mount_euler_deg"])


def evaluate_candidate(base_cfg, candidate, task_specs, plan_motion, verbose, ik_backend):
    if verbose:
        return rack_search.evaluate_candidate(base_cfg,
                                             candidate,
                                             plan_motion=plan_motion,
                                             task_specs=task_specs,
                                             ik_backend=ik_backend), ""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = rack_search.evaluate_candidate(base_cfg,
                                                candidate,
                                                plan_motion=plan_motion,
                                                task_specs=task_specs,
                                                ik_backend=ik_backend)
    return result, stream.getvalue()


def iter_indexed_candidates(search_cfg, args):
    checked_count = 0
    for index, candidate in enumerate(rack_search.iter_search_candidates(search_cfg), start=1):
        if index < args.start_index:
            continue
        if args.max_candidates is not None and checked_count >= args.max_candidates:
            break
        checked_count += 1
        yield index, candidate


def print_feasible_result(index, result, elapsed, checked_count=None, result_path=None):
    console.success("[Candidate result] feasible.")
    console.key_value("candidate_index", index, color=console.Fore.GREEN)
    if checked_count is not None:
        console.key_value("completed_candidates", checked_count, color=console.Fore.GREEN)
    console.key_value("score", f"{result['score']:.3f}", color=console.Fore.GREEN)
    console.key_value("frame_count", result["frame_count"], color=console.Fore.GREEN)
    console.key_value("elapsed", f"{elapsed:.2f}s", color=console.Fore.GREEN)
    console.section("Rack search result")
    console.key_value("rack.base_pos", result["candidate"]["rack_base_pos"], color=console.Fore.GREEN)
    console.key_value("rack.yaw_deg", result["candidate"]["rack_yaw_deg"], color=console.Fore.GREEN)
    console.key_value("rack.arm_distance", result["candidate"]["arm_distance"], color=console.Fore.GREEN)
    console.key_value("rack.left_arm_mount_euler_deg",
                      result["candidate"]["left_arm_mount_euler_deg"],
                      color=console.Fore.GREEN)
    if result_path is not None:
        console.key_value("result_yaml", result_path, color=console.Fore.GREEN)


def search_first_result_parallel(base_cfg, search_cfg, task_specs, args):
    worker_count = max(1, int(args.workers))
    checked_count = 0
    fail_count = 0
    tic = time.perf_counter()
    console.info("Parallel search returns the first feasible candidate that finishes, not the lowest index.")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=worker_count,
                    initializer=_init_worker,
                    initargs=(base_cfg, task_specs, args.plan_motion, args.verbose, args.ik_backend))
    terminated = False
    try:
        for index, result, raw_log, worker_elapsed in pool.imap_unordered(
                _evaluate_candidate_payload,
                iter_indexed_candidates(search_cfg, args),
                chunksize=1):
            checked_count += 1
            print_candidate(index, result["candidate"])
            if args.verbose and raw_log:
                print(raw_log.rstrip())
            console.key_value("worker_elapsed", f"{worker_elapsed:.2f}s")
            if result["ok"]:
                elapsed = time.perf_counter() - tic
                result_path = search_result.write_result(args.result_path,
                                                         base_cfg,
                                                         search_cfg,
                                                         args,
                                                         task_specs,
                                                         result,
                                                         candidate_index=index,
                                                         completed_candidates=checked_count,
                                                         elapsed=elapsed,
                                                         worker_elapsed=worker_elapsed)
                print_feasible_result(index,
                                      result,
                                      elapsed,
                                      checked_count=checked_count,
                                      result_path=result_path)
                pool.terminate()
                terminated = True
                pool.join()
                return result
            fail_count += 1
            console.warning(f"[Candidate result] failed: {result['reason']}")
            if args.show_traceback and "traceback" in result:
                console.dim(result["traceback"])
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        terminated = True
        pool.join()
        raise
    finally:
        if not terminated:
            pass
    elapsed = time.perf_counter() - tic
    console.section("Search summary")
    console.key_value("elapsed", f"{elapsed:.2f}s")
    console.key_value("completed", checked_count)
    console.key_value("failed", fail_count, color=console.Fore.YELLOW)
    raise RuntimeError("No feasible one-pick-one-place rack result found in the scanned candidates.")


def search_first_result(base_cfg, search_cfg, task_specs, args):
    if int(args.workers) > 1:
        return search_first_result_parallel(base_cfg, search_cfg, task_specs, args)
    checked_count = 0
    fail_count = 0
    tic = time.perf_counter()
    for index, candidate in iter_indexed_candidates(search_cfg, args):
        checked_count += 1
        print_candidate(index, candidate)
        result, raw_log = evaluate_candidate(base_cfg,
                                             candidate,
                                             task_specs,
                                             plan_motion=args.plan_motion,
                                             verbose=args.verbose,
                                             ik_backend=args.ik_backend)
        if args.verbose and raw_log:
            print(raw_log)
        if result["ok"]:
            elapsed = time.perf_counter() - tic
            result_path = search_result.write_result(args.result_path,
                                                     base_cfg,
                                                     search_cfg,
                                                     args,
                                                     task_specs,
                                                     result,
                                                     candidate_index=index,
                                                     completed_candidates=checked_count,
                                                     elapsed=elapsed)
            print_feasible_result(index, result, elapsed, result_path=result_path)
            return result
        fail_count += 1
        console.warning(f"[Candidate result] failed: {result['reason']}")
        if args.show_traceback and "traceback" in result:
            console.dim(result["traceback"])
    elapsed = time.perf_counter() - tic
    console.section("Search summary")
    console.key_value("elapsed", f"{elapsed:.2f}s")
    console.key_value("checked", checked_count)
    console.key_value("failed", fail_count, color=console.Fore.YELLOW)
    raise RuntimeError("No feasible one-pick-one-place rack result found in the scanned candidates.")


def main():
    parser = argparse.ArgumentParser(
        description="Search the first rack candidate for one pick pose and one place pose per arm.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--lft-pick-index", type=int, default=0)
    parser.add_argument("--lft-place-index", type=int, default=0)
    parser.add_argument("--rgt-pick-index", type=int, default=0)
    parser.add_argument("--rgt-place-index", type=int, default=0)
    parser.add_argument("--lft-grasp-index", type=int, default=None)
    parser.add_argument("--rgt-grasp-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes. Use 1 for deterministic index order.")
    parser.add_argument("--plan-motion", dest="plan_motion", action="store_true", default=False,
                        help="Require the candidate to also generate a synchronized RRT path.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False,
                        help="Print raw IK/RRT logs for every scanned candidate.")
    parser.add_argument("--show-traceback", dest="show_traceback", action="store_true", default=False)
    parser.add_argument("--ik-backend", default=None,
                        help="Override rack_search.yaml ik_backend, e.g. ikfast or tracik.")
    parser.add_argument("--result-path", default=search_result.DEFAULT_RESULT_PATH,
                        help="YAML file used to record the first feasible rack search result.")
    args = parser.parse_args()
    _ = _TracIK

    base_cfg = cfgutils.load_config(args.config)
    search_cfg = search_cfgutils.load_search_config(args.search_config)
    args.ik_backend = rack_search.resolve_ik_backend(base_cfg, search_cfg, args.ik_backend)
    ik_backends.set_backend(base_cfg, args.ik_backend)
    object_specs = cfgutils.object_specs(base_cfg)
    task_specs = make_one_pickplace_task_specs(base_cfg,
                                               object_specs,
                                               lft_pick_index=args.lft_pick_index,
                                               lft_place_index=args.lft_place_index,
                                               rgt_pick_index=args.rgt_pick_index,
                                               rgt_place_index=args.rgt_place_index,
                                               lft_grasp_index=args.lft_grasp_index,
                                               rgt_grasp_index=args.rgt_grasp_index)
    task_specs = search_cfgutils.apply_task_overrides(task_specs, search_cfg)
    print_runtime_header(base_cfg, search_cfg, args)
    print_one_pose_setup(task_specs)
    search_first_result(base_cfg, search_cfg, task_specs, args)


if __name__ == "__main__":
    main()
