import argparse
import time

from trac_ik import TracIK as _TracIK

from wrs import wd, mgm

from yanpu_pnp import animation
from yanpu_pnp import console
from yanpu_pnp import config as cfgutils
from yanpu_pnp import debug
from yanpu_pnp import ik
from yanpu_pnp import ik_backends
from yanpu_pnp import planner
from yanpu_pnp import rack_search
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene
from yanpu_pnp import search_config as search_cfgutils


def print_runtime_header(base_cfg, search_cfg=None):
    console.section("Runtime")
    console.key_value("TracIK loaded before Panda3D/WRS",
                      f"{bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__})",
                      color=console.Fore.GREEN if _TracIK else console.Fore.RED)
    console.key_value("terminal color", console.color_package_hint(), color=console.Fore.CYAN)
    console.key_value("config", base_cfg.get("_config_path", "<memory>"))
    if search_cfg is not None:
        console.key_value("search_config", search_cfg.get("_config_path", "<memory>"))
    console.key_value("ik_backend", base_cfg["pnp"].get("ik_backend", "<default>"))
    console.key_value("pnp.ik_seed_count", base_cfg["pnp"].get("ik_seed_count", 80))
    console.key_value("pnp.ik_max_branch_count", base_cfg["pnp"].get("ik_max_branch_count", 8))


def print_candidate_block(index, candidate):
    console.separator()
    console.section(f"Search candidate #{index}")
    console.key_value("rack_base_pos", candidate["rack_base_pos"])
    console.key_value("rack_yaw_deg", candidate["rack_yaw_deg"])
    console.key_value("arm_distance", candidate["arm_distance"])
    console.key_value("left_arm_mount_euler_deg", candidate["left_arm_mount_euler_deg"])


def candidate_by_index(search_cfg, candidate_index):
    for index, candidate in enumerate(rack_search.iter_search_candidates(search_cfg), start=1):
        if index == candidate_index:
            return candidate
    raise RuntimeError(f"Search candidate index {candidate_index} is out of range.")


def find_best_ik_candidate(base_cfg, search_cfg, max_candidates, start_index=1, task_specs=None, ik_backend=None):
    best_result = None
    ok_count = 0
    fail_count = 0
    checked_count = 0
    tic = time.perf_counter()
    for index, candidate in enumerate(rack_search.iter_search_candidates(search_cfg), start=1):
        if index < start_index:
            continue
        if max_candidates is not None and checked_count >= max_candidates:
            break
        checked_count += 1
        print_candidate_block(index, candidate)
        result = rack_search.evaluate_candidate(base_cfg,
                                                candidate,
                                                plan_motion=False,
                                                task_specs=task_specs,
                                                ik_backend=ik_backend)
        if result["ok"]:
            ok_count += 1
            console.success(f"[Candidate result] IK ok; score={result['score']:.3f}.")
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
                console.success("[Candidate result] selected as current best.")
        else:
            fail_count += 1
            console.warning(f"[Candidate result] IK failed: {result['reason']}")
    elapsed = time.perf_counter() - tic
    console.section("Search scan summary")
    console.key_value("elapsed", f"{elapsed:.2f}s")
    console.key_value("checked", checked_count)
    console.key_value("ok", ok_count, color=console.Fore.GREEN)
    console.key_value("failed", fail_count, color=console.Fore.YELLOW)
    if best_result is None:
        raise RuntimeError("No IK-feasible search candidate found; increase --max-candidates or adjust search YAML.")
    console.success(f"Best IK score: {best_result['score']:.3f}")
    console.key_value("Best IK candidate", best_result["candidate"], color=console.Fore.GREEN)
    return best_result["candidate"]


def summarize_tasks(task_list):
    for task in task_list:
        console.info(f"Task {task.arm_name}: object={task.object_name}, "
                     f"pick_pose_index={task.pick_pose_index}, grasp_index={task.grasp_index}, "
                     f"pick_symmetry_deg={task.symmetry_angle * 180.0 / 3.141592653589793:.1f}, "
                     f"place_pose_index={task.place_pose_index}, "
                     f"place_symmetry_deg={task.place_symmetry_angle * 180.0 / 3.141592653589793:.1f}")


def build_pickplace_path(cfg, toggle_visual=True, debug_on_failure=True, task_specs=None):
    base = None
    if toggle_visual:
        base = wd.World(**cfgutils.camera_kwargs(cfg))
        mgm.gen_frame().attach_to(base)

    object_specs = cfgutils.object_specs(cfg)
    if task_specs is None:
        task_specs = cfgutils.task_specs(cfg)
    robot = robot_factory.build_robot(cfg, enable_cc=True)
    robot_factory.print_ik_status(robot)
    obstacle_list, payload_dict = scene.build_scene(base, cfg, object_specs, attach_visuals=toggle_visual)
    try:
        task_list = ik.build_pick_place_tasks(robot,
                                             cfg,
                                             obstacle_list,
                                             object_specs=object_specs,
                                             task_specs=task_specs)
        summarize_tasks(task_list)
        frame_list = planner.build_frame_list(robot, cfg, obstacle_list, task_list)
    except RuntimeError as error:
        if debug_on_failure and toggle_visual:
            debug.attach_planning_failure_debug(base, cfg, robot, error, object_specs)
            console.error(f"Planning failed; debug scene is displayed for: {error}")
            base.run()
        raise

    task_dict = {task.object_name: task for task in task_list}
    console.success(f"Generated {len(frame_list)} pick-place frames.")
    if toggle_visual:
        animation.play(base, robot, frame_list, payload_dict, task_dict)
    return frame_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pick-place path from a rack-search candidate for regression testing.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--search-config", default=search_cfgutils.DEFAULT_SEARCH_CONFIG_PATH)
    parser.add_argument("--max-candidates", type=int, default=50,
                        help="How many search candidates to IK-scan before planning one full path.")
    parser.add_argument("--start-index", type=int, default=1,
                        help="Start scanning search candidates from this 1-based index.")
    parser.add_argument("--candidate-index", type=int, default=None,
                        help="Use one exact 1-based search candidate index instead of scanning for the best IK score.")
    parser.add_argument("--use-current-config", action="store_true",
                        help="Skip search and generate a path from default.yaml/current --config directly.")
    parser.add_argument("--ik-backend", default=None,
                        help="Override rack_search.yaml/default.yaml IK backend, e.g. ikfast or tracik.")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    _ = _TracIK

    base_cfg = cfgutils.load_config(args.config)
    task_specs = None
    if args.use_current_config:
        args.ik_backend = rack_search.resolve_ik_backend(base_cfg, override=args.ik_backend)
        ik_backends.set_backend(base_cfg, args.ik_backend)
        cfg = base_cfg
        print_runtime_header(base_cfg)
        console.info("Using current config directly; search candidate scan is skipped.")
    else:
        search_cfg = search_cfgutils.load_search_config(args.search_config)
        args.ik_backend = rack_search.resolve_ik_backend(base_cfg, search_cfg, args.ik_backend)
        ik_backends.set_backend(base_cfg, args.ik_backend)
        print_runtime_header(base_cfg, search_cfg)
        task_specs = search_cfgutils.apply_task_overrides(cfgutils.task_specs(base_cfg), search_cfg)
        if args.candidate_index is not None:
            candidate = candidate_by_index(search_cfg, args.candidate_index)
            print_candidate_block(args.candidate_index, candidate)
            console.info("[Search] using explicit candidate; IK scan is skipped.")
        else:
            candidate = find_best_ik_candidate(base_cfg,
                                               search_cfg,
                                               max_candidates=args.max_candidates,
                                               start_index=args.start_index,
                                               task_specs=task_specs,
                                               ik_backend=args.ik_backend)
        cfg = cfgutils.apply_search_candidate(base_cfg, candidate)
        ik_backends.set_backend(cfg, args.ik_backend)

    build_pickplace_path(cfg,
                         toggle_visual=not args.no_visual,
                         debug_on_failure=not args.no_debug,
                         task_specs=task_specs)


if __name__ == "__main__":
    main()
