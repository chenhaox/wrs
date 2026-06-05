import argparse

from trac_ik import TracIK as _TracIK

from wrs import wd, mgm

from yanpu_pnp import animation
from yanpu_pnp import console
from yanpu_pnp import config as cfgutils
from yanpu_pnp import debug
from yanpu_pnp import ik
from yanpu_pnp import planner
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene


def print_runtime_header(cfg):
    console.section("Runtime")
    console.key_value("TracIK loaded before Panda3D/WRS",
                      f"{bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__})",
                      color=console.Fore.GREEN if _TracIK else console.Fore.RED)
    console.key_value("terminal color", console.color_package_hint(), color=console.Fore.CYAN)
    console.key_value("config", cfg.get("_config_path", "<memory>"))
    console.key_value("pnp.ik_seed_count", cfg["pnp"].get("ik_seed_count", 80))
    console.key_value("pnp.ik_max_branch_count", cfg["pnp"].get("ik_max_branch_count", 8))


def _select_index(seq, index, label):
    if index < 0 or index >= len(seq):
        raise IndexError(f"{label} index {index} out of range [0, {len(seq) - 1}]")
    return seq[index]


def make_single_task_specs(cfg,
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
        task_spec["pick_symmetry_angle_count"] = 1
        task_spec["place_symmetry_angle_count"] = 1
        task_spec.pop("pick_symmetry_angles", None)
        task_spec.pop("pick_symmetry_angles_deg", None)
        task_spec.pop("place_symmetry_angles", None)
        task_spec.pop("place_symmetry_angles_deg", None)
        if grasp_index is not None:
            task_spec["grasp_indices"] = [int(grasp_index)]
    return task_specs


def summarize_single_setup(task_specs):
    console.section("Single-pose test setup")
    for arm_name, task_spec in task_specs.items():
        console.key_value(f"{arm_name} object", task_spec["object_name"])
        console.key_value(f"{arm_name} pick_pose_count", len(task_spec["pick_pose_candidates"]))
        console.key_value(f"{arm_name} place_indices", task_spec["place_indices"])
        console.key_value(f"{arm_name} pick_symmetry_angle_count", task_spec["pick_symmetry_angle_count"])
        console.key_value(f"{arm_name} place_symmetry_angle_count", task_spec["place_symmetry_angle_count"])
        if "grasp_indices" in task_spec:
            console.key_value(f"{arm_name} grasp_indices", task_spec["grasp_indices"])


def summarize_tasks(task_list):
    for task in task_list:
        console.info(f"Task {task.arm_name}: object={task.object_name}, "
                     f"pick_pose_index={task.pick_pose_index}, grasp_index={task.grasp_index}, "
                     f"pick_symmetry_deg={task.symmetry_angle * 180.0 / 3.141592653589793:.1f}, "
                     f"place_pose_index={task.place_pose_index}, "
                     f"place_symmetry_deg={task.place_symmetry_angle * 180.0 / 3.141592653589793:.1f}")


def run(cfg, task_specs, toggle_visual=True, debug_on_failure=True):
    base = None
    if toggle_visual:
        base = wd.World(**cfgutils.camera_kwargs(cfg))
        mgm.gen_frame().attach_to(base)

    object_specs = cfgutils.object_specs(cfg)
    robot = robot_factory.build_robot(cfg, enable_cc=True)
    robot_factory.print_tracik_status(robot)
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
    parser = argparse.ArgumentParser(description="Minimal dual-arm pick-and-place test: one pick pose and one place pose.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--lft-pick-index", type=int, default=0)
    parser.add_argument("--lft-place-index", type=int, default=0)
    parser.add_argument("--rgt-pick-index", type=int, default=0)
    parser.add_argument("--rgt-place-index", type=int, default=0)
    parser.add_argument("--lft-grasp-index", type=int, default=None)
    parser.add_argument("--rgt-grasp-index", type=int, default=None)
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    _ = _TracIK

    cfg = cfgutils.load_config(args.config)
    object_specs = cfgutils.object_specs(cfg)
    task_specs = make_single_task_specs(cfg,
                                        object_specs,
                                        lft_pick_index=args.lft_pick_index,
                                        lft_place_index=args.lft_place_index,
                                        rgt_pick_index=args.rgt_pick_index,
                                        rgt_place_index=args.rgt_place_index,
                                        lft_grasp_index=args.lft_grasp_index,
                                        rgt_grasp_index=args.rgt_grasp_index)
    print_runtime_header(cfg)
    summarize_single_setup(task_specs)
    run(cfg, task_specs, toggle_visual=not args.no_visual, debug_on_failure=not args.no_debug)


if __name__ == "__main__":
    main()
