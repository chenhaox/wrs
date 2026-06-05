import argparse

from trac_ik import TracIK as _TracIK

from wrs import wd, mgm

from yanpu_pnp import animation
from yanpu_pnp import config as cfgutils
from yanpu_pnp import debug
from yanpu_pnp import ik
from yanpu_pnp import planner
from yanpu_pnp import robot as robot_factory
from yanpu_pnp import scene


def run(cfg, toggle_visual=True, debug_on_failure=True):
    base = None
    if toggle_visual:
        base = wd.World(**cfgutils.camera_kwargs(cfg))
        mgm.gen_frame().attach_to(base)

    object_specs = cfgutils.object_specs(cfg)
    task_specs = cfgutils.task_specs(cfg)
    robot = robot_factory.build_robot(cfg, enable_cc=True)
    robot_factory.print_tracik_status(robot)
    obstacle_list, payload_dict = scene.build_scene(base, cfg, object_specs, attach_visuals=toggle_visual)
    try:
        task_list = ik.build_pick_place_tasks(robot,
                                             cfg,
                                             obstacle_list,
                                             object_specs=object_specs,
                                             task_specs=task_specs)
        task_dict = {task.object_name: task for task in task_list}
        frame_list = planner.build_frame_list(robot, cfg, obstacle_list, task_list)
    except RuntimeError as error:
        if debug_on_failure and toggle_visual:
            debug.attach_planning_failure_debug(base, cfg, robot, error, object_specs)
            print(f"Planning failed; debug scene is displayed for: {error}")
            base.run()
        raise
    print(f"Generated {len(frame_list)} dual-arm pick-and-place frames for UR7E + DH50.")
    if toggle_visual:
        animation.play(base, robot, frame_list, payload_dict, task_dict)
    return frame_list


def main():
    parser = argparse.ArgumentParser(description="Run structured UR7E + DH50 dual-arm pick-and-place.")
    parser.add_argument("--config", default=cfgutils.DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    _ = _TracIK
    cfg = cfgutils.load_config(args.config)
    run(cfg, toggle_visual=not args.no_visual, debug_on_failure=not args.no_debug)


if __name__ == "__main__":
    main()

