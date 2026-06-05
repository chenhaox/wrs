import numpy as np

from wrs import mgm

from yanpu_pnp import config as cfgutils
from yanpu_pnp import scene
from yanpu_pnp.models import MultiArmPlanningError, PickPlacePlanningError


def _sample_indices(path_len, max_samples):
    if path_len <= 0:
        return []
    if path_len <= max_samples:
        return list(range(path_len))
    return sorted(set(np.linspace(0, path_len - 1, max_samples, dtype=int).tolist()))


def _copy_conf_dict(conf_dict):
    return {arm_name: np.asarray(conf, dtype=float).copy() for arm_name, conf in conf_dict.items()}


def attach_tcp_trace(base, robot, path_dict, start_conf_dict, max_samples=70):
    color_dict = {
        "lft_arm": np.array([.0, .62, 1.0]),
        "rgt_arm": np.array([1.0, .42, .0]),
    }
    for arm_name, path in path_dict.items():
        if arm_name not in robot.arm_dict or not path:
            continue
        rgb = color_dict.get(arm_name, np.array([.85, .15, .95]))
        prev_pos = None
        for path_id in _sample_indices(len(path), max_samples):
            conf_dict = _copy_conf_dict(start_conf_dict)
            conf_dict[arm_name] = np.asarray(path[path_id], dtype=float)
            robot.goto_conf_dict(conf_dict)
            tcp_pos = robot.arm_dict[arm_name].gl_tcp_pos.copy()
            mgm.gen_sphere(pos=tcp_pos,
                           radius=.008,
                           rgb=rgb,
                           alpha=.65).attach_to(base)
            if prev_pos is not None:
                mgm.gen_stick(spos=prev_pos,
                              epos=tcp_pos,
                              radius=.0025,
                              rgb=rgb,
                              alpha=.35).attach_to(base)
            prev_pos = tcp_pos


def attach_multi_arm_planning_debug(base, robot, debug_info):
    if not debug_info:
        robot.gen_meshmodel(alpha=.9,
                            toggle_tcp_frame=True,
                            toggle_cdprim=True).attach_to(base)
        return
    start_conf_dict = debug_info.get("start_conf_dict", {})
    goal_conf_dict = debug_info.get("goal_conf_dict", {})
    path_dict = debug_info.get("path_dict", {})
    sipp_conf_dict = debug_info.get("sipp_conf_dict")
    failed_arm = debug_info.get("failed_arm")
    stage = debug_info.get("stage")
    sipp_state = debug_info.get("sipp_state")
    print("Drawing multi-arm planning debug:",
          f"stage={stage}, failed_arm={failed_arm}, sipp_state={sipp_state},",
          f"path_lengths={debug_info.get('path_lengths')}.")

    robot.backup_state()
    try:
        if start_conf_dict:
            robot.goto_conf_dict(start_conf_dict)
            robot.gen_meshmodel(alpha=.18,
                                toggle_tcp_frame=False,
                                toggle_cdprim=False).attach_to(base)
        if goal_conf_dict:
            robot.goto_conf_dict(goal_conf_dict)
            robot.gen_meshmodel(alpha=.22,
                                toggle_tcp_frame=True,
                                toggle_cdprim=False).attach_to(base)
        if start_conf_dict and path_dict:
            attach_tcp_trace(base, robot, path_dict, start_conf_dict)
        if sipp_conf_dict:
            robot.goto_conf_dict(sipp_conf_dict)
            robot.gen_meshmodel(alpha=.92,
                                toggle_tcp_frame=True,
                                toggle_cdprim=True).attach_to(base)
    finally:
        robot.restore_state()
    if sipp_conf_dict:
        robot.goto_conf_dict(sipp_conf_dict)


def attach_planning_failure_debug(base, cfg, robot, error, object_specs):
    if isinstance(error, MultiArmPlanningError):
        attach_multi_arm_planning_debug(base, robot, error.debug_info)
        return
    if isinstance(error, PickPlacePlanningError) and error.conf_dict:
        robot.goto_conf_dict(error.conf_dict)
    robot.gen_meshmodel(alpha=.9,
                        toggle_tcp_frame=True,
                        toggle_cdprim=True).attach_to(base)
    if isinstance(error, PickPlacePlanningError) and error.object_name in object_specs:
        scene.attach_grasp_debug_previews(base,
                                          cfgutils.model_dir(cfg),
                                          object_specs[error.object_name],
                                          error.grasp_indices)
