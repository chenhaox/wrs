import time

import numpy as np

import wrs.basis.robot_math as rm

from yanpu_pnp import config as cfgutils
from yanpu_pnp import console
from yanpu_pnp import grasping
from yanpu_pnp import ik_backends
from yanpu_pnp import scene
from yanpu_pnp import timing
from yanpu_pnp.models import PickPlacePlanningError, PickPlaceTask


def _is_arm_conf_collision_free(arm, conf, obstacle_list, other_robot_list=None, timing_stats=None):
    tic = time.perf_counter()
    arm.goto_given_conf(conf)
    if timing_stats is not None:
        timing_stats["fk_time"] += time.perf_counter() - tic
    tic = time.perf_counter()
    is_free = not arm.is_collided(obstacle_list=obstacle_list,
                                  other_robot_list=[] if other_robot_list is None else other_robot_list,
                                  toggle_dbg=False)
    if timing_stats is not None:
        timing_stats["collision_time"] += time.perf_counter() - tic
    return is_free


def _active_ik_backend(arm, ik_backend=None):
    if ik_backend is not None:
        return ik_backends.normalize_backend(ik_backend)
    arm_backend = getattr(arm, "ik_backend_name", None)
    if arm_backend is not None:
        return ik_backends.normalize_backend(arm_backend)
    return ik_backends.DEFAULT_IK_BACKEND


def _first_valid_seed(seed_conf_list, fallback_conf):
    for seed_conf in seed_conf_list:
        if seed_conf is not None:
            return np.asarray(seed_conf, dtype=float)
    return np.asarray(fallback_conf, dtype=float)


def _filter_ik_candidate_confs(arm,
                               candidate_conf_list,
                               tgt_pos,
                               tgt_rotmat,
                               obstacle_list,
                               other_robot_list=None,
                               pos_tol=.015,
                               rot_tol=.08,
                               duplicate_tol=1e-4,
                               timing_stats=None):
    conf_list = []
    for conf in candidate_conf_list:
        if conf is None:
            continue
        if timing_stats is not None:
            timing_stats["seed_trials"] += 1
        if not _is_arm_conf_collision_free(arm,
                                           conf,
                                           obstacle_list,
                                           other_robot_list,
                                           timing_stats=timing_stats):
            if timing_stats is not None:
                timing_stats["collision_rejects"] += 1
            continue
        tic = time.perf_counter()
        pos_err = np.linalg.norm(arm.gl_tcp_pos - tgt_pos)
        rot_err = np.linalg.norm(rm.delta_w_between_rotmat(arm.gl_tcp_rotmat, tgt_rotmat))
        if timing_stats is not None:
            timing_stats["pose_check_time"] += time.perf_counter() - tic
        if pos_err > pos_tol or rot_err > rot_tol:
            if timing_stats is not None:
                timing_stats["pose_rejects"] += 1
            continue
        conf = np.asarray(conf, dtype=float)
        if any(np.linalg.norm(conf - existing_conf) <= duplicate_tol for existing_conf in conf_list):
            if timing_stats is not None:
                timing_stats["duplicate_rejects"] += 1
            continue
        conf_list.append(conf)
        if timing_stats is not None:
            timing_stats["accepted_confs"] += 1
    return conf_list


def collect_ik_confs(arm,
                     tgt_pos,
                     tgt_rotmat,
                     obstacle_list,
                     seed_conf_list,
                     other_robot_list=None,
                     pos_tol=.015,
                     rot_tol=.08,
                     duplicate_tol=1e-4,
                     timing_stats=None,
                     ik_backend=None):
    backend = _active_ik_backend(arm, ik_backend)
    if ik_backends.is_multi_solution_backend(backend):
        reference_seed = _first_valid_seed(seed_conf_list, arm.arm.home_conf)
        tic = time.perf_counter()
        conf_candidates = arm.ik_all(tgt_pos=tgt_pos,
                                     tgt_rotmat=tgt_rotmat,
                                     seed_jnt_values=reference_seed,
                                     toggle_dbg=False)
        if timing_stats is not None:
            timing_stats["ik_time"] += time.perf_counter() - tic
            if not conf_candidates:
                timing_stats["ik_none"] += 1
        return _filter_ik_candidate_confs(arm,
                                          conf_candidates,
                                          tgt_pos,
                                          tgt_rotmat,
                                          obstacle_list,
                                          other_robot_list,
                                          pos_tol=pos_tol,
                                          rot_tol=rot_tol,
                                          duplicate_tol=duplicate_tol,
                                          timing_stats=timing_stats)
    conf_list = []
    for seed_conf in seed_conf_list:
        if seed_conf is None:
            continue
        if timing_stats is not None:
            timing_stats["seed_trials"] += 1
        tic = time.perf_counter()
        conf = arm.ik(tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rotmat,
                      seed_jnt_values=seed_conf,
                      toggle_dbg=False)
        if timing_stats is not None:
            timing_stats["ik_time"] += time.perf_counter() - tic
        if conf is None:
            if timing_stats is not None:
                timing_stats["ik_none"] += 1
            continue
        if not _is_arm_conf_collision_free(arm,
                                           conf,
                                           obstacle_list,
                                           other_robot_list,
                                           timing_stats=timing_stats):
            if timing_stats is not None:
                timing_stats["collision_rejects"] += 1
            continue
        tic = time.perf_counter()
        pos_err = np.linalg.norm(arm.gl_tcp_pos - tgt_pos)
        rot_err = np.linalg.norm(rm.delta_w_between_rotmat(arm.gl_tcp_rotmat, tgt_rotmat))
        if timing_stats is not None:
            timing_stats["pose_check_time"] += time.perf_counter() - tic
        if pos_err > pos_tol or rot_err > rot_tol:
            if timing_stats is not None:
                timing_stats["pose_rejects"] += 1
            continue
        conf = np.asarray(conf, dtype=float)
        if any(np.linalg.norm(conf - existing_conf) <= duplicate_tol for existing_conf in conf_list):
            if timing_stats is not None:
                timing_stats["duplicate_rejects"] += 1
            continue
        conf_list.append(conf)
        if timing_stats is not None:
            timing_stats["accepted_confs"] += 1
    return conf_list


def solve_task_conf(arm,
                    tgt_pos,
                    tgt_rotmat,
                    obstacle_list,
                    seed_conf_list,
                    other_robot_list=None,
                    ik_seed_count=80,
                    target_label="target",
                    reference_conf=None,
                    timing_stats=None,
                    ik_backend=None):
    solve_start = time.perf_counter()
    stage = timing.target_stage(target_label)
    target_seed_start = 0 if timing_stats is None else timing_stats["seed_trials"]
    target_accepted_start = 0 if timing_stats is None else timing_stats["accepted_confs"]
    stage_stats = None
    if timing_stats is not None:
        timing_stats["target_calls"] += 1
        stage_stats = timing.stage_timing_stats(timing_stats, stage)
        stage_stats["calls"] += 1
    backend = _active_ik_backend(arm, ik_backend)
    rng = np.random.default_rng(grasping.stable_seed("ik", arm.name, np.round(tgt_pos, 4)))
    jnt_ranges = arm.arm.jnt_ranges
    ik_seed_list = list(seed_conf_list)
    if not ik_backends.is_multi_solution_backend(backend):
        ik_seed_list.extend(rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1], size=(ik_seed_count, len(jnt_ranges))))
    conf_list = collect_ik_confs(arm,
                                 tgt_pos,
                                 tgt_rotmat,
                                 obstacle_list,
                                 ik_seed_list,
                                 other_robot_list,
                                 timing_stats=timing_stats,
                                 ik_backend=backend)
    elapsed = time.perf_counter() - solve_start
    if timing_stats is not None:
        timing_stats["solve_time"] += elapsed
        target_seed_trials = timing_stats["seed_trials"] - target_seed_start
        target_accepted_count = timing_stats["accepted_confs"] - target_accepted_start
        stage_stats["time"] += elapsed
        stage_stats["seed_trials"] += target_seed_trials
        stage_stats["accepted_confs"] += target_accepted_count
        timing_stats["slow_targets"].append((elapsed, stage, target_label, target_seed_trials, target_accepted_count))
    if conf_list:
        if timing_stats is not None:
            timing_stats["target_successes"] += 1
            stage_stats["successes"] += 1
        if reference_conf is None:
            return conf_list[0], "exact_ik"
        reference_conf = np.asarray(reference_conf, dtype=float)
        distance_list = [np.linalg.norm(conf - reference_conf) for conf in conf_list]
        best_id = int(np.argmin(distance_list))
        if timing_stats is None or timing_stats.get("print_target_solutions", False):
            print(f"[IK target] {arm.name}: {target_label}; "
                  f"accepted exact IK configs={len(conf_list)}, "
                  f"selected nearest reference joint_delta={distance_list[best_id]:.3f}.")
        return conf_list[best_id], "nearest_exact_ik"
    if timing_stats is not None:
        timing_stats["target_failures"] += 1
        stage_stats["failures"] += 1
    raise RuntimeError(
        f"No collision-free exact IK for {arm.name} {target_label}; fallback is disabled.")


def solve_task_conf_list(arm,
                         tgt_pos,
                         tgt_rotmat,
                         obstacle_list,
                         seed_conf_list,
                         other_robot_list=None,
                         ik_seed_count=0,
                         target_label="target",
                         reference_conf=None,
                         max_solutions=None,
                         timing_stats=None,
                         ik_backend=None):
    solve_start = time.perf_counter()
    stage = timing.target_stage(target_label)
    target_seed_start = 0 if timing_stats is None else timing_stats["seed_trials"]
    target_accepted_start = 0 if timing_stats is None else timing_stats["accepted_confs"]
    stage_stats = None
    if timing_stats is not None:
        timing_stats["target_calls"] += 1
        stage_stats = timing.stage_timing_stats(timing_stats, stage)
        stage_stats["calls"] += 1
    backend = _active_ik_backend(arm, ik_backend)
    rng = np.random.default_rng(grasping.stable_seed("ik", arm.name, np.round(tgt_pos, 4)))
    jnt_ranges = arm.arm.jnt_ranges
    ik_seed_list = list(seed_conf_list)
    if ik_seed_count > 0 and not ik_backends.is_multi_solution_backend(backend):
        ik_seed_list.extend(rng.uniform(jnt_ranges[:, 0],
                                        jnt_ranges[:, 1],
                                        size=(int(ik_seed_count), len(jnt_ranges))))
    conf_list = collect_ik_confs(arm,
                                 tgt_pos,
                                 tgt_rotmat,
                                 obstacle_list,
                                 ik_seed_list,
                                 other_robot_list,
                                 timing_stats=timing_stats,
                                 ik_backend=backend)
    elapsed = time.perf_counter() - solve_start
    if timing_stats is not None:
        timing_stats["solve_time"] += elapsed
        target_seed_trials = timing_stats["seed_trials"] - target_seed_start
        target_accepted_count = timing_stats["accepted_confs"] - target_accepted_start
        stage_stats["time"] += elapsed
        stage_stats["seed_trials"] += target_seed_trials
        stage_stats["accepted_confs"] += target_accepted_count
        timing_stats["slow_targets"].append((elapsed, stage, target_label, target_seed_trials, target_accepted_count))
    if not conf_list:
        if timing_stats is not None:
            timing_stats["target_failures"] += 1
            stage_stats["failures"] += 1
        return []
    if timing_stats is not None:
        timing_stats["target_successes"] += 1
        stage_stats["successes"] += 1
    if reference_conf is not None:
        reference_conf = np.asarray(reference_conf, dtype=float)
        conf_list = sorted(conf_list, key=lambda conf: np.linalg.norm(conf - reference_conf))
    if max_solutions is not None:
        conf_list = conf_list[:int(max_solutions)]
    if timing_stats is None or timing_stats.get("print_target_solutions", False):
        print(f"[IK target] {arm.name}: {target_label}; "
              f"accepted exact IK configs={len(conf_list)}.")
    return conf_list


def try_solve_task_conf(*args, **kwargs):
    try:
        return solve_task_conf(*args, **kwargs)
    except RuntimeError:
        return None


def _current_conf_dict(robot):
    return {arm_name: arm.get_jnt_values().copy() for arm_name, arm in robot.arm_dict.items()}


def _solved_arm_list_at(robot, task_list, conf_attr):
    solved_arm_list = []
    for solved_task in task_list:
        solved_arm = robot.arm_dict[solved_task.arm_name]
        solved_arm.goto_given_conf(getattr(solved_task, conf_attr))
        solved_arm_list.append(solved_arm)
    return solved_arm_list


def _print_ik_search_header(arm,
                            object_name,
                            candidate_pick_poses,
                            candidate_place_pose_items,
                            pick_symmetry_angles,
                            place_symmetry_angles,
                            candidate_grasp_indices,
                            pnp_cfg):
    ik_seed_count = int(pnp_cfg.get("ik_seed_count", 80))
    ik_backend = _active_ik_backend(arm, pnp_cfg.get("ik_backend"))
    is_multi_backend = ik_backends.is_multi_solution_backend(ik_backend)
    total_candidate_count = (len(candidate_pick_poses) * len(pick_symmetry_angles) *
                             len(candidate_grasp_indices) * len(candidate_place_pose_items) *
                             len(place_symmetry_angles))
    console.section(f"IK search: {arm.name} / {object_name}")
    print(f"  object pick poses: {len(candidate_pick_poses)} "
          f"(candidate object poses from YAML, not gripper grasps)")
    print(f"  object place poses: {len(candidate_place_pose_items)}")
    print(f"  symmetry rotations: pick={len(pick_symmetry_angles)}, place={len(place_symmetry_angles)}")
    print(f"  grasp records: {len(candidate_grasp_indices)} "
          f"(indices from grasp pickle: {list(candidate_grasp_indices)})")
    print(f"  grasp tuples to test: {total_candidate_count} = "
          f"{len(candidate_pick_poses)} pick poses x "
          f"{len(pick_symmetry_angles)} pick symmetries x "
          f"{len(candidate_grasp_indices)} grasp records x "
          f"{len(candidate_place_pose_items)} place poses x "
          f"{len(place_symmetry_angles)} place symmetries")
    if is_multi_backend:
        print(f"  IK backend: {ik_backend}; multi-solution call per target; "
              "random seeds are not used; "
              "stage order: pick branches -> lift -> place -> pre-place")
    else:
        print(f"  IK backend: {ik_backend}; pick random seeds={ik_seed_count}; "
              "follow stages use previous feasible conf only; "
              "stage order: pick branches -> lift -> place -> pre-place")


def build_pick_place_tasks(robot, cfg, obstacle_list=None, object_specs=None, task_specs=None):
    if object_specs is None:
        object_specs = cfgutils.object_specs(cfg)
    if task_specs is None:
        task_specs = cfgutils.task_specs(cfg)
    planning_obstacle_list = scene.make_planning_obstacle_list(robot, obstacle_list)
    model_dir = cfgutils.model_dir(cfg)
    pnp_cfg = cfg["pnp"]
    task_list = []
    robot.backup_state()
    try:
        for arm_name, task_spec in task_specs.items():
            object_name = task_spec["object_name"]
            object_spec = object_specs[object_name]
            grasp_info_list = grasping.load_grasp_info_list(model_dir, object_spec)
            arm = robot.arm_dict[arm_name]
            initial_conf = arm.get_jnt_values().copy()
            selected_task = None
            selected_transfer_delta = np.inf
            pick_failure_count = 0
            lift_failure_count = 0
            pre_place_failure_count = 0
            place_failure_count = 0
            candidate_grasp_indices = grasping.candidate_grasp_indices(task_spec, grasp_info_list)
            candidate_pick_poses = grasping.candidate_pick_poses(task_spec, object_spec)
            candidate_place_pose_items = grasping.candidate_place_pose_items(task_spec, object_spec)
            pick_symmetry_angles = grasping.candidate_pick_symmetry_angles(task_spec, object_spec)
            place_symmetry_angles = grasping.candidate_place_symmetry_angles(task_spec, object_spec)
            ik_backend = _active_ik_backend(arm, pnp_cfg.get("ik_backend"))
            total_candidate_count = (len(candidate_pick_poses) * len(pick_symmetry_angles) *
                                     len(candidate_grasp_indices) * len(candidate_place_pose_items) *
                                     len(place_symmetry_angles))
            ik_pick_seed_count = int(pnp_cfg.get("ik_seed_count", 80))
            ik_max_branch_count = int(pnp_cfg.get("ik_max_branch_count", 8))
            timing_stats = timing.new_ik_timing_stats(arm.name, object_name)
            timing_stats["print_target_solutions"] = bool(pnp_cfg.get("print_target_solutions", False))
            candidate_attempt_count = 0
            last_progress_time = time.perf_counter()

            def maybe_print_progress(force=False):
                nonlocal last_progress_time
                now = time.perf_counter()
                interval = float(pnp_cfg.get("timing_progress_interval", 10.0))
                if force or now - last_progress_time >= interval:
                    timing.print_ik_timing_progress(timing_stats,
                                                    candidate_attempt_count,
                                                    total_candidate_count,
                                                    pick_failure_count,
                                                    lift_failure_count,
                                                    pre_place_failure_count,
                                                    place_failure_count)
                    last_progress_time = now

            _print_ik_search_header(arm,
                                    object_name,
                                    candidate_pick_poses,
                                    candidate_place_pose_items,
                                    pick_symmetry_angles,
                                    place_symmetry_angles,
                                    candidate_grasp_indices,
                                    pnp_cfg)
            for pick_pose_index, raw_pick_pose in enumerate(candidate_pick_poses):
                for pick_symmetry_angle in pick_symmetry_angles:
                    pick_pose = grasping.apply_pose_symmetry(raw_pick_pose, object_spec, pick_symmetry_angle)
                    lift_pose = (pick_pose[0] + np.array([0.0, 0.0, pnp_cfg["pick_lift_height"]]), pick_pose[1])
                    for grasp_index in candidate_grasp_indices:
                        grasp_info = grasp_info_list[grasp_index]
                        jaw_width = float(np.clip(grasp_info[0],
                                                  arm.hnd.jaw_range[0],
                                                  arm.hnd.jaw_range[1]))
                        pick_tcp_pos, pick_tcp_rotmat = grasping.grasp_tcp_pose(pick_pose, grasp_info)
                        lift_tcp_pos, lift_tcp_rotmat = grasping.grasp_tcp_pose(lift_pose, grasp_info)
                        pick_other_robot_list = _solved_arm_list_at(robot, task_list, "pick_conf")
                        target_suffix = (f"pick pose #{pick_pose_index}, "
                                         f"pick symmetry {np.degrees(pick_symmetry_angle):.1f} deg, "
                                         f"grasp #{grasp_index}")
                        pick_conf_list = solve_task_conf_list(
                            arm,
                            pick_tcp_pos,
                            pick_tcp_rotmat,
                            planning_obstacle_list,
                            [arm.arm.home_conf, initial_conf],
                            other_robot_list=pick_other_robot_list,
                            ik_seed_count=ik_pick_seed_count,
                            target_label=f"{object_name} pick {target_suffix}",
                            reference_conf=initial_conf,
                            max_solutions=ik_max_branch_count,
                            timing_stats=timing_stats,
                            ik_backend=ik_backend)
                        if not pick_conf_list:
                            skipped_count = len(candidate_place_pose_items) * len(place_symmetry_angles)
                            candidate_attempt_count += skipped_count
                            pick_failure_count += skipped_count
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue

                        lift_other_robot_list = _solved_arm_list_at(robot, task_list, "lift_conf")
                        lift_branch_list = []
                        for pick_branch_id, pick_conf in enumerate(pick_conf_list):
                            lift_conf_list = solve_task_conf_list(
                                arm,
                                lift_tcp_pos,
                                lift_tcp_rotmat,
                                planning_obstacle_list,
                                [pick_conf],
                                other_robot_list=lift_other_robot_list,
                                ik_seed_count=0,
                                target_label=f"{object_name} lift {target_suffix}, pick branch #{pick_branch_id}",
                                reference_conf=pick_conf,
                                max_solutions=1,
                                timing_stats=timing_stats,
                                ik_backend=ik_backend)
                            if not lift_conf_list:
                                arm.goto_given_conf(initial_conf)
                                continue
                            lift_branch_list.append((pick_branch_id, pick_conf, lift_conf_list[0]))

                        if not lift_branch_list:
                            skipped_count = len(candidate_place_pose_items) * len(place_symmetry_angles)
                            candidate_attempt_count += skipped_count
                            lift_failure_count += skipped_count
                            arm.goto_given_conf(initial_conf)
                            maybe_print_progress()
                            continue

                        for place_pose_index, raw_place_pose in candidate_place_pose_items:
                            for place_symmetry_angle in place_symmetry_angles:
                                candidate_attempt_count += 1
                                place_pose = grasping.apply_pose_symmetry(raw_place_pose,
                                                                          object_spec,
                                                                          place_symmetry_angle)
                                pre_place_pose = (
                                    place_pose[0] + np.array([0.0, 0.0, pnp_cfg["place_approach_distance"]]),
                                    place_pose[1])
                                place_tcp_pos, place_tcp_rotmat = grasping.grasp_tcp_pose(place_pose, grasp_info)
                                pre_place_tcp_pos, pre_place_tcp_rotmat = grasping.grasp_tcp_pose(pre_place_pose,
                                                                                                  grasp_info)
                                tuple_has_place = False
                                tuple_has_pre_place = False
                                for pick_branch_id, pick_conf, lift_conf in lift_branch_list:
                                    place_label = (f"{target_suffix}, place pose #{place_pose_index}, "
                                                   f"place symmetry {np.degrees(place_symmetry_angle):.1f} deg, "
                                                   f"pick branch #{pick_branch_id}")
                                    place_conf_list = solve_task_conf_list(
                                        arm,
                                        place_tcp_pos,
                                        place_tcp_rotmat,
                                        planning_obstacle_list,
                                        [lift_conf],
                                        other_robot_list=_solved_arm_list_at(robot, task_list, "place_conf"),
                                        ik_seed_count=0,
                                        target_label=f"{object_name} place {place_label}",
                                        reference_conf=lift_conf,
                                        max_solutions=1,
                                        timing_stats=timing_stats,
                                        ik_backend=ik_backend)
                                    if not place_conf_list:
                                        arm.goto_given_conf(initial_conf)
                                        continue
                                    tuple_has_place = True
                                    place_conf = place_conf_list[0]
                                    place_solution_type = "nearest_exact_ik"

                                    pre_place_conf_list = solve_task_conf_list(
                                        arm,
                                        pre_place_tcp_pos,
                                        pre_place_tcp_rotmat,
                                        planning_obstacle_list,
                                        [place_conf],
                                        other_robot_list=_solved_arm_list_at(robot, task_list, "pre_place_conf"),
                                        ik_seed_count=0,
                                        target_label=f"{object_name} pre-place {place_label}",
                                        reference_conf=place_conf,
                                        max_solutions=1,
                                        timing_stats=timing_stats,
                                        ik_backend=ik_backend)
                                    if not pre_place_conf_list:
                                        arm.goto_given_conf(initial_conf)
                                        continue
                                    tuple_has_pre_place = True
                                    pre_place_conf = pre_place_conf_list[0]
                                    pre_place_solution_type = "nearest_exact_ik"
                                    lift_solution_type = "nearest_exact_ik"

                                    arm.goto_given_conf(pick_conf)
                                    payload_rel_pose = arm.cvt_gl_pose_to_tcp(pick_pose[0], pick_pose[1])
                                    candidate_task = PickPlaceTask(arm_name=arm_name,
                                                                   object_name=object_name,
                                                                   grasp_index=grasp_index,
                                                                   pick_pose_index=pick_pose_index,
                                                                   symmetry_angle=pick_symmetry_angle,
                                                                   place_pose_index=place_pose_index,
                                                                   place_symmetry_angle=place_symmetry_angle,
                                                                   pick_conf=pick_conf,
                                                                   lift_conf=lift_conf,
                                                                   pre_place_conf=pre_place_conf,
                                                                   place_conf=place_conf,
                                                                   jaw_width=jaw_width,
                                                                   pick_pose=pick_pose,
                                                                   lift_pose=lift_pose,
                                                                   pre_place_pose=pre_place_pose,
                                                                   place_pose=place_pose,
                                                                   payload_rel_pose=payload_rel_pose,
                                                                   pick_solution_type="exact_ik_branch",
                                                                   lift_solution_type=lift_solution_type,
                                                                   pre_place_solution_type=pre_place_solution_type,
                                                                   place_solution_type=place_solution_type)
                                    transfer_delta = (np.linalg.norm(lift_conf - pick_conf) +
                                                      np.linalg.norm(pre_place_conf - lift_conf) +
                                                      np.linalg.norm(place_conf - pre_place_conf))
                                    console.success(
                                        f"{arm.name}: feasible {object_name} pick pose #{pick_pose_index}, "
                                        f"pick symmetry {np.degrees(pick_symmetry_angle):.1f} deg, "
                                        f"grasp #{grasp_index}, place pose #{place_pose_index}, "
                                        f"place symmetry {np.degrees(place_symmetry_angle):.1f} deg; "
                                        f"pick-lift-pre-place-place joint_delta={transfer_delta:.3f}.")
                                    if transfer_delta < selected_transfer_delta:
                                        selected_task = candidate_task
                                        selected_transfer_delta = transfer_delta
                                    arm.goto_given_conf(initial_conf)
                                if not tuple_has_place:
                                    place_failure_count += 1
                                elif not tuple_has_pre_place:
                                    pre_place_failure_count += 1
                                maybe_print_progress()
            if selected_task is None:
                arm.goto_given_conf(initial_conf)
                maybe_print_progress(force=True)
                timing.print_ik_timing_summary(timing_stats, total_candidate_count)
                raise PickPlacePlanningError(
                    f"No collision-free exact IK grasp found for {arm.name} {object_name}; "
                    f"tried {total_candidate_count} candidates "
                    f"({len(candidate_pick_poses)} pick poses x {len(pick_symmetry_angles)} pick symmetries x "
                    f"{len(candidate_grasp_indices)} grasps x {len(candidate_place_pose_items)} place poses x "
                    f"{len(place_symmetry_angles)} place symmetries) "
                    f"({pick_failure_count} failed at pick, {lift_failure_count} failed at lift, "
                    f"{place_failure_count} failed at place, {pre_place_failure_count} failed at pre-place).",
                    conf_dict=_current_conf_dict(robot),
                    arm_name=arm_name,
                    object_name=object_name,
                    grasp_indices=candidate_grasp_indices)
            console.success(f"{arm.name}: selected {object_name} grasp #{selected_task.grasp_index} "
                            f"at pick pose #{selected_task.pick_pose_index}, "
                            f"pick symmetry {np.degrees(selected_task.symmetry_angle):.1f} deg, "
                            f"place pose #{selected_task.place_pose_index}, "
                            f"place symmetry {np.degrees(selected_task.place_symmetry_angle):.1f} deg "
                            f"from {total_candidate_count} candidates; "
                            f"best pick-lift-pre-place-place joint_delta={selected_transfer_delta:.3f}.")
            maybe_print_progress(force=True)
            timing.print_ik_timing_summary(timing_stats, total_candidate_count, selected_task)
            task_list.append(selected_task)
    finally:
        robot.restore_state()
    return task_list
