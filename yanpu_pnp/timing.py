import time

import numpy as np

from yanpu_pnp import console


SLOW_TARGET_COUNT = 8


def new_ik_timing_stats(arm_name, object_name):
    return {
        "arm_name": arm_name,
        "object_name": object_name,
        "wall_start": time.perf_counter(),
        "target_calls": 0,
        "target_successes": 0,
        "target_failures": 0,
        "seed_trials": 0,
        "ik_none": 0,
        "collision_rejects": 0,
        "pose_rejects": 0,
        "duplicate_rejects": 0,
        "accepted_confs": 0,
        "ik_time": 0.0,
        "fk_time": 0.0,
        "collision_time": 0.0,
        "pose_check_time": 0.0,
        "solve_time": 0.0,
        "stage_stats": {},
        "slow_targets": [],
    }


def target_stage(target_label):
    for stage in ("pre-place", "place", "lift", "pick"):
        if f" {stage} " in f" {target_label} ":
            return stage
    return "target"


def stage_timing_stats(timing_stats, stage):
    return timing_stats["stage_stats"].setdefault(stage, {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "time": 0.0,
        "seed_trials": 0,
        "accepted_confs": 0,
    })


def fmt_seconds(seconds):
    return f"{seconds:.3f}s"


def _safe_rate(total, count):
    if count <= 0:
        return 0.0
    return total / count


def _collision_check_count(timing_stats):
    return max(0,
               timing_stats["collision_rejects"] +
               timing_stats["pose_rejects"] +
               timing_stats["duplicate_rejects"] +
               timing_stats["accepted_confs"])


def print_ik_timing_progress(timing_stats,
                             candidate_attempt_count,
                             total_candidate_count,
                             pick_failure_count,
                             lift_failure_count,
                             pre_place_failure_count,
                             place_failure_count):
    elapsed = time.perf_counter() - timing_stats["wall_start"]
    avg_trial_per_target = _safe_rate(timing_stats["seed_trials"], timing_stats["target_calls"])
    avg_ik = _safe_rate(timing_stats["ik_time"], timing_stats["seed_trials"])
    collision_checks = _collision_check_count(timing_stats)
    avg_collision = _safe_rate(timing_stats["collision_time"], collision_checks)
    console.section(f"IK progress: {timing_stats['arm_name']} / {timing_stats['object_name']}")
    print(f"  grasp tuples tried: {candidate_attempt_count}/{total_candidate_count}; "
          f"stage failures pick/lift/place/pre-place="
          f"{pick_failure_count}/{lift_failure_count}/{place_failure_count}/{pre_place_failure_count}; "
          f"elapsed={fmt_seconds(elapsed)}")
    print(f"  targets={timing_stats['target_calls']}, IK trials={timing_stats['seed_trials']} "
          f"(avg {avg_trial_per_target:.1f}/target); "
          f"IK={fmt_seconds(timing_stats['ik_time'])} "
          f"(avg {fmt_seconds(avg_ik)}/trial); "
          f"collision={fmt_seconds(timing_stats['collision_time'])} "
          f"(checks={collision_checks}, avg {fmt_seconds(avg_collision)}/check)")


def print_ik_timing_summary(timing_stats, total_candidate_count, selected_task=None):
    elapsed = time.perf_counter() - timing_stats["wall_start"]
    avg_trial_per_target = _safe_rate(timing_stats["seed_trials"], timing_stats["target_calls"])
    avg_ik = _safe_rate(timing_stats["ik_time"], timing_stats["seed_trials"])
    collision_checks = _collision_check_count(timing_stats)
    avg_collision = _safe_rate(timing_stats["collision_time"], collision_checks)
    console.section(f"IK summary: {timing_stats['arm_name']} / {timing_stats['object_name']}")
    print(f"  grasp tuples: {total_candidate_count}; target poses solved: "
          f"{timing_stats['target_successes']}/{timing_stats['target_calls']} "
          f"(failures={timing_stats['target_failures']}); elapsed={fmt_seconds(elapsed)}")
    print(f"  IK trials: {timing_stats['seed_trials']} "
          f"(avg {avg_trial_per_target:.1f}/target)")
    print("  time:")
    print(f"    IK={fmt_seconds(timing_stats['ik_time'])} "
          f"(avg {fmt_seconds(avg_ik)}/trial)")
    print(f"    collision={fmt_seconds(timing_stats['collision_time'])} "
          f"(checks={collision_checks}, avg {fmt_seconds(avg_collision)}/check)")
    print(f"    fk={fmt_seconds(timing_stats['fk_time'])}, "
          f"pose_check={fmt_seconds(timing_stats['pose_check_time'])}, "
          f"solve_total={fmt_seconds(timing_stats['solve_time'])}")
    print("  rejects:")
    print("    "
          f"ik_empty={timing_stats['ik_none']}, "
          f"collision={timing_stats['collision_rejects']}, "
          f"pose_error={timing_stats['pose_rejects']}, "
          f"duplicate={timing_stats['duplicate_rejects']}, "
          f"accepted_confs={timing_stats['accepted_confs']}.")
    print("  stages:")
    for stage, stage_stats in timing_stats["stage_stats"].items():
        print(f"    {stage}: calls={stage_stats['calls']}, "
              f"successes={stage_stats['successes']}, failures={stage_stats['failures']}, "
              f"IK_trials={stage_stats['seed_trials']}, accepted_confs={stage_stats['accepted_confs']}, "
              f"time={fmt_seconds(stage_stats['time'])}.")
    slow_targets = sorted(timing_stats["slow_targets"], reverse=True)[:SLOW_TARGET_COUNT]
    if slow_targets:
        print("  slowest targets:")
        for elapsed_target, stage, label, seed_trials, accepted_count in slow_targets:
            print(f"    {fmt_seconds(elapsed_target)} {stage}: {label} "
                  f"(IK_trials={seed_trials}, accepted_confs={accepted_count})")
    if selected_task is not None:
        console.success(f"  selected: pick_pose=#{selected_task.pick_pose_index}, "
                        f"pick_symmetry={np.degrees(selected_task.symmetry_angle):.1f}deg, "
                        f"grasp=#{selected_task.grasp_index}, "
                        f"place_pose=#{selected_task.place_pose_index}, "
                        f"place_symmetry={np.degrees(selected_task.place_symmetry_angle):.1f}deg.")
