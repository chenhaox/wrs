import argparse

from trac_ik import TracIK as _TracIK

from yanpu_pnp import console
from yanpu_pnp import search_result
from yanpu_pnp import show_robot


def selected_record(record, result_index=0, candidate_index=None):
    if "results" not in record:
        return record
    results = record.get("results", [])
    if not results:
        raise RuntimeError("The all-results YAML does not contain any feasible result.")
    if candidate_index is not None:
        for result in results:
            if int(result["candidate_index"]) == int(candidate_index):
                selected = result
                break
        else:
            raise RuntimeError(f"candidate_index {candidate_index} is not stored in this result file.")
    else:
        if result_index < 0 or result_index >= len(results):
            raise RuntimeError(f"result_index {result_index} out of range [0, {len(results) - 1}].")
        selected = results[result_index]
    single = dict(selected)
    single["source"] = record.get("source", {})
    if "base_config_snapshot" in record:
        single["base_config_snapshot"] = record["base_config_snapshot"]
    single["task_overrides"] = record.get("task_overrides", {})
    return single


def print_result_summary(record, result_path):
    search = record.get("search", {})
    candidate = record.get("candidate", {})
    console.section("Rack search result file")
    console.key_value("result", result_path)
    console.key_value("TracIK loaded before Panda3D/WRS",
                      f"{bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__})",
                      color=console.Fore.GREEN if _TracIK else console.Fore.RED)
    console.key_value("candidate_index", search.get("candidate_index"))
    if "candidate_index" in record:
        console.key_value("selected_candidate_index", record["candidate_index"], color=console.Fore.GREEN)
    console.key_value("completed_candidates", search.get("completed_candidates"))
    console.key_value("score", search.get("score"))
    if "score" in record:
        console.key_value("selected_score", record["score"], color=console.Fore.GREEN)
    console.key_value("frame_count", search.get("frame_count"))
    if "frame_count" in record:
        console.key_value("selected_frame_count", record["frame_count"])
    console.key_value("rack.base_pos", candidate.get("rack_base_pos"))
    console.key_value("rack.yaw_deg", candidate.get("rack_yaw_deg"))
    console.key_value("rack.arm_distance", candidate.get("arm_distance"))
    console.key_value("rack.left_arm_mount_euler_deg", candidate.get("left_arm_mount_euler_deg"))
    console.key_value("task_overrides", record.get("task_overrides", "<none>"))


def main():
    parser = argparse.ArgumentParser(description="Display a robot/environment configuration saved by rack search.")
    parser.add_argument("--result", default=search_result.DEFAULT_RESULT_PATH)
    parser.add_argument("--result-index", type=int, default=0,
                        help="For all-results YAML, display this 0-based result index after sorting by score.")
    parser.add_argument("--candidate-index", type=int, default=None,
                        help="For all-results YAML, display this exact rack candidate index.")
    args = parser.parse_args()
    _ = _TracIK

    result_path = search_result.resolve_result_path(args.result)
    record = selected_record(search_result.load_result(result_path),
                             result_index=args.result_index,
                             candidate_index=args.candidate_index)
    print_result_summary(record, result_path)
    cfg = search_result.config_from_result(record)
    show_robot.run(cfg)


if __name__ == "__main__":
    main()
