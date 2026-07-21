# -*- coding: utf-8 -*-
"""
Replay a planned tower motion .pkl on the real Panthera-HT arm(s).

This is the "run on hardware" entry point that pairs with
``execute_layout_sequence_visual.py`` (which produces the .pkl).  It reads
the per-arm joint trajectories from the .pkl and plays them back through
:meth:`FafuRobotController.move_jntspace_path` (gripper open/close handled
at width-change boundaries by ``run_motion_pkl`` / ``replay_dual_arm_motion_pkl``).

IMPORTANT
---------
* Use the Python 3.10 env that matches the cp310 driver, e.g.::

      D:\\Soft\\tools\\anaconda\\envs\\spatialvla\\python.exe -m sealp.examples.motion.replay_tower_on_robot --help

* It is **dry-run by default** (prints the plan, does NOT move).  Add
  ``--run`` to actually command the hardware.
* First real run: pass ``--home`` and a low ``--speed`` and verify each
  joint's zero / direction matches the simulation before trusting a full
  replay.

Examples
--------
# 0) offline plan check (no hardware needed for parsing, but constructing a
#    controller opens the serial port -- so this still needs the arm):
python -m sealp.examples.motion.replay_tower_on_robot --arm lft

# 1) single left arm, low speed, go home first, then actually move:
python -m sealp.examples.motion.replay_tower_on_robot --arm lft --home --speed 12 --run

# 2) dual arm (two boards / two serial ports):
python -m sealp.examples.motion.replay_tower_on_robot --arm dual \
    --port-lft COM5 --port-rgt COM6 --speed 12 --run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# Make the repo root importable when run as a file (python path/to/this.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import (  # noqa: E402
    FafuRobotController,
    load_motion_pkl,
    extract_arm_segments,
    replay_dual_arm_motion_pkl,
    apply_jaw_close_delta,
    should_home_after_part,
    _resolve_home_between,
    _split_path_by_gripper,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PKL = os.path.join(_HERE, "_output", "tower_optimal_initial_motions.pkl")
_DEFAULT_CFG = os.path.join(
    _REPO_ROOT, "wrs", "robot_con", "panthera_ht", "robot.cfg"
)


def _build_arm(cfg_path: str, port: str | None, gripper_id: int | None):
    return FafuRobotController(
        cfg_path=cfg_path,
        port=port,
        has_gripper=gripper_id is not None,
        gripper_motor_id=gripper_id,
    )


def _fmt_deg(jv) -> str:
    return "[" + ", ".join(f"{math.degrees(v):.1f}" for v in jv) + "]"


def _key_poses_from_seg(seg, thresh: float = 0.003):
    """Extract the grasp ('抓取位') and release ('放置位') key frames of a step.

    Returns ``(label, jv_frame, gripper_action, jaw_width_m)`` 列表。
    jaw_width_m 来自 pkl ev_list, 真机回放时按此宽度开合而非全开/全关。
    """
    jv, ev = seg["jv"], seg["ev"]
    poses = []
    for sp in _split_path_by_gripper(jv, ev, thresh):
        jw = sp.get("jaw_width")
        if sp["gripper"] == "close":
            poses.append(("抓取位(去拿)", sp["path"][-1], "close", jw))
        elif sp["gripper"] == "open":
            poses.append(("放置位(放好)", sp["path"][-1], "open", jw))
    if not poses:
        poses.append(("末帧位", jv[-1], None, None))
    return poses


def _parse_home_between(spec):
    """把 --home-between 字符串解析为 run_motion_pkl 能用的值。

    "all" -> 每个零件后都回 home(默认); "none"/"" -> 不回; 否则按逗号拆成零件 id 列表。
    """
    if spec is None:
        return "all"
    s = str(spec).strip().lower()
    if s in ("", "none", "off", "0", "false"):
        return None
    if s == "all":
        return "all"
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def _make_tcp_lifter(lift_m: float, *, tag: str = "lift"):
    """构造 jv->jv: 把关节角对应 TCP 在世界 z 方向抬高 lift_m (FK->抬高->IK)。"""
    if not lift_m:
        return None
    try:
        from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm
        robot = PantheraHTSglArm(enable_cc=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] 无法加载仿真模型, 跳过补偿 ({exc})")
        return None
    up = np.array([0.0, 0.0, float(lift_m)])
    state = {"fail": 0, "ok": 0}

    def _lift(jv, seed=None):
        jv = np.asarray(jv, dtype=float)
        try:
            pos, rotmat = robot.fk(jnt_values=jv, toggle_jacobian=False, update=False)
        except TypeError:
            robot.goto_given_conf(jnt_values=jv)
            pos, rotmat = robot.gl_tcp_pos, robot.gl_tcp_rotmat
        sol = robot.ik(tgt_pos=pos + up, tgt_rotmat=rotmat,
                       seed_jnt_values=seed if seed is not None else jv)
        if sol is None:
            state["fail"] += 1
            return jv
        state["ok"] += 1
        return np.asarray(sol, dtype=float)

    _lift.state = state
    return _lift


def _apply_arm_lift(payload: dict, arm_side: str, lift_m: float) -> int:
    """就地把 payload 中指定臂所有帧的关节角对应 TCP 抬高 lift_m。返回处理帧数。"""
    arm_cn = {"lft": "左臂", "rgt": "右臂"}.get(arm_side, arm_side)
    tag = f"{arm_side}-lift"
    lifter = _make_tcp_lifter(lift_m, tag=tag)
    if lifter is None:
        return 0
    nframes = 0
    for st in payload.get("steps", []):
        for seg in st.get("segments", []):
            if seg.get("arm_side") != arm_side:
                continue
            jv_list = seg.get("jv_list") or []
            seed = None
            new_list = []
            for fr in jv_list:
                nv = lifter(fr, seed=seed)
                seed = nv
                new_list.append(nv)
                nframes += 1
            seg["jv_list"] = new_list
    fails = getattr(lifter, "state", {}).get("fail", 0)
    print(f"[{tag}] {arm_cn} TCP 抬高 {lift_m * 1000:.0f}mm: 处理 {nframes} 帧"
          + (f", 其中 {fails} 帧 IK 失败已保留原值" if fails else ""))
    return nframes


def _do_key_poses(ctrl, seg, arm_side: str, *, speed: int, dry_run: bool,
                  actuate_gripper: bool, pose_pause: float = 1.0) -> None:
    do_grip = actuate_gripper and getattr(ctrl, "has_gripper", False) and not dry_run
    for label, jv_frame, grip, jaw_w in _key_poses_from_seg(seg):
        jaw_txt = ""
        if grip and jaw_w is not None and np.isfinite(jaw_w):
            jaw_txt = f"  jaw={float(jaw_w) * 1000:.1f}mm"
        print(f"      [{arm_side}] {label} 关节角(度)={_fmt_deg(jv_frame)}"
              + (f"  ->  夹爪{'闭合(抓住)' if grip == 'close' else '张开(放下)'}{jaw_txt}"
                 if grip else ""))
        if dry_run:
            continue
        # 逐点阻塞式: 先走到关键位, 到位后再动夹爪, 然后停顿一下(便于观察 / 让夹爪稳)。
        ctrl.move_j(jv_frame, is_radians=True, speed=speed, block=True)
        if do_grip and grip in ("close", "open"):
            ctrl.actuate_gripper_from_pkl(
                grip, jaw_w, part_id=seg.get("part_id", ""), arm_side=arm_side,
            )
        if pose_pause > 0:
            time.sleep(pose_pause)


def _key_frames_single(arm, payload, arm_side: str, *, speed: int, dry_run: bool,
                       actuate_gripper: bool, pose_pause: float = 1.0,
                       home_between="all", home_between_speed: int = 25) -> None:
    """逐点回放单臂关键帧。

    默认每个零件完成后先回 home 再去下一个零件(``home_between="all"``)。
    """
    segs = extract_arm_segments(payload, arm_side)
    if not segs:
        print(f"[key-frames] arm={arm_side}: 无该臂步骤, 跳过。")
        return
    home_all, home_set = _resolve_home_between(home_between)
    mode_txt = (" (每个零件经 home 往返)" if home_all
                else (f" (仅 {sorted(home_set)} 后回 home)" if home_set
                      else " (零件间直接衔接, 不回 home)"))
    print(f"[key-frames] arm={arm_side}: {len(segs)} 步, 每步 move_j 到 抓取位/放置位{mode_txt}"
          + ("  [DRY-RUN]" if dry_run else ""))
    n = len(segs)
    if home_all and not dry_run:
        print(f"  [{arm_side}] 先回到 home (speed={home_between_speed}) ...")
        arm.go_home(speed=home_between_speed)
    for i, seg in enumerate(segs, 1):
        pid = seg["part_id"]
        do_home = should_home_after_part(pid, home_between)
        flow = "  [抓取(夹)->放置(松)->回 home]" if do_home else ""
        print(f"  [{arm_side}] 第 {i}/{n} 步 零件[{pid}]{flow}")
        _do_key_poses(arm, seg, arm_side, speed=speed, dry_run=dry_run,
                      actuate_gripper=actuate_gripper, pose_pause=pose_pause)
        if i < n:
            nxt = segs[i]["part_id"]
            if pose_pause > 0 and not dry_run:
                via = "先回 home 再去" if do_home else "直接前往(不回 home)"
                print(f"  [{arm_side}] —— 零件 [{pid}] 完成, 停顿 {pose_pause:.1f}s "
                      f"后{via}下一个零件 [{nxt}] ——")
                time.sleep(pose_pause)
            if do_home and not dry_run:
                print(f"  [{arm_side}] 零件 [{pid}] 完成, 回到 home (speed={home_between_speed}) ...")
                arm.go_home(speed=home_between_speed)
            elif do_home:
                print(f"  [{arm_side}] (dry-run) 零件 [{pid}] 后将回 home 再去 [{nxt}]")


def _key_frames_dual(lft, rgt, payload, *, speed: int, dry_run: bool,
                     actuate_gripper: bool, step_pause: float = 1.0,
                     home_between="all", home_between_speed: int = 25) -> None:
    """Step-ordered dual-arm: move_j to each step's grasp pose then place pose."""
    ctrls = {"lft": lft, "rgt": rgt}
    steps = payload.get("steps", [])
    home_all, home_set = _resolve_home_between(home_between)
    mode_txt = (" (每个零件经 home 往返)" if home_all
                else (f" (仅 {sorted(home_set)} 后回 home)" if home_set
                      else " (零件间直接衔接, 不回 home)"))
    print(f"[key-frames] 双臂: {len(steps)} 步, 每步 move_j 到 抓取位/放置位{mode_txt}"
          + ("  [DRY-RUN]" if dry_run else ""))
    for idx, st in enumerate(steps, 1):
        sides = st.get("arm_sides") or [s.get("arm_side") for s in st.get("segments", [])]
        distinct = [s for i, s in enumerate(sides) if s and s not in sides[:i]]
        pid = st.get("part_id", "")
        if st.get("handover") or len(distinct) > 1:
            print(f"  [第 {idx}/{len(steps)} 步] 零件[{pid}] 真换手, 跳过。")
            continue
        side = distinct[0] if distinct else None
        ctrl = ctrls.get(side)
        arm_name = {"lft": "左臂", "rgt": "右臂"}.get(side, side)
        if ctrl is None:
            print(f"  [第 {idx}/{len(steps)} 步] 零件[{pid}] arm={side} 未绑定控制器, 跳过。")
            continue
        ex = extract_arm_segments({"steps": [st]}, side)
        if not ex:
            continue
        print(f"  [第 {idx}/{len(steps)} 步] {arm_name}({side}) 零件[{pid}]")
        _do_key_poses(ctrl, ex[-1], side, speed=speed, dry_run=dry_run,
                      actuate_gripper=actuate_gripper)
        if idx < len(steps):
            nxt_pid = steps[idx].get("part_id", "")
            do_home = should_home_after_part(pid, home_between)
            if step_pause > 0 and not dry_run:
                via = "先回 home 再去" if do_home else "直接前往(不回 home)"
                print(f"  —— 零件 [{pid}] 完成, 停顿 {step_pause:.1f}s 后{via}下一步 [{nxt_pid}] ——")
                time.sleep(step_pause)
            if do_home and not dry_run:
                print(f"  [{side}] 零件 [{pid}] 完成, 回到 home (speed={home_between_speed}) ...")
                ctrl.go_home(speed=home_between_speed)
            elif do_home:
                print(f"  [{side}] (dry-run) 零件 [{pid}] 后将回 home 再去 [{nxt_pid}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a tower motion .pkl on the real Panthera-HT arm(s).")
    ap.add_argument("--pkl", default=_DEFAULT_PKL, help="path to the motion .pkl")
    ap.add_argument("--arm", choices=["lft", "rgt", "dual"], default="lft",
                    help="which arm(s) to drive (default: lft)")
    ap.add_argument("--cfg", default=_DEFAULT_CFG, help="robot.cfg (single-arm or shared)")
    ap.add_argument("--cfg-lft", default=None, help="override cfg for left arm (dual)")
    ap.add_argument("--cfg-rgt", default=None, help="override cfg for right arm (dual)")
    ap.add_argument("--port", default=None, help="serial port for single-arm (else cfg/auto)")
    ap.add_argument("--port-lft", default=None, help="serial port for left arm (dual)")
    ap.add_argument("--port-rgt", default=None, help="serial port for right arm (dual)")
    ap.add_argument("--gripper-id", type=int, default=7, help="gripper motor id (default 7; -1 = no gripper)")
    ap.add_argument("--speed", type=int, default=12, help="speed percent forwarded to move_jntspace_path")
    ap.add_argument("--home", action="store_true", help="go_home (low speed) before replay")
    ap.add_argument("--home-only", action="store_true",
                    help="only go_home then exit (no trajectory replay). "
                         "Use with --run to identify which physical arm is on a "
                         "port and to verify joint zeros/direction before a full replay.")
    ap.add_argument("--home-end", action="store_true", default=True,
                    help="全部零件回放结束后回 home(默认开启, 如放完 top_cross 后回零位)。"
                         "断连前执行, 避免停在最后一个放置位。")
    ap.add_argument("--no-home-end", action="store_false", dest="home_end",
                    help="回放结束后不回 home, 停在最后一个放置位。")
    ap.add_argument("--end-release", choices=["stop", "brake", "hold"], default="brake",
                    help="joint state when disconnecting: stop=free/limp (sags), "
                         "brake=damped hold w/o current (default), hold=actively "
                         "energised hold. Use brake/hold to avoid the arm dropping "
                         "after '[replay] done.'.")
    ap.add_argument("--key-frames", action="store_true",
                    help="point-to-point mode: for each step, skip the dense path and "
                         "just move_j (blocking S-curve) to the GRASP pose (where the "
                         "jaw closes) and then the PLACE pose (where it opens), "
                         "actuating the gripper at each. Prints both poses in degrees "
                         "for sim<->real comparison. Use --no-gripper-action to skip "
                         "the gripper.")
    ap.add_argument("--step-pause", type=float, default=1.0,
                    help="拿完一个零件后、前往下一个零件前的停顿秒数(默认会先回 home)。"
                         "调大更便于肉眼区分每个零件。默认 1.0s, 设 0 关闭。"
                         "(--key-frames 模式下作为每个关键位/零件的停顿秒数)")
    ap.add_argument("--follow-path", action="store_true",
                    help="完整轨迹回放沿 pkl 规划路径逐航点阻塞 move_j(每点都停, 最稳但顿挫)。"
                         "一般不需要; 默认的流式回放已是段内连续、只在取放点停。")
    ap.add_argument("--waypoint-stride", type=int, default=2,
                    help="--follow-path 时每隔几帧取一个阻塞航点(末帧必留)。默认 2。")
    ap.add_argument("--key-pause", type=float, default=0.6,
                    help="完整轨迹回放: 每个取/放关键点动完夹爪后停顿的秒数(段内连续不停)。默认 0.6。")
    ap.add_argument("--path-dt", type=float, default=0.05,
                    help="完整轨迹流式回放的帧间隔(秒)。调大整体更慢更稳, 调小更快。默认 0.05。")
    ap.add_argument("--home-between", default="all",
                    help="左右手零件间是否回 home。默认 'all'(每个零件后都回 home)。"
                         "可填 'none' 关闭, 或逗号分隔的零件 id 仅在这些零件后回 home, "
                         "如 'middle_plate' / 'middle_plate,post_fr'。")
    ap.add_argument("--home-between-speed", type=int, default=25,
                    help="--home-between 时回 home 的速度百分比。默认 25。")
    ap.add_argument("--lft-lift", type=float, default=0.0,
                    help="左臂标定补偿: 把左臂所有关键位/轨迹的末端在世界 z 方向抬高这么多米"
                         "(FK->抬高->IK)。只影响左臂(--arm lft 或 dual)。"
                         "例: --lft-lift 0.05 表示抬高 5cm。默认 0(不补偿)。")
    ap.add_argument("--rgt-lift", type=float, default=0.0,
                    help="右臂标定补偿: 把右臂所有关键位/轨迹的末端在世界 z 方向抬高这么多米"
                         "(FK->抬高->IK), 用于补偿'右臂真机实际比仿真偏低'导致的碰桌。"
                         "只影响右臂(--arm rgt 或 dual)。默认 0(不补偿); 先从 0.02 试。")
    ap.add_argument("--jaw-close-delta", type=float, default=0.0,
                    help="运行时再把 pkl 里夹持宽度(非张开位)加上此增量(米)。"
                         "默认 0(直接用 pkl 原值); 若 pkl 未用 adjust_pkl_jaw_close 收紧, "
                         "可填 -0.01 临时再紧 1cm。")
    ap.add_argument("--run", action="store_true", help="actually move (default: dry-run only)")
    ap.add_argument("--no-gripper-action", action="store_true",
                    help="do not open/close the gripper during replay")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        ap.error(f"pkl not found: {args.pkl}")
    gripper_id = None if args.gripper_id is not None and args.gripper_id < 0 else args.gripper_id
    dry_run = not args.run

    # Quick offline summary of what's in the pkl (no hardware needed).
    payload = load_motion_pkl(args.pkl)
    print(f"[replay] pkl = {args.pkl}")

    if args.jaw_close_delta:
        n_jaw = apply_jaw_close_delta(payload, args.jaw_close_delta)
        print(f"[replay] jaw-close-delta={args.jaw_close_delta * 1000:+.1f}mm: "
              f"调整 {n_jaw} 帧夹持宽度")

    # 左右臂抬高补偿(在内存里就地修改 payload, 对 key-frames 与完整轨迹回放都生效)。
    if args.lft_lift and args.arm in ("lft", "dual"):
        _apply_arm_lift(payload, "lft", args.lft_lift)
    elif args.lft_lift:
        print(f"[lft-lift] --arm={args.arm} 不含左臂, 忽略 --lft-lift。")
    if args.rgt_lift and args.arm in ("rgt", "dual"):
        _apply_arm_lift(payload, "rgt", args.rgt_lift)
    elif args.rgt_lift:
        print(f"[rgt-lift] --arm={args.arm} 不含右臂, 忽略 --rgt-lift。")
    for side in ("lft", "rgt"):
        segs = extract_arm_segments(payload, side)
        if segs:
            print(f"[replay] {side}: " + ", ".join(
                f"step{s['step_id']}:{s['part_id']}({s['jv'].shape[0]}f)" for s in segs))

    if dry_run:
        print("[replay] DRY-RUN (no hardware command). Add --run to move.")

    home_speed = min(args.speed, 12)
    home_between = _parse_home_between(args.home_between)

    if args.arm == "dual":
        lft = _build_arm(args.cfg_lft or args.cfg, args.port_lft, gripper_id)
        rgt = _build_arm(args.cfg_rgt or args.cfg, args.port_rgt, gripper_id)
        if args.home or args.home_only:
            if dry_run:
                print(f"[replay] (dry-run) would go_home both arms @ speed={home_speed}")
            else:
                print(f"[replay] go_home both arms @ speed={home_speed}")
                lft.go_home(speed=home_speed)
                rgt.go_home(speed=home_speed)
        if args.home_only:
            print("[replay] --home-only: skipping trajectory replay.")
        elif args.key_frames:
            _key_frames_dual(lft, rgt, payload, speed=args.speed, dry_run=dry_run,
                             actuate_gripper=not args.no_gripper_action,
                             step_pause=args.step_pause,
                             home_between=home_between,
                             home_between_speed=args.home_between_speed)
        else:
            replay_dual_arm_motion_pkl(
                lft, rgt, args.pkl, speed=args.speed, dry_run=dry_run,
                actuate_gripper=not args.no_gripper_action,
                payload=payload,
                home_between=home_between,
                home_between_speed=args.home_between_speed,
                step_pause_s=args.step_pause,
            )
        if args.home_end and not args.home_only:
            if dry_run:
                print(f"[replay] (dry-run) 全部完成后将 go_home both arms @ speed={home_speed}")
            else:
                print(f"[replay] 全部零件完成, 回到 home (end) @ speed={home_speed}")
                lft.go_home(speed=home_speed)
                rgt.go_home(speed=home_speed)
        lft.close_connection(joint_release=args.end_release)
        rgt.close_connection(joint_release=args.end_release)
    else:
        arm = _build_arm(args.cfg, args.port, gripper_id)
        if args.home or args.home_only:
            if dry_run:
                print(f"[replay] (dry-run) would go_home @ speed={home_speed}")
            else:
                print(f"[replay] go_home @ speed={home_speed}")
                arm.go_home(speed=home_speed)
        if args.home_only:
            print("[replay] --home-only: skipping trajectory replay.")
        elif args.key_frames:
            _key_frames_single(arm, payload, args.arm, speed=args.speed, dry_run=dry_run,
                               actuate_gripper=not args.no_gripper_action,
                               pose_pause=args.step_pause,
                               home_between=home_between,
                               home_between_speed=args.home_between_speed)
        else:
            arm.run_motion_pkl(
                args.pkl, args.arm, speed=args.speed, dry_run=dry_run,
                actuate_gripper=not args.no_gripper_action,
                step_pause_s=args.step_pause,
                key_pause_s=args.key_pause,
                follow_path_blocking=args.follow_path,
                waypoint_stride=args.waypoint_stride,
                control_frequency=args.path_dt,
                payload=payload,
                home_between=home_between,
                home_between_speed=args.home_between_speed,
            )
        if args.home_end and not args.home_only:
            if dry_run:
                print(f"[replay] (dry-run) 全部完成后将 go_home @ speed={home_speed}")
            else:
                print(f"[replay] 全部零件完成, 回到 home (end) @ speed={home_speed}")
                arm.go_home(speed=home_speed)
        arm.close_connection(joint_release=args.end_release)

    print("[replay] done.")


if __name__ == "__main__":
    main()
