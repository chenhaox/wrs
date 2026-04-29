"""
Dual-Arm Sequence Execution Demo — YuanChair Assembly
======================================================

1. 初始 staging **不定死**：候选网格联合搜索；5 件 staging 两两不相交；桌面初始位不得
   侵入 **直接装在 fixture 上的零件**（椅面）的装配 goal。腿的 goal 相对椅面定义，若
   与桌面 staging 做 is_mcdwith 会大量误杀，故不参与此项检测。每只零件在 staging→goal
   上存在「左臂优先、右臂兜底」的 ``reason_common_gids`` 可行抓取。预搜索失败时回退为
   种子位姿并仍打开窗口、尝试执行与动画。
2. 执行阶段：RRT/PickPlace 的 ``obstacle_list`` 为 **地面 + 其余未抓件 staging +
   已装配件（goal）**；当前步规划前从列表中移除本件 staging。
3. fixture_pos 决定最终装配位；彩色显示选中的初始摆放，半透明显示目标 ghost。

Usage::

    python -m sealp.examples.motion.dual_sequence_execution
"""

import os
import pickle
import sys
import numpy as np
import wrs.basis.robot_math as rm
from wrs import wd, mgm, mcm
from direct.task.TaskManagerGlobal import taskMgr


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef, TaskPlan, StepParams
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.executor import SequenceExecutor
from sealp.primitives.transport import TransportPrimitive
from sealp.primitives.base import PrimitiveResult
from wrs.manipulation.pick_place import PickPlacePlanner

import wrs.robot_sim.robots.piper.piper_dual_arm as pda

# PickPlace：直线接近抓取点 / 装配点的方向（世界系 −X，与 approach_distance 配合）
_APPROACH_LINEAR_DIR = -rm.const.x_ax
# 桌腿装插到位后的撤离：世界 −X，15 cm（gen_pick_and_place 的 place_depart_*）
_LEG_PLACE_DEPART_DIR = -rm.const.x_ax
_LEG_PLACE_DEPART_DIST_M = 0.04

# ── 候选搜索：仅作「种子」，最终位姿由预搜索给出 ─────────────────
STAGING_SEEDS = {
    "seat": np.array([0.30, -0.10, 0.00]),
    "leg_fl": np.array([0.25, 0.20, 0.00]),
    "leg_bl": np.array([0.40, 0.15, 0.00]),
    "leg_fr": np.array([0.25, -0.60, 0.00]),
    "leg_br": np.array([0.40, -0.60, 0.00]),
}

STAGING_ZONES = {
    "seat": {"x": [0.22, 0.27, 0.32, 0.37], "y": [-0.22, -0.15, -0.10, -0.05]},
    "leg_fl": {"x": [0.18, 0.23, 0.28, 0.33, 0.38, 0.43], "y": [0.08, 0.13, 0.18, 0.23, 0.28]},
    "leg_bl": {"x": [0.28, 0.33, 0.38, 0.43], "y": [0.08, 0.13, 0.18, 0.23]},
    "leg_fr": {"x": [0.18, 0.23, 0.28, 0.33, 0.38], "y": [-0.72, -0.67, -0.62, -0.57, -0.52]},
    "leg_br": {"x": [0.33, 0.38, 0.43], "y": [-0.72, -0.67, -0.62, -0.57]},
}

INITIAL_STAGING_PART_IDS = ("seat", "leg_fl", "leg_bl", "leg_fr", "leg_br")

def load_obstacles_from_config(config_path, base):
    """从 sample_config 读取 environment.obstacles 并在场景中显示。"""
    if not os.path.isfile(config_path):
        print(f"[WARN] 未找到配置文件: {config_path}，将不加载静态环境障碍物。")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs,
        base_dir=cfg.config_dir,
    )
    loaded_obstacles = list(env.obstacle_list)
    for obs in loaded_obstacles:
        obs.attach_to(base)
        obs._sealp_role = "environment_obstacle"
    print(f"[环境] 已从配置加载 {len(loaded_obstacles)} 个静态障碍物。")
    return loaded_obstacles

def generate_staging_candidates(part_id, seed_pos, include_cross_arm=True):
    """候选列表：种子永远第一个，再补 STAGING_ZONES 网格，按距种子平面距离排序。"""
    candidates = [np.asarray(seed_pos, dtype=float).copy()]
    zone = STAGING_ZONES.get(part_id)
    if zone is None:
        return candidates

    grid = []
    for x in zone["x"]:
        for y in zone["y"]:
            grid.append(np.array([float(x), float(y), 0.0]))

    if include_cross_arm:
        cross = (
            {"x": [0.28, 0.33], "y": [-0.30, -0.25]}
            if part_id in ("leg_fr", "leg_br")
            else {"x": [0.28, 0.33], "y": [0.12, 0.18]}
        )
        for x in cross["x"]:
            for y in cross["y"]:
                grid.append(np.array([float(x), float(y), 0.0]))

    seen = {tuple(np.round(candidates[0], 4))}
    extra = []
    for p in grid:
        t = tuple(np.round(p, 4))
        if t in seen:
            continue
        seen.add(t)
        extra.append(p)
    extra.sort(key=lambda q: float(np.linalg.norm(q[:2] - seed_pos[:2])))
    candidates.extend(extra)
    return candidates


def is_pose_collision_free_with_selected(part_id, pos, rotmat, staging_obstacles, selected_part_ids):
    probe = staging_obstacles.get(part_id)
    if probe is None:
        return True
    old_pos, old_rot = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid in selected_part_ids:
            other = staging_obstacles.get(oid)
            if other is None:
                continue
            if probe.is_mcdwith(other):
                return False
        return True
    finally:
        probe.pos = old_pos
        probe.rotmat = old_rot


def build_goal_obstacle_models(asm, world_poses):
    """各零件在装配 goal 处的碰撞体（仅用于预搜索：初始位不得侵入他人装配域）。"""
    out = {}
    for pid, (gp, gr) in world_poses.items():
        # world_poses 含虚拟根 "fixture"，无对应 PartDef / 模型
        if pid not in asm.part_ids:
            continue
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue
        cm = mcm.CollisionModel(initor=mp)
        cm.pos = gp
        cm.rotmat = gr
        cm._sealp_role = "goal_placeholder"
        cm._sealp_part_id = pid
        out[pid] = cm
    return out


def _step_parent_id(asm, part_id):
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def is_staging_clear_of_foreign_goals(part_id, pos, rotmat, staging_obstacles, goal_models, asm):
    """桌面 staging 不得侵入「直接装在 fixture 上」的零件的 goal 体积（如椅面）。

    腿的装配位相对椅面定义，其 goal 与桌面初始位在几何上易误判相交，故跳过 parent!=fixture
    的 goal；执行阶段仍以 staging + 已装件为 RRT 障碍。
    """
    probe = staging_obstacles.get(part_id)
    if probe is None or not goal_models:
        return True
    old_pos, old_rot = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid, gcm in goal_models.items():
            if oid == part_id or gcm is None:
                continue
            if _step_parent_id(asm, oid) != "fixture":
                continue
            if probe.is_mcdwith(gcm):
                return False
        return True
    finally:
        probe.pos = old_pos
        probe.rotmat = old_rot


def apply_seed_staging(plan, staging_obstacles, seeds):
    """预搜索失败时回退：用种子位姿写回 plan 与 staging 碰撞体。"""
    for pid in INITIAL_STAGING_PART_IDS:
        st = plan.get_staging(pid)
        if st is None:
            continue
        p = seeds[pid].copy() if pid in seeds else st.pos.copy()
        r = np.eye(3)
        plan.set_staging(pid, pos=p, rotmat=r)
        o = staging_obstacles.get(pid)
        if o is not None:
            o.pos, o.rotmat = p, r


def model_alias_for_part(part_id):
    return "seat_model" if part_id == "seat" else "leg_model"


def load_grasp_cache(grasp_paths):
    out = {}
    for alias, path in grasp_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            out[alias] = pickle.load(f)
        print(f"[预搜索] 已加载 {alias!r}: {len(out[alias])} 个抓取")
    return out


def pick_place_reason_common_ok(arm, grasp_collection, sp, sr, gp, gr, obstacle_list):
    if grasp_collection is None or len(grasp_collection) == 0:
        return False, 0
    planner = PickPlacePlanner(robot=arm)
    gids = planner.reason_common_gids(
        grasp_collection=grasp_collection,
        goal_pose_list=[(sp, sr), (gp, gr)],
        obstacle_list=obstacle_list,
    )
    return len(gids) > 0, len(gids)


def _sync_staging_obstacles(search_part_ids, part_id, cand_pos, cand_rot, chosen_layout,
                            default_poses, staging_obstacles):
    for pid in search_part_ids:
        if pid == part_id:
            staging_obstacles[pid].pos = cand_pos
            staging_obstacles[pid].rotmat = cand_rot
        elif pid in chosen_layout:
            p, r = chosen_layout[pid]
            staging_obstacles[pid].pos = p
            staging_obstacles[pid].rotmat = r
        else:
            p, r = default_poses[pid]
            staging_obstacles[pid].pos = p
            staging_obstacles[pid].rotmat = r


def _obstacle_list_for_reason(search_part_ids, part_id, staging_obstacles, env_obstacles):
    # 将单一 ground 替换为环境障碍物列表
    obs = list(env_obstacles)
    for pid in search_part_ids:
        if pid != part_id:
            obs.append(staging_obstacles[pid])
    return obs


def prevalidate_initial_staging_layout(plan, robot, staging_obstacles, grasp_cache,
                                     world_poses, env_obstacles, seeds, assembly_def):
    goal_models = build_goal_obstacle_models(assembly_def, world_poses)
    search_part_ids = [
        p for p in INITIAL_STAGING_PART_IDS
        if plan.get_staging(p) is not None and p in staging_obstacles
    ]
    default_poses = {}
    candidate_map = {}
    for pid in search_part_ids:
        st = plan.get_staging(pid)
        rot = st.rotmat.copy() if st.rotmat is not None else np.eye(3)
        default_poses[pid] = (st.pos.copy(), rot)
        seed = seeds.get(pid, st.pos.copy())
        candidate_map[pid] = generate_staging_candidates(pid, seed)

    chosen_layout = {}
    selected = []

    def restore(pid):
        p, r = default_poses[pid]
        plan.set_staging(pid, pos=p, rotmat=r)
        o = staging_obstacles.get(pid)
        if o is not None:
            o.pos, o.rotmat = p, r

    def dfs(depth):
        if depth >= len(search_part_ids):
            return True
        pid = search_part_ids[depth]
        _, base_rot = default_poses[pid]
        cands = candidate_map[pid]
        print(f"\n  [预搜索] {pid}: {len(cands)} 个候选")

        for idx, cand_pos in enumerate(cands):
            staging_obstacles[pid].pos = cand_pos
            staging_obstacles[pid].rotmat = base_rot

            if not is_pose_collision_free_with_selected(
                    pid, cand_pos, base_rot, staging_obstacles, selected):
                print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} 与已选件碰撞")
                continue

            if not is_staging_clear_of_foreign_goals(
                    pid, cand_pos, base_rot, staging_obstacles, goal_models, assembly_def):
                print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} 侵入 fixture 上零件装配域")
                continue

            if pid not in world_poses:
                continue
            gp, gr = world_poses[pid]
            gc = grasp_cache[model_alias_for_part(pid)]
            _sync_staging_obstacles(
                search_part_ids, pid, cand_pos, base_rot, chosen_layout,
                default_poses, staging_obstacles)
            obs = _obstacle_list_for_reason(
                search_part_ids, pid, staging_obstacles, env_obstacles)

            ok_l, nl = pick_place_reason_common_ok(
                robot.lft_arm, gc, cand_pos, base_rot, gp, gr, obs)
            if ok_l:
                tag = "左臂"
                n = nl
            else:
                ok_r, nr = pick_place_reason_common_ok(
                    robot.rgt_arm, gc, cand_pos, base_rot, gp, gr, obs)
                if not ok_r:
                    print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} 双臂均无共同可行抓取")
                    continue
                tag = "右臂"
                n = nr

            plan.set_staging(pid, pos=cand_pos, rotmat=base_rot)
            chosen_layout[pid] = (cand_pos.copy(), base_rot.copy())
            selected.append(pid)
            print(f"    + #{idx + 1} {np.round(cand_pos, 3).tolist()} ({tag}, 共同抓取数={n})")

            if dfs(depth + 1):
                return True

            selected.pop()
            chosen_layout.pop(pid, None)

        restore(pid)
        return False

    print("\n[预搜索] 联合搜索初始 staging（件间无碰撞 + 不侵入他人 goal + staging→goal 共同抓取）…")
    if not dfs(0):
        for pid in search_part_ids:
            restore(pid)
        raise RuntimeError(
            "未找到满足「件间无碰撞 + 不侵入 fixture 上椅面装配域 + 双臂之一可 reason_common」的 "
            "5 件初始布局。")

    print("\n[预搜索] 成功：")
    for pid in search_part_ids:
        print(f"  {pid}: {np.round(chosen_layout[pid][0], 3).tolist()}")
    return chosen_layout


def _motion_replay_collision_free(robot_arm, mot_data, obstacle_list, granularity=0.03):
    """
    严格的轨迹复验：引入关节空间密集插值，彻底防止细长障碍物的「穿模」漏检。
    granularity=0.03 (约1.7度)，保证末端执行器的单步位移极小，无法跳过椅腿。
    """
    if mot_data is None or len(mot_data.jv_list) == 0:
        return True, None

    obs = list(obstacle_list) if obstacle_list else []
    if not obs:
        return True, None
    robot_arm.backup_state()
    try:
        jv_list = mot_data.jv_list
        for i in range(len(jv_list) - 1):
            jv0 = np.array(jv_list[i])
            jv1 = np.array(jv_list[i + 1])
            dist = np.linalg.norm(jv1 - jv0)
            steps = max(int(dist / granularity), 1)
            for step in range(steps):
                alpha = step / float(steps)
                interp_jv = jv0 * (1.0 - alpha) + jv1 * alpha

                robot_arm.goto_given_conf(interp_jv)
                hit = robot_arm.is_collided(obstacle_list=obs)
                collided = hit[0] if isinstance(hit, tuple) else hit
                if collided:
                    return False, interp_jv

        # 必须检查最后一个关键帧
        robot_arm.goto_given_conf(jv_list[-1])
        hit = robot_arm.is_collided(obstacle_list=obs)
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided:
            return False, jv_list[-1]

        return True, None
    finally:
        robot_arm.restore_state()


def animate_sequence(base, execution_result, interval=0.01):
    all_motions = []
    for sr in execution_result.steps:
        if not sr.success:
            continue
        step_motions = {"step_id": sr.step_id, "part_id": sr.part_id, "meshes_to_show": []}
        if hasattr(sr, "mot_data") and sr.mot_data is not None:
            step_motions["meshes_to_show"].append(sr.mot_data.mesh_list)
        else:
            if hasattr(sr, "mot_data_lft") and sr.mot_data_lft is not None:
                step_motions["meshes_to_show"].append(sr.mot_data_lft.mesh_list)
            if hasattr(sr, "mot_data_rgt") and sr.mot_data_rgt is not None:
                step_motions["meshes_to_show"].append(sr.mot_data_rgt.mesh_list)
        if step_motions["meshes_to_show"]:
            all_motions.append(step_motions)

    if not all_motions:
        print("No motions to animate.")
        return

    class _State:
        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.total_steps = len(all_motions)

    state = _State()

    def _update(st, task):
        if st.step_idx >= st.total_steps:
            st.step_idx = 0
            st.frame_idx = 0
        current_step = all_motions[st.step_idx]
        mesh_lists = current_step["meshes_to_show"]
        max_frames = max(len(ml) for ml in mesh_lists)
        if st.frame_idx > 0:
            for ml in mesh_lists:
                if st.frame_idx - 1 < len(ml):
                    ml[st.frame_idx - 1].detach()
        elif st.step_idx > 0:
            prev_step = all_motions[st.step_idx - 1]
            for ml in prev_step["meshes_to_show"]:
                if len(ml) > 0:
                    ml[-1].detach()
        if st.frame_idx >= max_frames:
            for ml in mesh_lists:
                for m in ml:
                    m.detach()
            st.step_idx += 1
            st.frame_idx = 0
            return task.again
        for ml in mesh_lists:
            frame_to_show = min(st.frame_idx, len(ml) - 1)
            ml[frame_to_show].attach_to(base)
        if base.inputmgr.keymap["space"]:
            st.frame_idx += 1
        return task.again

    taskMgr.doMethodLater(interval, _update, "sequence_animate", extraArgs=[state], appendTask=True)


def main():
    base = wd.World(cam_pos=[1.5, -0.3, 1.2], lookat_pos=[0.3, -0.3, 0.1])
    mgm.gen_frame().attach_to(base)

    # --- 动态加载 config 中的障碍物 ---
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "sample_config.yaml")
    )
    env_obstacles = load_obstacles_from_config(config_path, base)

    asmdef_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(os.path.join(asmdef_dir, "yuanchair.asmdef"))
    if not os.path.isfile(asmdef_path):
        print(f"Assembly definition not found at: {asmdef_path}, generating…")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()
    if not os.path.isfile(asmdef_path):
        print(f"ERROR: {asmdef_path}")
        return

    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    plan = TaskPlan(
        assembly_file=asmdef_path,
        name="YuanChair Dual-Arm Execution",
        description="Dual-arm sequential assembly of the YuanChair.",
    )
    plan.set_assembly(asm)

    center_y_offset = -0.30
    plan.fixture_pos = np.array([0.0, center_y_offset, 0.0])
    plan.fixture_rotmat = np.eye(3)

    # 仅种子：预搜索成功后会覆盖为最终 staging
    for pid in INITIAL_STAGING_PART_IDS:
        plan.set_staging(pid, pos=STAGING_SEEDS[pid].copy(), rotmat=np.eye(3))

    for i in range(asm.n_steps):
        plan.set_step_params(StepParams(
            step_id=i,
            primitive="single_arm_transport",
            approach_distance=0.06,
            depart_distance=0.05,
        ))

    world_poses = asm.compute_world_poses(
        fixture_pos=plan.fixture_pos,
        fixture_rotmat=plan.fixture_rotmat,
    )

    for pid in asm.part_ids:
        if pid not in world_poses:
            continue
        gp, gr = world_poses[pid]
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            ghost = mcm.CollisionModel(initor=mp)
            ghost.pos, ghost.rotmat = gp, gr
            ghost.alpha = 0.15
            ghost.attach_to(base)

    robot = pda.DualPiperNoBody(enable_cc=True)
    home = np.zeros(6)
    robot.lft_arm.goto_given_conf(home)
    robot.rgt_arm.goto_given_conf(home)
    robot.use_lft()

    grasp_paths = {
        "leg_model": os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output", "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output", "demo_yuanchair-part1_grasps.pickle"),
    }

    staging_obstacles = {}
    initial_obstacles = list(env_obstacles)
    for pid in asm.part_ids:
        st = plan.get_staging(pid)
        if st is None:
            continue
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            obs = mcm.CollisionModel(initor=mp)
            obs.pos = st.pos
            obs.rotmat = st.rotmat
            obs._sealp_part_id = pid
            obs._sealp_role = "staging_on_table"
            initial_obstacles.append(obs)
            staging_obstacles[pid] = obs

    gc = load_grasp_cache(grasp_paths)
    try:
        prevalidate_initial_staging_layout(
            plan, robot, staging_obstacles, gc, world_poses, env_obstacles, STAGING_SEEDS, asm)
    except Exception as e:
        print(f"\n[WARN] 预搜索失败: {e}")
        print("  已回退为种子初始位；仍打开窗口并尝试执行规划，便于查看场景与成功步的轨迹动画。")
        apply_seed_staging(plan, staging_obstacles, STAGING_SEEDS)

    staging_colors = {
        "seat": np.array([0.9, 0.6, 0.3, 0.85]),
        "leg_fl": np.array([0.3, 0.7, 0.3, 0.85]),
        "leg_fr": np.array([0.3, 0.3, 0.8, 0.85]),
        "leg_bl": np.array([0.8, 0.3, 0.3, 0.85]),
        "leg_br": np.array([0.7, 0.3, 0.7, 0.85]),
    }
    print("\n最终初始摆放（彩色）：")
    for pid in asm.part_ids:
        st = plan.get_staging(pid)
        if st is None:
            continue
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            vis = mcm.CollisionModel(initor=mp)
            vis.pos, vis.rotmat = st.pos, st.rotmat
            vis.rgba = staging_colors.get(pid, np.array([0.5, 0.5, 0.5, 0.85]))
            vis.attach_to(base)
            mgm.gen_frame(pos=st.pos, ax_length=0.03).attach_to(base)
            print(f"  {pid}: {np.round(st.pos, 4).tolist()}")

    robot.gen_meshmodel(alpha=0.2).attach_to(base)

    executor = SequenceExecutor(
        robot=robot,
        assembly_def=asm,
        task_plan=plan,
        obstacle_list=initial_obstacles,
        grasp_paths=grasp_paths,
        # 略增大已装配件的 CD 包络，避免后续装配/撤离轨迹贴得过紧碰到已装好的腿
        assembled_cd_ex_radius=0.05,
    )

    lft_t = TransportPrimitive(robot.lft_arm)
    rgt_t = TransportPrimitive(robot.rgt_arm)

    class DualArmFallbackTransport:
        def __init__(self, lft, rgt):
            self.lft, self.rgt = lft, rgt
            self.planner = lft.planner

        def plan(self, **kwargs):
            kw_l = dict(kwargs)
            if kw_l.get("obstacle_list") is not None:
                kw_l["obstacle_list"] = list(kw_l["obstacle_list"])
            kw_l["end_jnt_values"] = self.lft.robot.get_jnt_values()
            kw_l.setdefault("pick_approach_direction", _APPROACH_LINEAR_DIR)
            kw_l.setdefault("place_approach_direction_list", [_APPROACH_LINEAR_DIR])
            # 抓取后沿世界 +Z 抬起 20 cm（PickPlacePlanner.gen_pick_and_place 的 pick_depart_distance）
            kw_l.setdefault("pick_depart_distance", 0.20)
            kw_r = dict(kwargs)
            if kw_r.get("obstacle_list") is not None:
                kw_r["obstacle_list"] = list(kw_r["obstacle_list"])
            kw_r["end_jnt_values"] = self.rgt.robot.get_jnt_values()
            kw_r.setdefault("pick_approach_direction", _APPROACH_LINEAR_DIR)
            kw_r.setdefault("place_approach_direction_list", [_APPROACH_LINEAR_DIR])
            kw_r.setdefault("pick_depart_distance", 0.20)
            pid = kw_l.get("part_id")
            if pid is not None and str(pid).startswith("leg_"):
                kw_l["place_depart_direction_list"] = [_LEG_PLACE_DEPART_DIR]
                kw_l["place_depart_distance_list"] = [_LEG_PLACE_DEPART_DIST_M]
                kw_r["place_depart_direction_list"] = [_LEG_PLACE_DEPART_DIR]
                kw_r["place_depart_distance_list"] = [_LEG_PLACE_DEPART_DIST_M]
            kw_l.pop("part_id", None)
            kw_r.pop("part_id", None)
            res_l = self.lft.plan(**kw_l)
            if res_l.success and res_l.mot_data is not None:
                ok, _ = _motion_replay_collision_free(
                    self.lft.robot, res_l.mot_data, kw_l.get("obstacle_list"))
                if not ok:
                    res_l = PrimitiveResult(
                        success=False,
                        error_msg="左臂轨迹复验与障碍碰撞",
                    )
            if res_l.success:
                self.lft.robot.goto_given_conf(res_l.end_jnt_values)
                return res_l
            res_r = self.rgt.plan(**kw_r)
            if res_r.success and res_r.mot_data is not None:
                ok, _ = _motion_replay_collision_free(
                    self.rgt.robot, res_r.mot_data, kw_r.get("obstacle_list"))
                if not ok:
                    res_r = PrimitiveResult(
                        success=False,
                        error_msg="右臂轨迹复验与障碍碰撞",
                    )
            if res_r.success:
                self.rgt.robot.goto_given_conf(res_r.end_jnt_values)
                return res_r
            return PrimitiveResult(
                success=False,
                error_msg=f"左: {res_l.error_msg} | 右: {res_r.error_msg}",
            )

    executor._selector._transport = DualArmFallbackTransport(lft_t, rgt_t)

    _orig = executor._execute_step

    def _find_pop(obs_list, part_id):
        for i, o in enumerate(obs_list):
            if getattr(o, "_sealp_role", None) == "staging_on_table" and \
                    getattr(o, "_sealp_part_id", None) == part_id:
                return obs_list.pop(i)
        return None

    def _patched(*, step, world_poses, obs_list, current_conf_rgt, current_conf_lft):
        removed = _find_pop(obs_list, step.part_id)
        print(f"  [障碍] step {step.step_id} {step.part_id!r}: n_obs={len(obs_list)} "
              f"(已去掉本件 staging)")
        step_result = None
        try:
            step_result = _orig(
                step=step, world_poses=world_poses, obs_list=obs_list,
                current_conf_rgt=current_conf_rgt, current_conf_lft=current_conf_lft)
        finally:
            if removed is not None and (step_result is None or not step_result.success):
                obs_list.append(removed)
        return step_result

    executor._execute_step = _patched

    print("\nExecuting…")
    result = executor.execute_all(stop_on_failure=False)
    if result.total_frames > 0:
        print(f"\n按 SPACE 逐帧播放轨迹（共 {result.total_frames} 帧）。")
        animate_sequence(base, result)
    else:
        print("\n无成功轨迹可播放：仍显示初始摆放、目标 ghost 与机械臂，可旋转视角查看。")
    base.run()


if __name__ == "__main__":
    main()
