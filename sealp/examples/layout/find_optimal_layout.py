"""
Fast Optimal Dual-Arm Layout Search (CEM + 综合可执行性评分)
=====================================================
结合了交叉熵方法 (CEM) 的高效率与多维度的综合评分：
评分标准 (Layout Score) =
    w1 * 抓取冗余度 (Grasp Count) +
    w2 * 灵巧度 (Manipulability) +
    w3 * 移动代价 (Distance Cost, 负权重)

机器人：双臂 Panthera-HT（``DualPantheraHTNoBody``，右臂 base y=-0.62）。
与下游 ``dual_sequence_execution.py`` / ``eval_dual_layout.py`` / ``search_dual_layout.py``
完全对齐，写出的 ``dual_yuanchair_optimal_searched.layout`` 会被
``dual_sequence_execution.py`` 自动加载并跑出动画。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner
import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import WorkspaceLayout
from sealp.layout.dual_staging_search import find_obstacle_def
from sealp.examples.layout.eval_dual_layout import (
    STAGING_SEEDS, FIXTURE_POS, FIXTURE_ROTMAT,
    ROBOT_BASE_POS, ROBOT_BASE_ROTMAT,
    load_grasp_cache, model_alias_for_part,
)
from sealp.layout.reachability import check_pose_reachability

# Panthera-HT 双臂右臂 base 相对左臂的 y 偏移（见
# ``DualPantheraHTNoBody.__init__`` 的 ``arm_y_offset`` 默认值）。Piper 时代
# 这里是 -0.597；切换到 Panthera-HT 后必须同步，否则 staging 搜索的右半
# 区采样窗口会偏。
_DUAL_ARM_Y_OFFSET = 0.62

# ══════════════════════════════════════════════════════════════
#  评分权重配置 (可根据实际需求微调)
# ══════════════════════════════════════════════════════════════
#  Layout Score 由三项加权和构成（在 ``l2`` 里组装；越大越好）：
#
#  Layout Score
#    = WEIGHT_GRASP_COUNT * (min(n)^GRASP_MIN_EXP + 0.1 * mean(n))
#      + WEIGHT_MANIPULABILITY * avg_manip                            (≥0)
#      + WEIGHT_DISTANCE * avg_dist                                   (≤0)
#
#  Tuning 提示
#  -----------
#  * **WEIGHT_GRASP_COUNT** 控制"瓶颈件"压力。``GRASP_MIN_EXP`` 取 1.5
#    后，把 min(n) 升级为凸函数：n=7 → 18.5、n=10 → 31.6、n=15 → 58
#    —— 让 optimizer 显著偏向"抬高短板"而非"只刷 mean"。已观察到下
#    游 ``dual_sequence_execution`` 在 n≤7 的瓶颈件上必败（RRT 复验把
#    cdprim 通过的擦边轨迹也过滤掉），实测把瓶颈推到 n≥10 才稳定。
#  * **WEIGHT_MANIPULABILITY** 旧值 5.0 几乎不起作用：manip 典型量级
#    O(1e-3)，5×0.005=0.025，与 grasp_score O(10) 相差 400×。这里
#    放大到 80，把灵巧度贡献抬到 0.2~0.6 区间，远离机械臂奇异点的
#    布局在评分上能压过一点点更短的距离。
#  * **WEIGHT_DISTANCE** 取负权重；旧值 -0.3 太轻。Panthera-HT 右臂
#    base 在 y=-0.62，右半区件 staging 离 assembly 区 (y≈-0.30) 经常
#    有 0.5m+ 跨距，RRT 轨迹易扫穿已装件。-0.8 加重距离惩罚后更倾
#    向于让右臂件 staging 与 goal 同区。
WEIGHT_GRASP_COUNT = 2.5     # 抓取冗余度权重 (放大瓶颈件的影响)
WEIGHT_MANIPULABILITY = 80.0 # 灵巧度权重 (manip 量级 O(1e-3)，需大幅放大)
WEIGHT_DISTANCE = -0.8       # 距离惩罚权重 (越远扣分越多，引导短运输)
GRASP_MIN_EXP = 1.5          # min(n) 的指数，>1 时强烈惩罚"低于其它件"

# ══════════════════════════════════════════════════════════════
#  几何流常量
# ══════════════════════════════════════════════════════════════
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0])
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0])
PLACE_APPROACH_DIST = 0.05
LEG_PLACE_DEPART_DIR = np.array([-1.0, 0.0, 0.0])
SEAT_PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0])
LEG_PLACE_DEPART_DIST = 0.05
SEAT_PLACE_DEPART_DIST = 0.05
APPROACH_DIST = 0.0
LINEAR_GRANULARITY = 0.04
HOME_JV = np.zeros(6)


# ══════════════════════════════════════════════════════════════
#  任务规范
# ══════════════════════════════════════════════════════════════
def _here(*parts) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


@dataclass
class FastSearchTask:
    name: str
    asmdef_path: str
    grasp_pickles: Dict[str, str]
    part_ids: Tuple[str, ...]
    staging_seeds: Dict[str, np.ndarray]
    fixture_pos: np.ndarray
    fixture_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    robot_base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    robot_base_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    config_yaml_path: str = ""
    table_obstacle_name: str = "work_table"
    table_margin: float = 0.06
    part_xy_bounds: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    model_alias_fn: Optional[Callable[[str], str]] = None
    output_layout_name: str = "dual_optimal_searched"


YUANCHAIR_FAST_TASK = FastSearchTask(
    name="yuanchair",
    asmdef_path=_here("..", "..", "assembly_sequence", "_demo_output", "yuanchair.asmdef"),
    grasp_pickles={
        "leg_model": _here("..", "grasp", "_output", "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": _here("..", "grasp", "_output", "demo_yuanchair-part1_grasps.pickle"),
    },
    part_ids=("seat", "leg_bl", "leg_br", "leg_fl", "leg_fr"),
    staging_seeds=STAGING_SEEDS,
    fixture_pos=FIXTURE_POS,
    fixture_rotmat=FIXTURE_ROTMAT,
    robot_base_pos=ROBOT_BASE_POS,
    robot_base_rotmat=ROBOT_BASE_ROTMAT,
    config_yaml_path=_here("..", "..", "config", "sample_config.yaml"),
    model_alias_fn=model_alias_for_part,
    output_layout_name="dual_yuanchair_optimal_searched",
)


# ══════════════════════════════════════════════════════════════
#  几何 helper
# ══════════════════════════════════════════════════════════════
def _step_parent_id(asm, part_id):
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def _arms_collide_at_home(robot, obstacle_list) -> bool:
    if not obstacle_list: return False
    for arm in (robot.lft_arm, robot.rgt_arm):
        hit = arm.is_collided(obstacle_list=list(obstacle_list))
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided: return True
    return False


def _segment_reachable(arm, grasp, base_pos, base_rot, direction, distance, n_samples=4) -> bool:
    if distance is None or distance <= 1e-6 or n_samples < 1: return True
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1e-9: return True
    dir_unit = direction / n
    tcp_pos = base_rot.dot(grasp.ac_pos) + base_pos
    tcp_rot = base_rot.dot(grasp.ac_rotmat)
    seed = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
    if seed is None: return False
    for i in range(1, n_samples + 1):
        d = distance * i / n_samples
        end_pos = tcp_pos + dir_unit * d
        jv = arm.ik(tgt_pos=end_pos, tgt_rotmat=tcp_rot, seed_jnt_values=seed)
        if jv is None: return False
        seed = jv
    return True


def _reason_common_ok(arm, gc, sp, sr, gp, gr, obstacle_list) -> Tuple[bool, int]:
    if gc is None or len(gc) == 0: return False, 0
    planner = PickPlacePlanner(robot=arm)
    gids = planner.reason_common_gids(
        grasp_collection=gc, goal_pose_list=[(sp, sr), (gp, gr)], obstacle_list=obstacle_list)
    if not gids: return False, 0
    gids = [g for g in gids if _segment_reachable(arm, gc[g], sp, sr, PICK_DEPART_DIR, PICK_DEPART_DIST)]
    if not gids: return False, 0
    rev = -np.asarray(PLACE_APPROACH_DIR, dtype=float)
    gids = [g for g in gids if _segment_reachable(arm, gc[g], gp, gr, rev, PLACE_APPROACH_DIST)]
    if not gids: return False, 0
    return True, len(gids)


def _arm_priority_for_part(part_id: str) -> Tuple[str, str]:
    pid = str(part_id)
    return ("rgt", "lft") if pid and pid[-1] == 'r' else ("lft", "rgt")


# ══════════════════════════════════════════════════════════════
#  layout + 评分容器 (新增了灵巧度和距离指标)
# ══════════════════════════════════════════════════════════════
@dataclass
class _ScoredLayout:
    xy: Dict[str, np.ndarray]
    layout_score: float = -np.inf
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    arm_choice: Dict[str, str] = field(default_factory=dict)
    manipulability_avg: float = 0.0  # NEW
    distance_cost: float = 0.0  # NEW
    l1_pass: bool = False
    l2_pass: bool = False
    l3_pass: bool = False
    fail_reason: str = ""


# ══════════════════════════════════════════════════════════════
#  核心搜索器
# ══════════════════════════════════════════════════════════════
class FastLayoutSearcher:
    def __init__(self, task: FastSearchTask, *, enable_l3: bool = False, ik_retry_n: int = 0):
        self.task = task
        self.enable_l3 = enable_l3
        self.model_alias_fn = (task.model_alias_fn or (lambda pid: f"{pid}_model"))
        self._world = None

        self.asm = AssemblyDef.load(task.asmdef_path)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=task.fixture_pos, fixture_rotmat=task.fixture_rotmat)

        self.env_obs: List = []
        self.table_def: Optional[dict] = None
        if task.config_yaml_path and os.path.isfile(task.config_yaml_path):
            cfg = load_config(task.config_yaml_path)
            env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
            self.env_obs = list(env.obstacle_list)
            self.table_def = find_obstacle_def(cfg.obstacle_defs, task.table_obstacle_name)

        self.robot = pda.DualPantheraHTNoBody(
            pos=task.robot_base_pos, rotmat=task.robot_base_rotmat,
            arm_y_offset=_DUAL_ARM_Y_OFFSET, enable_cc=True)
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)

        if ik_retry_n > 0:
            for arm_robot in (self.robot.lft_arm.manipulator, self.robot.rgt_arm.manipulator):
                if hasattr(arm_robot, "_ik_retry_n"): arm_robot._ik_retry_n = ik_retry_n

        self.grasp_cache = load_grasp_cache(task.grasp_pickles)

        self.search_part_ids = [p for p in task.part_ids if p in task.staging_seeds]
        self.staging_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid in self.search_part_ids:
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp): continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = np.zeros(3);
            cm.rotmat = np.eye(3)
            cm._sealp_part_id = pid;
            cm._sealp_role = "fast_staging"
            self.staging_obs[pid] = cm

        self.goal_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid, (gp, gr) in self.world_poses.items():
            if pid not in self.asm.part_ids: continue
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp): continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = gp;
            cm.rotmat = gr
            cm._sealp_part_id = pid;
            cm._sealp_role = "fast_goal"
            self.goal_obs[pid] = cm

        self.bounds = self._resolve_xy_bounds()

        self.funnel = {"sampled": 0, "l1_pass": 0, "l2_pass": 0, "l3_attempts": 0, "l3_pass": 0}
        self.history: List[Tuple[int, float, float]] = []
        self.l3_marked_gens: List[Tuple[int, bool]] = []

    def _resolve_xy_bounds(self):
        if self.task.part_xy_bounds is not None: return dict(self.task.part_xy_bounds)
        if self.table_def is not None and self.table_def.get("type") == "box":
            tp, te = self.table_def["pos"], self.table_def["extent"]
            tx_lo = float(tp[0]) - float(te[0]) / 2 + self.task.table_margin
            tx_hi = float(tp[0]) + float(te[0]) / 2 - self.task.table_margin
            ty_lo = float(tp[1]) - float(te[1]) / 2 + self.task.table_margin
            ty_hi = float(tp[1]) + float(te[1]) / 2 - self.task.table_margin
        else:
            tx_lo, tx_hi, ty_lo, ty_hi = 0.0, 0.7, -1.0, 0.4
        # 与 ``DualPantheraHTNoBody.__init__`` 中 ``arm_y_offset=0.62`` 对齐。
        # 切换到 Panthera-HT 后：右臂 base 在 y=-0.62（Piper 时代是 -0.597），
        # 同时桌腿 cdprim 较粗，DFS 实测有效的 seed 右臂腿 y≈-0.85。这里
        # 把右臂区下扩到 rgt_y-0.42 / 上扩到 rgt_y+0.20 与 dual demo 对齐。
        lft_y, rgt_y = 0.0, -_DUAL_ARM_Y_OFFSET
        out = {}
        for pid in self.search_part_ids:
            x_lo, x_hi = max(tx_lo, 0.18), min(tx_hi, 0.55)
            if pid == "seat":
                out[pid] = ((max(x_lo, 0.22), min(x_hi, 0.40)), (-0.22, -0.02))
            elif pid.endswith("r"):
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, rgt_y - 0.42), min(ty_hi, rgt_y + 0.20)))
            else:
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, lft_y + 0.06), min(ty_hi, lft_y + 0.32)))
        return out

    def _apply_xy(self, xy: Dict[str, np.ndarray]):
        for pid in self.search_part_ids:
            cm = self.staging_obs.get(pid)
            if cm is None: continue
            p = xy.get(pid, self.task.staging_seeds[pid])
            cm.pos = np.array([float(p[0]), float(p[1]), 0.0])
            cm.rotmat = np.eye(3)

    def l1(self, sl: _ScoredLayout) -> bool:
        self._apply_xy(sl.xy)
        ids = list(self.staging_obs.keys())
        for i in range(len(ids)):
            cm_i = self.staging_obs[ids[i]]
            for j in range(i + 1, len(ids)):
                cm_j = self.staging_obs[ids[j]]
                if cm_i.is_mcdwith(cm_j):
                    sl.fail_reason = f"L1: {ids[i]} vs {ids[j]} overlap"
                    return False
        for pid in ids:
            cm = self.staging_obs[pid]
            for gid, gcm in self.goal_obs.items():
                if gid == pid: continue
                if _step_parent_id(self.asm, gid) != "fixture": continue
                if cm.is_mcdwith(gcm):
                    sl.fail_reason = f"L1: {pid} 侵入 fixture goal {gid}"
                    return False
        if _arms_collide_at_home(self.robot, [self.staging_obs[p] for p in ids]):
            sl.fail_reason = "L1: home 姿态下 staging 与机械臂穿模"
            return False
        sl.l1_pass = True
        return True

    def _step_aware_obs(self, current_pid: str, placed: set) -> List:
        obs = list(self.env_obs)
        for other in placed:
            if other in self.goal_obs: obs.append(self.goal_obs[other])
        for other in self.search_part_ids:
            if other == current_pid or other in placed: continue
            if other in self.staging_obs: obs.append(self.staging_obs[other])
        return obs

    # === 【核心修改：综合评分注入】 ===
    def l2(self, sl: _ScoredLayout, *, retry: int = 0) -> bool:
        self._apply_xy(sl.xy)
        placed: set = set()
        counts: Dict[str, int] = {}
        arms: Dict[str, str] = {}

        total_manip = 0.0
        total_dist = 0.0

        for s in self.asm.steps:
            pid = s.part_id
            if pid not in self.search_part_ids or pid not in self.world_poses: continue

            sp = self.staging_obs[pid].pos.copy()
            sr = self.staging_obs[pid].rotmat.copy()
            gp, gr = self.world_poses[pid]
            gc = self.grasp_cache.get(self.model_alias_fn(pid))

            if gc is None:
                sl.fail_reason = f"L2: 抓取库缺 {pid}"
                return False

            obs = self._step_aware_obs(pid, placed)
            pref = _arm_priority_for_part(pid)
            arm0 = (self.robot.lft_arm if pref[0] == "lft" else self.robot.rgt_arm)
            arm1 = (self.robot.lft_arm if pref[1] == "lft" else self.robot.rgt_arm)

            tag, n, best_arm = None, 0, None
            for attempt in range(retry + 1):
                ok0, n0 = _reason_common_ok(arm0, gc, sp, sr, gp, gr, obs)
                if ok0:
                    tag, n, best_arm = pref[0], n0, arm0
                    break
                ok1, n1 = _reason_common_ok(arm1, gc, sp, sr, gp, gr, obs)
                if ok1:
                    tag, n, best_arm = pref[1], n1, arm1
                    break

            if tag is None:
                sl.fail_reason = f"L2: step={s.step_id} {pid} 双臂均无可行 reason"
                return False

            # 计算灵巧度 (仅抽取 5 个有效抓取进行快速评估)
            pick_reach = check_pose_reachability(best_arm, sp, sr, gc, obs, max_grasps=5)
            place_reach = check_pose_reachability(best_arm, gp, gr, gc, obs, max_grasps=5)
            avg_manip = (pick_reach.best_manipulability + place_reach.best_manipulability) / 2.0

            # 计算距离代价
            dist = float(np.linalg.norm(np.array(sp) - np.array(gp)))

            counts[pid] = n
            arms[pid] = tag
            total_manip += avg_manip
            total_dist += dist
            placed.add(pid)

        sl.grasp_counts = counts
        sl.arm_choice = arms

        if counts:
            n_parts = len(counts)
            # 基础抓取分
            vs = list(counts.values())
            base_grasp_score = float(min(vs) + 0.1 * (sum(vs) / n_parts)) * WEIGHT_GRASP_COUNT

            # 灵巧度分
            manip_score = (total_manip / n_parts) * WEIGHT_MANIPULABILITY

            # 距离惩罚
            dist_penalty = (total_dist / n_parts) * WEIGHT_DISTANCE

            sl.manipulability_avg = total_manip / n_parts
            sl.distance_cost = total_dist / n_parts
            sl.layout_score = base_grasp_score + manip_score + dist_penalty

        sl.l2_pass = True
        return True

    def l3(self, sl: _ScoredLayout) -> bool:
        if not self.enable_l3: return False
        from sealp.primitives.transport import TransportPrimitive
        self._apply_xy(sl.xy)
        lft_t = TransportPrimitive(self.robot.lft_arm)
        rgt_t = TransportPrimitive(self.robot.rgt_arm)
        placed: set = set()
        for s in self.asm.steps:
            pid = s.part_id
            if pid not in self.search_part_ids or pid not in self.world_poses: continue
            tag = sl.arm_choice.get(pid, "lft")
            transport = lft_t if tag == "lft" else rgt_t
            sp, sr = self.staging_obs[pid].pos.copy(), self.staging_obs[pid].rotmat.copy()
            gp, gr = self.world_poses[pid]
            mp = self.asm.model_path(pid)
            obj_cm = mcm.CollisionModel(initor=mp);
            obj_cm.pos = sp;
            obj_cm.rotmat = sr
            gc = self.grasp_cache.get(self.model_alias_fn(pid))
            obs = self._step_aware_obs(pid, placed)
            place_dep_dir = LEG_PLACE_DEPART_DIR if pid.startswith("leg_") else SEAT_PLACE_DEPART_DIR
            place_dep_dist = LEG_PLACE_DEPART_DIST if pid.startswith("leg_") else SEAT_PLACE_DEPART_DIST
            try:
                res = transport.plan(
                    obj_cmodel=obj_cm, grasp_collection=gc, goal_pose_list=[(gp, gr)], obstacle_list=obs,
                    approach_distance=APPROACH_DIST, depart_distance=PICK_DEPART_DIST,
                    pick_depart_direction=PICK_DEPART_DIR, pick_depart_distance=PICK_DEPART_DIST,
                    place_approach_direction_list=[PLACE_APPROACH_DIR],
                    place_approach_distance_list=[PLACE_APPROACH_DIST],
                    place_depart_direction_list=[place_dep_dir], place_depart_distance_list=[place_dep_dist],
                    linear_granularity=LINEAR_GRANULARITY,
                )
            except Exception as e:
                sl.fail_reason = f"L3: step {s.step_id} {pid} {tag} 抛错 {type(e).__name__}: {e!r}"
                return False
            if not res.success:
                sl.fail_reason = f"L3: step {s.step_id} {pid} {tag} 失败: {res.error_msg or 'no plan'}"
                return False
            placed.add(pid)
        sl.l3_pass = True
        return True


# ══════════════════════════════════════════════════════════════
#  CEM 优化器
# ══════════════════════════════════════════════════════════════
def _clip_xy(p, bounds):
    (xlo, xhi), (ylo, yhi) = bounds
    return np.array([float(np.clip(p[0], xlo, xhi)), float(np.clip(p[1], ylo, yhi))])


def _seed_with_noise(seeds, pids, bounds, rng, scale):
    return {pid: _clip_xy(np.array([seeds[pid][0], seeds[pid][1]]) + rng.normal(0, scale, 2), bounds[pid]) for pid in
            pids}


def cem_search(searcher: FastLayoutSearcher, *, pop_size: int = 24, elite_n: int = 6, generations: int = 8,
               sigma_init: float = 0.08, sigma_decay: float = 0.7, sigma_min: float = 0.015,
               try_l3_top_k: int = 0, rng_seed: int = 0, l2_retry: int = 0, verbose: bool = True) -> Optional[
    _ScoredLayout]:
    rng = np.random.default_rng(rng_seed)
    pids = searcher.search_part_ids
    bounds = searcher.bounds
    seeds = searcher.task.staging_seeds

    mu = {pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids}
    sigma = {pid: np.array([sigma_init, sigma_init]) for pid in pids}

    candidates: List[_ScoredLayout] = []
    candidates.append(_ScoredLayout(xy={pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids}))
    n_anchor = max(2, pop_size // 2)
    n_small_noise = (n_anchor - 1) // 2
    for _ in range(n_small_noise): candidates.append(
        _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init / 2)))
    for _ in range((n_anchor - 1) - n_small_noise): candidates.append(
        _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init)))
    while len(candidates) < pop_size:
        candidates.append(
            _ScoredLayout(xy={pid: _clip_xy(rng.normal(mu[pid], sigma_init * 1.4), bounds[pid]) for pid in pids}))

    best_overall: Optional[_ScoredLayout] = None

    for gen in range(generations):
        scored: List[_ScoredLayout] = []
        for sl in candidates:
            searcher.funnel["sampled"] += 1
            if not searcher.l1(sl): continue
            searcher.funnel["l1_pass"] += 1
            if searcher.l2(sl, retry=l2_retry):
                searcher.funnel["l2_pass"] += 1
                scored.append(sl)

        scored.sort(key=lambda s: s.layout_score, reverse=True)

        if scored:
            best = scored[0]
            elite = scored[:elite_n]
            elite_mean = float(np.mean([e.layout_score for e in elite]))
            if (best_overall is None or best.layout_score > best_overall.layout_score):
                best_overall = best
        else:
            best, elite, elite_mean = None, [], -np.inf

        searcher.history.append((gen, best.layout_score if best else -np.inf, elite_mean))

        if verbose:
            n_l1 = sum(1 for c in candidates if c.l1_pass)
            best_msg = (
                f"best_score={best.layout_score:.2f} (manip={best.manipulability_avg:.3f}, dist={best.distance_cost:.3f})" if best else "-")
            print(f"\n[CEM gen {gen}/{generations - 1}] sampled={pop_size} L1ok={n_l1} L2ok={len(scored)} "
                  f"elite_mean={elite_mean:.2f}  {best_msg}")

        if not scored:
            if verbose: print("  -> 本代无可行解，sigma 放大，半数样本全盒重采。")
            for pid in pids:
                sigma[pid] = np.minimum(sigma[pid] * 2.0, np.array([0.25, 0.25]))
                mu[pid] = np.array([float(seeds[pid][0]), float(seeds[pid][1])])
            candidates = [
                _ScoredLayout(xy={pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids})]
            n_noise = (pop_size - 1) // 2
            for _ in range(n_noise): candidates.append(
                _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init * 1.5)))
            for _ in range((pop_size - 1) - n_noise):
                xy = {pid: np.array([float(rng.uniform(bounds[pid][0][0], bounds[pid][0][1])),
                                     float(rng.uniform(bounds[pid][1][0], bounds[pid][1][1]))]) for pid in pids}
                candidates.append(_ScoredLayout(xy=xy))
            continue

        if try_l3_top_k > 0 and searcher.enable_l3:
            for sl in scored[:try_l3_top_k]:
                searcher.funnel["l3_attempts"] += 1
                t0 = time.time()
                ok = searcher.l3(sl)
                dt = time.time() - t0
                searcher.l3_marked_gens.append((gen, ok))
                tag = "[OK] L3 通过" if ok else "[NO] L3 失败"
                if verbose:
                    print(f"  | try_l3({sl.layout_score:.2f}) {tag}  ({dt:.1f}s)")
                    if not ok: print(f"    reason: {sl.fail_reason}")
                if ok:
                    searcher.funnel["l3_pass"] += 1
                    return sl

        for pid in pids:
            ps = np.array([e.xy[pid] for e in elite])
            mu[pid] = ps.mean(axis=0)
            sigma[pid] = np.maximum(ps.std(axis=0) * sigma_decay, sigma_min)

        candidates = [_ScoredLayout(xy=dict(best.xy))]
        for _ in range(pop_size - 1):
            xy = {pid: _clip_xy(rng.normal(mu[pid], sigma[pid]), bounds[pid]) for pid in pids}
            candidates.append(_ScoredLayout(xy=xy))

    return best_overall


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
def run_fast_search(task: FastSearchTask, *, pop_size: int = 24, elite_n: int = 6, generations: int = 8,
                    try_l3_top_k: int = 0, enable_l3: bool = False, sigma_init: float = 0.08,
                    sigma_decay: float = 0.7, rng_seed: int = 0, l2_retry: int = 0, ik_retry_n: int = 0,
                    verbose: bool = True) -> Tuple[Optional[str], Optional[_ScoredLayout]]:
    print("=" * 60)
    print(f"  Fast Optimal Layout Search - {task.name}")
    print(f"  (Scoring: GraspCount*{WEIGHT_GRASP_COUNT} + Manip*{WEIGHT_MANIPULABILITY} + Dist*{WEIGHT_DISTANCE})")
    print(f"  pop={pop_size}  elite={elite_n}  gens={generations} seed={rng_seed}")
    print("=" * 60)

    t0 = time.time()
    searcher = FastLayoutSearcher(task, enable_l3=enable_l3, ik_retry_n=ik_retry_n)
    solution = cem_search(
        searcher, pop_size=pop_size, elite_n=elite_n, generations=generations,
        sigma_init=sigma_init, sigma_decay=sigma_decay, try_l3_top_k=try_l3_top_k if searcher.enable_l3 else 0,
        rng_seed=rng_seed, l2_retry=l2_retry, verbose=verbose,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    if solution is not None and solution.l2_pass:
        print(f"[OK] 找到最佳布局，用时 {elapsed:.1f}s")
        print(f"  最终综合得分 score = {solution.layout_score:.2f}")
    else:
        print(f"[FAIL] 搜索失败，用时 {elapsed:.1f}s")
    print("=" * 60)

    out_path: Optional[str] = None
    if solution is not None and solution.l2_pass:
        out_dir = os.path.join(os.path.dirname(__file__), "_output")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "" if solution.l3_pass else ("" if not searcher.enable_l3 else "_l2only")
        out_path = os.path.join(out_dir, f"{task.output_layout_name}{suffix}.layout")
        layout = WorkspaceLayout(
            robot_base_pos=task.robot_base_pos.copy(),
            robot_base_rotmat=task.robot_base_rotmat.copy(),
            assembly_station_pos=task.fixture_pos.copy(),
            assembly_station_rotmat=task.fixture_rotmat.copy(),
            staging_positions={pid: (np.array([float(solution.xy[pid][0]), float(solution.xy[pid][1]), 0.0]), np.eye(3))
                               for pid in searcher.search_part_ids if pid in solution.xy},
            name=f"{task.output_layout_name}{suffix}",
            metadata={
                "search_method": "cem_optimal",
                # 关键：标注本布局是为哪种机器人搜出来的。下游
                # ``dual_sequence_execution.py`` 加载时会校验该字段，
                # 不匹配则丢弃缓存重新走 DFS / 重搜。
                "robot_type": "panthera_ht",
                "arm_y_offset": float(_DUAL_ARM_Y_OFFSET),
                "weights": {
                    "grasp_count": float(WEIGHT_GRASP_COUNT),
                    "manipulability": float(WEIGHT_MANIPULABILITY),
                    "distance": float(WEIGHT_DISTANCE),
                },
                "arm_choice": dict(solution.arm_choice),
                "grasp_counts": dict(solution.grasp_counts),
                "manipulability_avg": float(solution.manipulability_avg),
                "distance_cost": float(solution.distance_cost),
                "total_score": float(solution.layout_score),
            },
        )
        layout.save(out_path)
        print(f"\n[OK] 保存 optimal layout -> {os.path.relpath(out_path)}")

    return out_path, solution


def main():
    parser = argparse.ArgumentParser(description="Fast Optimal Layout Search")
    parser.add_argument("--pop", type=int, default=24, help="每代候选数")
    parser.add_argument("--gens", type=int, default=8, help="最大代数")
    args = parser.parse_args()

    task = YUANCHAIR_FAST_TASK
    run_fast_search(
        task,
        pop_size=args.pop,
        generations=args.gens,
        enable_l3=True,        # 开启真实 RRT 校验（防坑神器）
        try_l3_top_k=3         # 每代取前 3 名进行试跑，只要一个跑通就直接采用
    )

if __name__ == "__main__":
    main()