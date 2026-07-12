"""塔式装配初始布局的全局搜索。

搜索分为两步：
1. 在各装配区域内采样无碰撞布局，保留得分较高的候选；
2. 对候选布局逐零件调整 XY 位置，并用逐级减小的步长做局部优化。

布局评估、缓存、区域处理和 L3 验证沿用 NSGA2LayoutSearcher。

示例：
python -m sealp.examples.layout.find_optimal_initial_layout_tower_global \
    --n-samples 60 --cdprim-type box --output-name tower_global \
    --global-explore 60 --global-elite 3 \
    --global-refine-steps 0.03,0.015,0.008 --global-refine-rounds 2 \
    --global-max-evals 300
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 支持以模块方式运行。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import find_optimal_initial_layout_tower_strict as fol
import find_optimal_initial_layout_tower_strict_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2

LayoutCandidate = fol.LayoutCandidate


# 全局搜索参数，可由命令行覆盖。
GCFG: Dict[str, object] = {
    "explore": None,             # 探索次数，None 时使用 --n-samples。
    "elite": 3,                  # 进入局部优化的候选数。
    "refine_steps": [0.03, 0.015, 0.008],  # XY 优化步长，单位 m。
    "refine_rounds": 2,          # 每个步长的最大扫描轮数。
    "refine_diagonal": False,    # 是否检查对角方向。
    "refine_enabled": True,      # 使用 --no-refine 可关闭。
    "max_resample_layout": 80,   # 单个布局的最大重采样次数。
}


def _offsets(step: float, diagonal: bool) -> List[Tuple[float, float]]:
    base = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
    if diagonal:
        base += [(step, step), (step, -step), (-step, step), (-step, -step)]
    return base


class GlobalLayoutSearcher(nsga2.NSGA2LayoutSearcher):
    """全局采样与局部坐标优化。"""

    # 全局采样
    def _global_explore(self,
                        rng: np.random.Generator,
                        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
                        n_explore: int,
                        verbose: bool) -> List[LayoutCandidate]:
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        attempts = 0
        max_attempts = n_explore * 4 + 20
        i = 0
        while evaluated < n_explore and attempts < max_attempts:
            if self._eval_budget_exhausted():
                print("[global] explore 停止: 达到 --global-max-evals。")
                break
            attempts += 1
            # 按顺序轮询各装配区域。
            region = regions[i % len(regions)]
            i += 1
            self._set_region_from_tuple(region)
            xy = None
            for _ in range(int(GCFG["max_resample_layout"])):
                xy = self.sample_collision_free_xy(rng)
                if xy is not None:
                    break
            if xy is None:
                continue
            cand = self._evaluate_gene(xy, region)
            evaluated += 1
            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                score = float(getattr(cand, "layout_score", -1.0))
                print(f"[explore] {evaluated:03d}/{n_explore} {tag:5s} "
                      f"score={score:.4f} region={cand.assembly_region_id}")
                if not cand.l2_pass:
                    print(f"          fail: {getattr(cand, 'fail_reason', '?')}")
            if cand.l2_pass:
                feasible.append(cand)
        return feasible

    # 局部坐标优化
    def _pattern_refine(self,
                        cand: LayoutCandidate,
                        steps: Sequence[float],
                        rounds: int,
                        diagonal: bool,
                        verbose: bool) -> LayoutCandidate:
        region = getattr(cand, "_nsga_region_tuple", None)
        if region is None:
            region = self._region_by_id(self._assembly_region_candidates(), cand.assembly_region_id)

        first_pid = self._first_part_id() if self.preassemble_first_part else None
        free_parts = [p for p in self.part_order if p != first_pid]

        best = cand
        best_xy = nsga2._copy_xy(cand.xy)
        eps = 1e-4
        total_improved = 0.0

        for step in steps:
            for _r in range(max(1, rounds)):
                improved = False
                for pid in free_parts:
                    if pid not in best_xy:
                        continue
                    if self._eval_budget_exhausted():
                        if verbose:
                            print("[refine] 停止: 达到 --global-max-evals。")
                        return best
                    anchor = np.asarray(best_xy[pid], dtype=float)
                    for dx, dy in _offsets(step, diagonal):
                        trial_xy = nsga2._copy_xy(best_xy)
                        trial_xy[pid] = self._clip_xy_for_part(
                            pid, anchor + np.array([dx, dy], dtype=float))
                        child = self._evaluate_gene(trial_xy, region)
                        if bool(getattr(child, "l2_pass", False)) and \
                                float(child.layout_score) > float(best.layout_score) + eps:
                            total_improved += float(child.layout_score) - float(best.layout_score)
                            best = child
                            best_xy = nsga2._copy_xy(child.xy)
                            anchor = np.asarray(best_xy[pid], dtype=float)
                            improved = True
                if not improved:
                    break  # 当前步长没有改进。
        if verbose:
            print(f"[refine] region={best.assembly_region_id} "
                  f"score {float(cand.layout_score):.4f} -> {float(best.layout_score):.4f} "
                  f"(+{total_improved:.4f})")
        return best

    # 装配区域排序
    def _order_regions_center_first(
        self,
        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
        verbose: bool = True,
    ) -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
        """按候选区域到工作区参考中心的距离排序。"""
        regions = list(regions)
        if len(regions) <= 1:
            return regions

        pts = np.array([[float(p[0]), float(p[1])] for _, _, p in regions], dtype=float)
        ref_x = float(np.mean(pts[:, 0]))
        try:
            arm_ys = [float(xy[1]) for xy in self._arm_base_xy_map().values()]
            ref_y = float(np.mean(arm_ys)) if arm_ys else float(np.mean(pts[:, 1]))
        except Exception:
            ref_y = float(np.mean(pts[:, 1]))
        ref = np.array([ref_x, ref_y], dtype=float)

        d2 = np.sum((pts - ref) ** 2, axis=1)
        # 距离相同时保留原顺序。
        order = sorted(range(len(regions)), key=lambda i: (float(d2[i]), i))
        ordered = [regions[i] for i in order]

        if verbose:
            print("\n---------- Explore order: center-first ----------")
            print(f"reference center (x, y) = ({ref_x:.4f}, {ref_y:.4f})  "
                  f"# x=候选中心均值, y=双臂基座中点")
            for rank, i in enumerate(order, 1):
                rid, rc, p = regions[i]
                print(f"  #{rank:2d} {rid:6s} rc={rc} "
                      f"center=({float(p[0]):.4f}, {float(p[1]):.4f}) "
                      f"dist={float(np.sqrt(d2[i])):.4f}")
        return ordered

    # 搜索入口
    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "staging_aware",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        rng = np.random.default_rng(seed)
        self._reset_eval_progress_stats()
        regions = self._assembly_region_candidates()
        # 优先检查靠近工作区中心的区域。
        regions = self._order_regions_center_first(regions, verbose=verbose)
        GCFG["max_resample_layout"] = int(max_resample_layout)

        n_explore = int(GCFG["explore"] or n_samples)
        elite_k = max(1, int(GCFG["elite"]))
        steps = [float(s) for s in GCFG["refine_steps"]]  # type: ignore[union-attr]
        rounds = int(GCFG["refine_rounds"])
        diagonal = bool(GCFG["refine_diagonal"])
        refine_enabled = bool(GCFG["refine_enabled"]) and len(steps) > 0 and rounds > 0

        print("\n========== Global Layout Search ==========")
        print(f"explore samples   = {n_explore}  (center-first round-robin over {len(regions)} assembly regions)")
        print(f"elite (refine)    = {elite_k}")
        print(f"refine enabled    = {refine_enabled}")
        print(f"refine steps (m)  = {steps if refine_enabled else 'SKIPPED'}")
        print(f"refine rounds     = {rounds if refine_enabled else 'SKIPPED'}, diagonal = {diagonal}")
        print(f"max_evals         = {nsga2._CFG.get('max_evals')}")
        print(f"objective         = analytic layout_score (与 nsga2 BEST-L2 同口径)")
        print(f"L3 default/current = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()

        # 全局采样
        print("\n---------- Phase A: global explore ----------")
        feasible = self._global_explore(rng, regions, n_explore, verbose)
        if not feasible:
            print("\n[FAIL] Phase A 未找到任何 L2 可行布局。请放宽约束或增大 --global-explore。")
            return None

        elites = self._unique_elites(feasible, limit=max(elite_k, int(l3_top_k)))
        print(f"\nPhase A 完成: 可行 {len(feasible)} 个, 去重精英 {len(elites)} 个。")
        for r, c in enumerate(elites[:elite_k], 1):
            print(f"  elite#{r} score={c.layout_score:.4f} region={c.assembly_region_id}")

        # 对高分候选做局部优化。
        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: pattern refine ----------")
            for r, c in enumerate(elites[:elite_k], 1):
                if self._eval_budget_exhausted():
                    print("[global] 精修阶段停止: 达到 --global-max-evals。")
                    refined.append(c)
                    continue
                print(f"[refine] elite#{r} (start score={c.layout_score:.4f}) ...")
                refined.append(self._pattern_refine(c, steps, rounds, diagonal, verbose))
        else:
            print("\n---------- Phase B: SKIPPED (--no-refine) ----------")

        # 合并候选并选择最高分布局。
        pool = list(feasible) + list(refined)
        pool = [c for c in pool if bool(getattr(c, "l2_pass", False))]
        all_elites = self._unique_elites(pool, limit=max(int(GCFG["elite"]), int(l3_top_k)))
        if not all_elites:
            print("\n[FAIL] 无 L2 可行布局。")
            return None
        elites_by_score = sorted(all_elites, key=lambda c: c.layout_score, reverse=True)
        best = elites_by_score[0]

        print("\n========== Global Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"eval cache hits     = {self._nsga_cache_hits}")
        print(f"feasible found      = {len(feasible)}")
        self.print_search_eval_progress()
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} rc={best.assembly_region_rc}")
        print(f"  objectives  ={np.round(getattr(best, '_nsga_objectives', []), 4).tolist()}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")
        print(f"  pose_tag    ={best.pose_tag}")

        # 可选的完整序列验证。
        if enable_l3:
            print("\n========== Optional L3 full-process validation ==========")
            k = min(int(l3_top_k), len(elites_by_score))
            for rank, cand in enumerate(elites_by_score[:k], start=1):
                print(f"[L3] rank {rank}/{k} score={cand.layout_score:.4f} region={cand.assembly_region_id}")
                if self.validate_full_sequence_l3(cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed rank={rank}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k all failed.")
                return None
            print("\n[WARN] L3 failed, require_l3=False, fallback to L2 best.")

        return best


# 命令行入口

def _consume_global_args() -> None:
    v = fast._consume_extra_value("--global-explore")
    if v is not None:
        GCFG["explore"] = int(v)
    v = fast._consume_extra_value("--global-elite")
    if v is not None:
        GCFG["elite"] = int(v)
    v = fast._consume_extra_value("--global-refine-rounds")
    if v is not None:
        GCFG["refine_rounds"] = int(v)
    v = fast._consume_extra_value("--global-refine-steps")
    if v is not None:
        v_norm = str(v).strip().lower()
        if v_norm in ("none", "off", "skip", "0"):
            GCFG["refine_enabled"] = False
        else:
            try:
                GCFG["refine_steps"] = [float(s) for s in v.replace(",", " ").split()]
            except ValueError:
                print(f"[global] WARN: 无法解析 --global-refine-steps '{v}', 保留默认。")
    v = fast._consume_extra_value("--global-max-evals")
    if v is not None:
        try:
            nsga2._CFG["max_evals"] = int(v)
        except ValueError:
            print(f"[global] WARN: 无法解析 --global-max-evals '{v}', 忽略。")
    if fast._consume_extra_flag("--global-refine-diagonal"):
        GCFG["refine_diagonal"] = True
    if fast._consume_extra_flag("--no-refine"):
        GCFG["refine_enabled"] = False


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = GlobalLayoutSearcher


def main() -> None:
    # L3 默认设置与 NSGA-II 脚本保持一致。
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    _consume_global_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    print("[global] config:")
    for k, val in GCFG.items():
        print(f"    {k:20s} = {val}")
    print(f"    {'max_evals':20s} = {nsga2._CFG.get('max_evals')}")
    print(f"[global] scipy.cKDTree available = {fast._HAS_KDTREE}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[global] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
