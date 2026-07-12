"""神经网络辅助全局->局部布局搜索 (Neural-Guided Global-to-Local Layout Search)。

基于 find_optimal_initial_layout_tower_global.py 改造, **不破坏原脚本**。

核心原则:
    - 神经网络只做 proposal / pre-filter / ranking / score prediction;
    - 神经网络绝不决定最终可行性;
    - 所有候选最终都必须经过原始 evaluate_layout + pattern_refine + L3 (motion-level);
    - NN 产出的可行 layout 不足时, fallback 到原始 GlobalLayoutSearcher._global_explore。

流程:
    1. 加载训练好的模型 checkpoint;
    2. 获取 assembly regions / part_order / workspace bounds / part features;
    3. NN 生成 (generator) 或筛选 (scorer) top-K candidate layouts;
    4. 对 top-K 调用原始 evaluate_layout;
    5. feasible 加入 elite pool;
    6. NN feasible 不足则 fallback 到原始 global explore;
    7. 对 elite pool 调用原来的 pattern_refine;
    8. 排序; 9. motion-level (L3) validation; 10. 保存最终 layout。

用法示例:
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_neural \
        --model sagpn --checkpoint checkpoints/layout_models/sagpn_best.pt \
        --top-k-proposals 64 --global-elite 5 \
        --global-refine-steps 0.03,0.015,0.008 --global-max-evals 300 \
        --output-name tower_neural_sagpn
    # baseline: --model mlp/deepsets/gcn/gat/cvae/diffusion + 对应 checkpoint
    # 加速: --no-refine 跳过 Phase B; 或 --global-refine-steps 0.03 --global-refine-rounds 1 --global-elite 1
    #
    # 装配站采样模式 (--station-mode):
    #   continuous (默认) 连续可行域采样, SAGPN 回归站位 + 邻域抖动;
    #   grid3x3          复用原始 3x3 网格 center-first 站位 (SAGPN 仍在各站位做零件提案),
    #                    便于与 find_optimal_initial_layout_tower_global.py 在同口径下公平对比。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import find_optimal_initial_layout_tower_strict as fol
import find_optimal_initial_layout_tower_strict_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod
import generate_layout_dataset as gends

from layout_learning.infer import LayoutModelRunner

LayoutCandidate = fol.LayoutCandidate

# 神经搜索配置
NCFG: Dict[str, object] = {
    "model": "sagpn",
    "checkpoint": None,
    "top_k_proposals": 64,      # 交给 evaluate_layout 的候选 layout 数量
    "scorer_pool": 400,         # scorer 模型: NN 预筛前的候选池大小
    "min_feasible": 1,          # NN 找到的可行 layout 少于此数则触发 fallback
    "fallback_explore": 40,     # fallback 时 global explore 的评估次数
    "device": None,
    "feas_prob_min": 0.0,       # scorer: 过滤掉预测可行概率过低的候选 (0=不过滤)
    # 装配站采样模式:
    #   "continuous" -> 连续可行域采样 (SAGPN 回归站位 + 邻域抖动);
    #   "grid3x3"    -> 复用原始 3x3 网格 center-first 站位 (SAGPN 仍在各站位上做零件提案)。
    "station_mode": "continuous",
    # scorer 排序方式:
    #   "tuple" -> 先按 (feas_prob>=min, feas_prob, score) 字典序 (默认, 与旧行为一致);
    #   "blend" -> 按 rank_score = w*feas_prob + (1-w)*normalized_score 排序 (seqrel 推荐)。
    "rank_mode": "tuple",
    "rank_blend_feas": 0.7,     # blend 模式下 feasibility 概率的权重
}


def _build_cond(searcher, region: Tuple[str, Tuple[int, int], np.ndarray],
                xy: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """构造某装配区的条件 sample (供 NN 推理)。xy=None 表示只给静态条件。"""
    searcher._set_region_from_tuple(region)
    cand = LayoutCandidate(xy={k: np.asarray(v, float).copy() for k, v in (xy or {}).items()})
    cand.assembly_region_id = region[0]
    cand.assembly_region_rc = region[1]
    cand.assembly_station_pos = np.asarray(region[2], dtype=float)
    return gends.sample_from_candidate(searcher, cand, seed=0, region=region)


class NeuralGlobalSearcher(gmod.GlobalLayoutSearcher):
    """NN 辅助搜索器: 用 NN 产 top-K 候选, 交给原始 evaluate_layout, 保留 global fallback。"""

    _runner: Optional[LayoutModelRunner] = None

    # ---------- 连续装配站工具 (取代固定 3x3 网格) ----------
    def _sample_station(self, rng) -> Optional[Tuple]:
        """连续采样一个可行装配站 (region_id='cont', rc=(-1,-1))。"""
        return gends.sample_continuous_station(self, rng)

    def _sample_station_list(self, rng, n: int) -> List[Tuple]:
        out: List[Tuple] = []
        tries = 0
        while len(out) < n and tries < n * 5 + 20:
            tries += 1
            reg = self._sample_station(rng)
            if reg is not None:
                out.append(reg)
        return out

    def _station_region_from_xy(self, xy) -> Optional[Tuple]:
        """把一个 (可能越界的) 装配站 xy 收缩到可行域并校验, 返回 region 三元组。"""
        xs0, xs1, ys0, ys1 = gends.station_safe_bounds(self)
        x = float(np.clip(float(xy[0]), xs0, xs1))
        y = float(np.clip(float(xy[1]), ys0, ys1))
        pos = np.array([x, y, float(self.table_top_z)], dtype=float)
        if self._assembly_region_reject_reason(pos) is not None:
            return None
        return ("cont", (-1, -1), pos)

    def _sagpn_station_candidates(self, rng, n: int, verbose: bool) -> List[Tuple]:
        """SAGPN: 回归一个连续装配站, 再在其邻域抖动 + 少量随机站位, 兼顾利用与探索。"""
        runner = self._runner
        seed_reg = self._sample_station(rng)
        if seed_reg is None:
            return self._sample_station_list(rng, n)
        cond0 = _build_cond(self, seed_reg)
        pred = None
        try:
            pred = runner.predict_station(cond0)
        except Exception as e:
            print(f"[neural] predict_station 失败, 退回随机连续站位: {e!r}")
        if pred is None:
            return self._sample_station_list(rng, n)
        cands: List[Tuple] = []
        base = self._station_region_from_xy(pred)
        if base is not None:
            cands.append(base)
            if verbose:
                print(f"[gen] SAGPN predicted station = ({pred[0]:.4f}, {pred[1]:.4f})")
        # 邻域抖动 (利用)
        for _ in range(max(0, n // 2)):
            jit = np.asarray(pred, float) + rng.normal(0, 0.03, size=2)
            reg = self._station_region_from_xy(jit)
            if reg is not None:
                cands.append(reg)
        # 少量随机连续站位 (探索)
        cands += self._sample_station_list(rng, max(1, n - len(cands)))
        return cands[:max(1, n)]

    def _grid_station_candidates(self, verbose: bool) -> List[Tuple]:
        """3x3 网格站位: 复用原始 _assembly_region_candidates + center-first 排序。

        注意: 站位固定在网格中心, SAGPN 不再回归站位, 但仍会在**每个网格站位**上
        用 propose_layouts 生成零件布局 (即 SAGPN 依然做提案), 因此这只是把"站位
        搜索空间"从连续退回到离散 3x3, 便于与原始 global 基线在同口径下对比。
        """
        regions = self._assembly_region_candidates()
        return self._order_regions_center_first(regions, verbose=verbose)

    def _station_candidates(self, rng, n: int, prefer_sagpn: bool, verbose: bool) -> List[Tuple]:
        """按 station_mode 统一分发装配站候选。

        prefer_sagpn: 连续模式下, 若模型是 SAGPN 则用其回归站位 + 邻域抖动; 否则随机连续。
        grid3x3 模式忽略 prefer_sagpn (站位由网格决定, SAGPN 仅在站位上做零件提案)。
        """
        mode = str(NCFG.get("station_mode", "continuous"))
        if mode == "grid3x3":
            regions = self._grid_station_candidates(verbose=verbose)
            return regions[:max(1, n)] if regions else []
        # continuous
        runner = self._runner
        if prefer_sagpn and runner is not None and getattr(runner, "model_name", "") == "sagpn":
            return self._sagpn_station_candidates(rng, n, verbose)
        return self._sample_station_list(rng, n)

    # ---------- generator 路径 (连续装配站) ----------
    def _neural_generate(self, rng, verbose) -> List[LayoutCandidate]:
        runner = self._runner
        top_k = int(NCFG["top_k_proposals"])
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        first_pid = self._first_part_id() if self.preassemble_first_part else None

        # 生成一批装配站候选: 由 station_mode 决定 (连续 / 3x3 网格)。
        # 连续 + SAGPN -> 回归站位+邻域; 连续 + 其它 -> 随机连续; grid3x3 -> 网格中心。
        n_stations = max(1, min(top_k, 16))
        stations = self._station_candidates(rng, n_stations, prefer_sagpn=True, verbose=verbose)
        if not stations:
            return feasible
        per_station = max(1, top_k // len(stations) + 1)

        for region in stations:
            if evaluated >= top_k or self._eval_budget_exhausted():
                break
            self._set_region_from_tuple(region)
            cond = _build_cond(self, region)
            station_xy = np.asarray(region[2], float)[:2]
            try:
                layouts = runner.propose_layouts(
                    cond, k=per_station, part_ids=list(self.part_order), station_xy=station_xy)
            except Exception as e:
                print(f"[neural] propose 失败 station={np.round(station_xy,3).tolist()}: {e!r}")
                continue
            for layout_xy in layouts:
                if evaluated >= top_k or self._eval_budget_exhausted():
                    break
                xy = {}
                for pid in self.part_order:
                    if pid == first_pid:
                        continue  # 第一件由 evaluate_layout 预装, 无需 proposal
                    if pid in layout_xy:
                        xy[pid] = self._clip_xy_for_part(pid, np.asarray(layout_xy[pid], float))
                if len(xy) < 1:
                    continue
                cand = self._evaluate_gene(xy, region)
                evaluated += 1
                if verbose:
                    tag = "L2_OK" if cand.l2_pass else "FAIL"
                    print(f"[gen] {evaluated:03d}/{top_k} {tag:5s} "
                          f"score={float(getattr(cand,'layout_score',-1)):.4f} "
                          f"station=({station_xy[0]:.3f},{station_xy[1]:.3f})")
                if cand.l2_pass:
                    feasible.append(cand)
        return feasible

    # ---------- scorer 路径 (连续装配站) ----------
    def _neural_score_filter(self, rng, verbose) -> List[LayoutCandidate]:
        runner = self._runner
        top_k = int(NCFG["top_k_proposals"])
        pool_n = int(NCFG["scorer_pool"])
        feas_min = float(NCFG["feas_prob_min"])

        # 1) 连续采样装配站, 每个站位采样若干无碰撞 layout, 并在同一 station 状态下
        # 直接序列化 (每个 station 只 _set_assembly_station 一次, 避免重复重建 goal_models)。
        n_stations = max(1, min(pool_n, 40))
        stations = self._station_candidates(rng, n_stations, prefer_sagpn=False, verbose=verbose)
        if not stations:
            return []
        per_station = max(1, pool_n // len(stations) + 1)
        pool: List[Tuple[Tuple, Dict[str, np.ndarray]]] = []
        samples: List[Dict] = []
        max_resample = int(gmod.GCFG["max_resample_layout"])
        for region in stations:
            if len(pool) >= pool_n:
                break
            self._set_region_from_tuple(region)
            got = 0
            local_attempts = 0
            while got < per_station and len(pool) < pool_n and local_attempts < per_station * 4 + 20:
                local_attempts += 1
                xy = None
                for _ in range(max_resample):
                    xy = self.sample_collision_free_xy(rng)
                    if xy is not None:
                        break
                if xy is None:
                    continue
                cand = LayoutCandidate(xy={k: np.asarray(v, float).copy() for k, v in xy.items()})
                cand.assembly_region_id = region[0]
                cand.assembly_region_rc = region[1]
                cand.assembly_station_pos = np.asarray(region[2], float)
                samples.append(gends.sample_from_candidate(self, cand, seed=0, region=region))
                pool.append((region, xy))
                got += 1
        if not pool:
            return []

        # 2) NN 打分
        pred = runner.score_layouts(samples)
        feas_prob = pred["feas_prob"]
        score = pred["score"]

        # 3) 排序: tuple(默认) 或 blend(rank_score = w*feas + (1-w)*norm_score)
        rank_mode = str(NCFG.get("rank_mode", "tuple"))
        if rank_mode == "blend":
            w = float(NCFG.get("rank_blend_feas", 0.7))
            s = np.asarray(score, dtype=float)
            s_min, s_max = float(s.min()), float(s.max())
            s_norm = (s - s_min) / (s_max - s_min) if s_max - s_min > 1e-9 else np.zeros_like(s)
            rank_score = w * np.asarray(feas_prob, dtype=float) + (1.0 - w) * s_norm
            order = sorted(range(len(pool)),
                           key=lambda j: (feas_prob[j] >= feas_min, float(rank_score[j])),
                           reverse=True)
            if verbose:
                print(f"[score] rank_mode=blend w_feas={w:.2f}")
        else:
            order = sorted(range(len(pool)),
                           key=lambda j: (feas_prob[j] >= feas_min, feas_prob[j], score[j]),
                           reverse=True)

        # 4) top-K 交给原始 evaluate_layout
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        for j in order:
            if evaluated >= top_k or self._eval_budget_exhausted():
                break
            if feas_min > 0 and feas_prob[j] < feas_min:
                break
            region, xy = pool[j]
            cand = self._evaluate_gene(xy, region)
            evaluated += 1
            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                print(f"[score] {evaluated:03d}/{top_k} {tag:5s} "
                      f"nn_p={feas_prob[j]:.2f} nn_s={score[j]:.3f} "
                      f"real={float(getattr(cand,'layout_score',-1)):.4f} region={cand.assembly_region_id}")
            if cand.l2_pass:
                feasible.append(cand)
        return feasible

    # ---------- 主入口 ----------
    def random_search(self, n_samples, seed, max_resample_layout=80, verbose=True,
                      enable_l3=False, l3_top_k=3, l3_obstacle_mode="staging_aware",
                      require_l3=True) -> Optional[LayoutCandidate]:
        rng = np.random.default_rng(seed)
        self._reset_eval_progress_stats()
        gmod.GCFG["max_resample_layout"] = int(max_resample_layout)

        runner = self._runner
        is_gen = bool(runner is not None and runner.is_generator)

        elite_k = max(1, int(gmod.GCFG["elite"]))
        steps = [float(s) for s in gmod.GCFG["refine_steps"]]
        rounds = int(gmod.GCFG["refine_rounds"])
        diagonal = bool(gmod.GCFG["refine_diagonal"])
        refine_enabled = bool(gmod.GCFG["refine_enabled"]) and len(steps) > 0 and rounds > 0

        print("\n========== Neural-Guided Layout Search ==========")
        print(f"model           = {NCFG['model']}  ({'generator' if is_gen else 'scorer'})")
        print(f"checkpoint      = {NCFG['checkpoint']}")
        print(f"top_k_proposals = {NCFG['top_k_proposals']}")
        print(f"scorer_pool     = {NCFG['scorer_pool']}")
        print(f"station_mode    = {NCFG['station_mode']}")
        print(f"elite (refine)  = {elite_k}, refine enabled={refine_enabled}")
        if refine_enabled:
            print(f"refine steps    = {steps}, rounds={rounds}, diagonal={diagonal}")
        else:
            print(f"refine steps    = SKIPPED (--no-refine)")
        print(f"max_evals       = {nsga2._CFG.get('max_evals')}")
        print(f"L3 default/cur  = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()

        # Phase A': NN 生成/筛选 -> evaluate_layout
        print("\n---------- Phase A': neural proposal / filter ----------")
        if runner is None:
            print("[neural] WARN: 未加载模型, 直接 fallback 到 continuous explore。")
            feasible = []
        elif is_gen:
            feasible = self._neural_generate(rng, verbose)
        else:
            feasible = self._neural_score_filter(rng, verbose)
        print(f"[neural] NN 产出可行 layout = {len(feasible)}")

        # Phase A'' fallback: NN 可行不足 -> 原始 global explore (复用原逻辑, 但用连续装配站)
        if len(feasible) < int(NCFG["min_feasible"]) and not self._eval_budget_exhausted():
            print("\n---------- Fallback: original global explore (continuous stations) ----------")
            fb_regions = self._station_candidates(
                rng, max(1, int(NCFG["fallback_explore"])), prefer_sagpn=False, verbose=verbose)
            if not fb_regions:
                # 极端兜底: 采样失败时退回离散网格候选, 保证一定有 fallback。
                fb_regions = self._order_regions_center_first(
                    self._assembly_region_candidates(), verbose=verbose)
            fb = self._global_explore(rng, fb_regions, int(NCFG["fallback_explore"]), verbose)
            feasible = list(feasible) + list(fb)
            print(f"[neural] fallback 后可行 layout = {len(feasible)}")

        if not feasible:
            print("\n[FAIL] 未找到任何 L2 可行布局 (NN + fallback 均失败)。")
            return None

        elites = self._unique_elites(feasible, limit=max(elite_k, int(l3_top_k)))
        print(f"\nPhase A 完成: 可行 {len(feasible)} 个, 去重精英 {len(elites)} 个。")
        for r, c in enumerate(elites[:elite_k], 1):
            print(f"  elite#{r} score={c.layout_score:.4f} region={c.assembly_region_id}")

        # Phase B: pattern refine (原逻辑不变; 可用 --no-refine 跳过)
        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: pattern refine (original) ----------")
            for r, c in enumerate(elites[:elite_k], 1):
                if self._eval_budget_exhausted():
                    refined.append(c)
                    continue
                print(f"[refine] elite#{r} (start score={c.layout_score:.4f}) ...")
                refined.append(self._pattern_refine(c, steps, rounds, diagonal, verbose))
        else:
            print("\n---------- Phase B: SKIPPED (--no-refine) ----------")

        pool = [c for c in (list(feasible) + list(refined)) if bool(getattr(c, "l2_pass", False))]
        all_elites = self._unique_elites(pool, limit=max(int(gmod.GCFG["elite"]), int(l3_top_k)))
        if not all_elites:
            print("\n[FAIL] 无 L2 可行布局。")
            return None
        elites_by_score = sorted(all_elites, key=lambda c: c.layout_score, reverse=True)
        best = elites_by_score[0]

        print("\n========== Neural Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"eval cache hits     = {self._nsga_cache_hits}")
        print(f"feasible found      = {len(feasible)}")
        self.print_search_eval_progress()
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} rc={best.assembly_region_rc}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")

        # motion-level (L3) validation (原逻辑不变)
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


# ============================================================
# Entry point
# ============================================================

def _consume_neural_args() -> None:
    v = fast._consume_extra_value("--model")
    if v is not None:
        NCFG["model"] = str(v)
    v = fast._consume_extra_value("--checkpoint")
    if v is not None:
        NCFG["checkpoint"] = str(v)
    v = fast._consume_extra_value("--top-k-proposals")
    if v is not None:
        NCFG["top_k_proposals"] = int(v)
    v = fast._consume_extra_value("--scorer-pool")
    if v is not None:
        NCFG["scorer_pool"] = int(v)
    v = fast._consume_extra_value("--min-feasible")
    if v is not None:
        NCFG["min_feasible"] = int(v)
    v = fast._consume_extra_value("--fallback-explore")
    if v is not None:
        NCFG["fallback_explore"] = int(v)
    v = fast._consume_extra_value("--nn-device")
    if v is not None:
        NCFG["device"] = str(v)
    v = fast._consume_extra_value("--feas-prob-min")
    if v is not None:
        NCFG["feas_prob_min"] = float(v)
    v = fast._consume_extra_value("--station-mode")
    if v is not None:
        mode = str(v).strip().lower()
        if mode not in ("continuous", "grid3x3"):
            print(f"[neural] WARN: 未知 --station-mode '{v}', 回退 continuous。")
            mode = "continuous"
        NCFG["station_mode"] = mode
    v = fast._consume_extra_value("--rank-mode")
    if v is not None:
        rm = str(v).strip().lower()
        if rm not in ("tuple", "blend"):
            print(f"[neural] WARN: 未知 --rank-mode '{v}', 回退 tuple。")
            rm = "tuple"
        NCFG["rank_mode"] = rm
    v = fast._consume_extra_value("--rank-blend-feas")
    if v is not None:
        NCFG["rank_blend_feas"] = float(v)


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = NeuralGlobalSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    gmod._consume_global_args()
    _consume_neural_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    # 加载模型 checkpoint (若提供)。加载失败时仍可运行 (纯 fallback)。
    runner = None
    if NCFG["checkpoint"]:
        try:
            runner = LayoutModelRunner(str(NCFG["checkpoint"]), device=NCFG["device"])
            NCFG["model"] = runner.model_name
            print(f"[neural] loaded checkpoint: {NCFG['checkpoint']} "
                  f"(model={runner.model_name}, generator={runner.is_generator})")
        except Exception as e:
            print(f"[neural] WARN: 加载 checkpoint 失败, 将纯 fallback: {e!r}")
    else:
        print("[neural] 未提供 --checkpoint, 将纯 fallback 到 global explore。")

    NeuralGlobalSearcher._runner = runner

    print("[neural] config:")
    for k, val in NCFG.items():
        print(f"    {k:16s} = {val}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[neural] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
