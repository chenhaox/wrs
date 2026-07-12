"""统一训练循环 + 指标 + best checkpoint。

供 ``train_layout_network.py`` 调用, 也可直接:
    from layout_learning.train import train_model

设计要点 (为修复小数据集上 SAGPN 崩溃):
    - stratified / seed_holdout / random 三种 split;
    - score regression 只在 feasible 样本上算;
    - xy / station proposal loss 只在 "elite feasible" (>= 分位阈值) 上算;
    - 丰富验证指标: ROC-AUC / PR-AUC / recall@K / precision@K /
      top-K avg true score / score MAE·RMSE·Spearman / enrichment factor;
    - best checkpoint 用 composite metric, 不再只看 topk_hit;
    - early stopping + ReduceLROnPlateau。
"""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import features as F
from .dataset import LayoutDataset, collate_items, move_batch, load_jsonl
from .losses import LossWeights, compute_loss
from .models import build_model, is_generator


# ------------------------------------------------------------
# 指标 (全部无 sklearn 依赖)
# ------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC (Mann-Whitney U)。"""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels > 0.5].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC = average precision (按预测概率降序累积)。"""
    n_pos = int((labels > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = (labels[order] > 0.5).astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / n_pos
    # AP = sum over thresholds (R_i - R_{i-1}) * P_i
    prev_r = 0.0
    ap = 0.0
    for p_i, r_i in zip(precision, recall):
        ap += (r_i - prev_r) * p_i
        prev_r = r_i
    return float(ap)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关 = 秩上的 Pearson。"""
    if len(a) < 3:
        return float("nan")

    def _rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x))
        return r

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom < 1e-9:
        return float("nan")
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate(model, loader, device, gen: bool, topk: int = 10) -> Dict[str, float]:
    model.eval()
    feas_logits, feas_labels, score_preds, score_tgts = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        feas_logits.append(out["feas_logit"].cpu().numpy())
        feas_labels.append(batch["feas"].cpu().numpy())
        score_preds.append(out["score_pred"].cpu().numpy())
        score_tgts.append(batch["score"].cpu().numpy())

    fl = np.concatenate(feas_logits)
    fy = np.concatenate(feas_labels)
    sp = np.concatenate(score_preds)
    st = np.concatenate(score_tgts)          # 真 layout_score (infeasible=0)
    prob = 1.0 / (1.0 + np.exp(-fl))
    n = len(prob)
    val_feas_rate = float((fy > 0.5).mean()) if n else float("nan")

    acc = float(((prob > 0.5) == (fy > 0.5)).mean()) if n else float("nan")
    roc = _roc_auc(prob, fy)
    pr = _pr_auc(prob, fy)

    # score 指标: 只在 feasible 上算
    fmask = fy > 0.5
    if fmask.sum() >= 1:
        mae = float(np.abs(sp[fmask] - st[fmask]).mean())
        rmse = float(np.sqrt(((sp[fmask] - st[fmask]) ** 2).mean()))
    else:
        mae = rmse = float("nan")
    spearman = _spearman(sp[fmask], st[fmask]) if fmask.sum() >= 3 else float("nan")

    # top-K (按预测可行概率排序)
    k = int(min(topk, n))
    if k > 0:
        top_prob = np.argsort(-prob)[:k]
        topk_hit = float(fy[top_prob].mean())                 # = precision@K
        precision_at_k = topk_hit
        n_pos = float((fy > 0.5).sum())
        recall_at_k = float(fy[top_prob].sum() / n_pos) if n_pos > 0 else float("nan")
        enrichment = (precision_at_k / val_feas_rate) if val_feas_rate > 1e-9 else float("nan")
    else:
        topk_hit = precision_at_k = recall_at_k = enrichment = float("nan")

    # top-K 真分数 (按预测 score 排序) 以及归一化 (相对 oracle top-K)
    if k > 0:
        top_score = np.argsort(-sp)[:k]
        topk_avg_true = float(st[top_score].mean())
        oracle = float(np.sort(st)[::-1][:k].mean())
        topk_avg_norm = float(topk_avg_true / oracle) if oracle > 1e-9 else 0.0
    else:
        topk_avg_true = topk_avg_norm = float("nan")

    def _z(v):  # nan -> 0 (用于 composite)
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    composite = (0.4 * _z(pr) + 0.3 * _z(recall_at_k)
                 + 0.2 * _z(topk_avg_norm) + 0.1 * _z(spearman))

    return {
        "feas_acc": acc,
        "auc": roc,               # 保留旧名 (= ROC-AUC)
        "roc_auc": roc,
        "pr_auc": pr,
        "score_mae": mae,
        "score_rmse": rmse,
        "score_spearman": spearman,
        "topk_hit": topk_hit,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "topk_avg_true_score": topk_avg_true,
        "topk_avg_score_norm": topk_avg_norm,
        "enrichment": enrichment,
        "composite": composite,
    }


# ------------------------------------------------------------
# 实验元数据 / 保存
# ------------------------------------------------------------

def _git_commit() -> str:
    """返回当前 git commit hash; 失败时返回 'unknown'。"""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def experiment_save_dir(base_dir: str, model_name: str, split_mode: str,
                        training_seed: int) -> str:
    """规范实验目录: {base}/{model}/{split}/seed{N}/。"""
    return os.path.join(base_dir, model_name, split_mode, f"seed{training_seed}")


def _save_training_artifacts(save_dir: str, model_name: str, *,
                             config: Dict, metrics: Dict, history: List[Dict],
                             train_indices: List[int], val_indices: List[int],
                             sample_ids: List) -> None:
    """保存 config / metrics / history CSV / split indices; 复制 train.log。"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    split_payload = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_ids": [sample_ids[i] for i in train_indices],
        "val_sample_ids": [sample_ids[i] for i in val_indices],
    }
    with open(os.path.join(save_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)
    if history:
        hist_path = os.path.join(save_dir, "training_history.csv")
        keys = list(history[0].keys())
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in history:
                w.writerow(row)
    src_log = os.path.join(save_dir, f"{model_name}_train.log")
    dst_log = os.path.join(save_dir, "train.log")
    if os.path.isfile(src_log) and not os.path.isfile(dst_log):
        with open(src_log, "r", encoding="utf-8") as sf, \
             open(dst_log, "w", encoding="utf-8") as df:
            df.write(sf.read())


# ------------------------------------------------------------
# train / val split
# ------------------------------------------------------------

def _region_bucket(sample: Dict, mode: str, xy_grid: int,
                   xy_bounds: Tuple[float, float, float, float]) -> str:
    """给一条样本分配 region 桶标签 (供 region_holdout split 使用)。

    mode:
        rc      -> 用 assembly_region_rc (离散 3x3 网格)。
        xy_grid -> 用 assembly_station_pos 落在 xy_grid×xy_grid 网格的格子。
        auto    -> rc 有效 (非 [-1,-1]) 用 rc, 否则退回 xy_grid。
    """
    rc = sample.get("assembly_region_rc", [-1, -1])
    rc_valid = int(rc[0]) >= 0 and int(rc[1]) >= 0
    use = mode
    if mode == "auto":
        use = "rc" if rc_valid else "xy_grid"
    if use == "rc" and rc_valid:
        return f"rc_{int(rc[0])}_{int(rc[1])}"
    # xy_grid
    pos = np.asarray(sample.get("assembly_station_pos", [0.0, 0.0, 0.0]), dtype=float)
    xlo, xhi, ylo, yhi = xy_bounds
    gx = int(np.clip((pos[0] - xlo) / max(xhi - xlo, 1e-6) * xy_grid, 0, xy_grid - 1))
    gy = int(np.clip((pos[1] - ylo) / max(yhi - ylo, 1e-6) * xy_grid, 0, xy_grid - 1))
    return f"xy_{gx}_{gy}"


def _split_indices(samples: List[Dict], val_ratio: float, seed: int,
                   mode: str, region_holdout_mode: str = "auto",
                   xy_grid: int = 3) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    n = len(samples)
    idx = np.arange(n)

    if mode == "region_holdout":
        st = np.array([[float(v) for v in (s.get("assembly_station_pos") or [0, 0, 0])[:2]]
                       for s in samples], dtype=float)
        xy_bounds = (float(st[:, 0].min()), float(st[:, 0].max()),
                     float(st[:, 1].min()), float(st[:, 1].max())) if len(st) else (0, 1, 0, 1)
        buckets = np.array([_region_bucket(s, region_holdout_mode, xy_grid, xy_bounds)
                            for s in samples])
        uniq = sorted(set(buckets.tolist()))
        if len(uniq) >= 2:
            rng.shuffle(uniq)
            n_hold = max(1, int(round(len(uniq) * val_ratio)))
            hold = set(uniq[:n_hold])
            val = idx[np.isin(buckets, list(hold))].tolist()
            train = idx[~np.isin(buckets, list(hold))].tolist()
            if train and val:
                return sorted(train), sorted(val)
        # 桶不足 -> 退回 stratified
        mode = "stratified"

    if mode == "seed_holdout":
        seeds = np.array([int(s.get("seed", 0)) for s in samples])
        uniq = sorted(set(seeds.tolist()))
        if len(uniq) >= 2:
            n_hold = max(1, int(round(len(uniq) * val_ratio)))
            hold = set(uniq[-n_hold:])
            val = idx[np.isin(seeds, list(hold))].tolist()
            train = idx[~np.isin(seeds, list(hold))].tolist()
            if train and val:
                return train, val
        # 只有 1 个 seed -> 退回 stratified
        mode = "stratified"

    if mode == "stratified":
        feas = np.array([bool(s.get("l2_pass", False)) for s in samples])
        train, val = [], []
        for cls in (True, False):
            cls_idx = idx[feas == cls]
            rng.shuffle(cls_idx)
            n_val = int(round(len(cls_idx) * val_ratio))
            if len(cls_idx) >= 2:
                n_val = max(1, min(n_val, len(cls_idx) - 1))
            val.extend(cls_idx[:n_val].tolist())
            train.extend(cls_idx[n_val:].tolist())
        if train and val:
            return sorted(train), sorted(val)
        mode = "random"

    # random
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1) if n >= 2 else 0
    val = idx[:n_val].tolist()
    train = idx[n_val:].tolist()
    return sorted(train), sorted(val)


# ------------------------------------------------------------
# 训练
# ------------------------------------------------------------

def _group_grad_norms(model) -> Dict[str, float]:
    """按参数名首段分组统计 grad L2 范数 (用于 --debug-grad)。"""
    groups: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        head = name.split(".")[0]
        groups[head] = groups.get(head, 0.0) + float(p.grad.detach().pow(2).sum().item())
    return {k: float(np.sqrt(v)) for k, v in groups.items()}


def train_model(dataset_path: str,
                model_name: str,
                save_dir: str,
                epochs: int = 100,
                batch_size: int = 64,
                lr: float = 1e-3,
                val_ratio: float = 0.15,
                weight_decay: float = 1e-5,
                seed: int = 0,
                device: Optional[str] = None,
                loss_weights: Optional[LossWeights] = None,
                pos_weight=None,
                model_kwargs: Optional[Dict] = None,
                max_parts: int = F.MAX_PARTS_DEFAULT,
                topk: int = 10,
                split_mode: str = "stratified",
                region_holdout_mode: str = "auto",
                region_xy_grid: int = 3,
                feature_version: str = "v1",
                elite_quantile: Optional[float] = None,
                early_stop_patience: int = 0,
                early_stop_metric: str = "composite",
                min_delta: float = 1e-4,
                debug_grad: bool = False,
                debug_batches: int = 3,
                limit_samples: int = 0,
                shuffle_labels: bool = False,
                verbose: bool = True) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = load_jsonl(dataset_path)
    if not samples:
        raise RuntimeError(f"数据集为空: {dataset_path}")
    if limit_samples and limit_samples > 0 and limit_samples < len(samples):
        # 分层截断: 尽量保留正/负样本各一部分 (overfit smoke test 用)。
        rng_lim = np.random.RandomState(seed)
        pos = [s for s in samples if s.get("l2_pass", False)]
        neg = [s for s in samples if not s.get("l2_pass", False)]
        rng_lim.shuffle(pos)
        rng_lim.shuffle(neg)
        n_pos_keep = max(1, min(len(pos), limit_samples // 2))
        n_neg_keep = max(1, limit_samples - n_pos_keep)
        samples = pos[:n_pos_keep] + neg[:n_neg_keep]
        rng_lim.shuffle(samples)
        if verbose:
            print(f"[train] limit_samples={limit_samples} -> 实际 {len(samples)} "
                  f"(pos={n_pos_keep} neg={min(len(neg), n_neg_keep)})")
    if shuffle_labels:
        # 标签置换 sanity check: 打乱 l2_pass / layout_score, 模型不应还能拿到正常指标。
        rng_sh = np.random.RandomState(seed + 12345)
        perm = rng_sh.permutation(len(samples))
        feas_shuf = [bool(samples[i].get("l2_pass", False)) for i in perm]
        score_shuf = [float(samples[i].get("layout_score", 0.0)) for i in perm]
        for s, f_new, sc_new in zip(samples, feas_shuf, score_shuf):
            s["l2_pass"] = f_new
            s["layout_score"] = sc_new if f_new else 0.0
        if verbose:
            print("[train] shuffle_labels=True -> 标签已随机置换 (sanity check)")
    ds = LayoutDataset(samples, max_parts=max_parts, feature_version=feature_version)

    train_idx, val_idx = _split_indices(samples, val_ratio, seed, split_mode,
                                        region_holdout_mode=region_holdout_mode,
                                        xy_grid=region_xy_grid)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_items, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_items)

    flat_dim = F.flatten_feature_dim(max_parts)
    gen = is_generator(model_name)
    model = build_model(model_name, flat_dim=flat_dim, **(model_kwargs or {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(3, early_stop_patience // 3 or 5),
        min_lr=1e-6)
    # 使用深拷贝，避免 auto pos_weight 修改调用方复用的 LossWeights 对象。
    weights = copy.deepcopy(loss_weights) if loss_weights is not None else LossWeights()

    # ---- elite 分位阈值: 用训练集 feasible 分数的分位数 ----
    n_feas = sum(1 for s in samples if s.get("l2_pass", False))
    if elite_quantile is not None:
        train_feas_scores = np.array(
            [float(samples[i].get("layout_score", 0.0)) for i in train_idx
             if samples[i].get("l2_pass", False)])
        if len(train_feas_scores) >= 1:
            weights.score_threshold = float(np.percentile(train_feas_scores,
                                                          elite_quantile * 100.0))
        if verbose:
            print(f"[train] elite quantile={elite_quantile:.2f} "
                  f"-> score_threshold={weights.score_threshold:.4f} "
                  f"(train feasible={len(train_feas_scores)})")

    # ---- pos_weight: 仅根据当前训练 split 自动计算，避免验证集泄漏 ----
    n_pos = sum(1 for i in train_idx if samples[i].get("l2_pass", False))
    n_neg = len(train_idx) - n_pos
    suggested_pos_weight = float(n_neg / max(n_pos, 1))

    requested_pos_weight = weights.pos_weight if pos_weight is None else pos_weight
    if isinstance(requested_pos_weight, str):
        spec = requested_pos_weight.strip().lower()
        if spec == "auto":
            effective_pos_weight = suggested_pos_weight if n_pos > 0 else 1.0
        else:
            try:
                effective_pos_weight = float(spec)
            except ValueError as exc:
                raise ValueError(
                    "pos_weight 必须是 'auto' 或正浮点数，例如 2.5"
                ) from exc
    else:
        effective_pos_weight = float(requested_pos_weight)

    if not np.isfinite(effective_pos_weight) or effective_pos_weight <= 0:
        raise ValueError(f"effective pos_weight 必须为正有限值，得到 {effective_pos_weight}")
    weights.pos_weight = float(effective_pos_weight)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] model={model_name} generator={gen} params={n_params:,} "
              f"size={model_kwargs}")
        print(f"[train] samples={len(ds)} (feasible={n_feas}) train={len(train_idx)} "
              f"val={len(val_idx)} split={split_mode} feature={feature_version} "
              f"device={device} flat_dim={flat_dim}")
        if feature_version == "v2":
            print("[train] NOTE: 使用 v2 自适应归一化特征, 与 v1 checkpoint 不兼容。")
        mode_text = "auto" if isinstance(requested_pos_weight, str) and requested_pos_weight.strip().lower() == "auto" else "manual"
        print(f"[train] train pos/neg={n_pos}/{n_neg} "
              f"suggested={suggested_pos_weight:.4f} "
              f"requested={requested_pos_weight} "
              f"effective={weights.pos_weight:.4f} mode={mode_text}")
        # 小数据 + 生成式 -> 警告
        if gen and (len(ds) < 3000 or n_feas < 500):
            print("=" * 70)
            print("WARNING:")
            print("Dataset too small for SAGPN / generator.")
            print("Recommended to train MLP / DeepSets / GCN first, or collect more data.")
            print("如果一定要跑, 建议 --model-size small。")
            print("=" * 70)

    best_metric = -np.inf
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_path = os.path.join(save_dir, f"{model_name}_best.pt")
    log_path = os.path.join(save_dir, f"{model_name}_train.log")
    history: List[Dict] = []
    since_improve = 0
    t0 = time.time()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    zero_grad_streak = 0
    if debug_grad and verbose:
        print(f"[debug-grad] trainable_params={n_trainable:,} "
              f"optimizer_params={n_opt_params:,} "
              f"(match={'YES' if n_trainable == n_opt_params else 'NO'})")

    with open(log_path, "w", encoding="utf-8") as logf:
        for epoch in range(1, epochs + 1):
            model.train()
            ep_logs: Dict[str, float] = {}
            nb = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                out = model(batch)
                res = compute_loss(out, batch, weights, gen)
                opt.zero_grad()
                res["loss"].backward()

                if debug_grad and nb < debug_batches:
                    with torch.no_grad():
                        logit = out["feas_logit"].detach()
                        sp = out["score_pred"].detach()
                        vpc = float(batch["node_mask"].sum(dim=1).float().mean().item())
                        gnorms = _group_grad_norms(model)
                        cur_lr = opt.param_groups[0]["lr"]
                        logit_std = float(logit.std().item())
                        gline = " ".join(f"{k}={v:.2e}" for k, v in sorted(gnorms.items()))
                        print(f"[debug-grad] ep{epoch} b{nb} "
                              f"node={tuple(batch['node_feat'].shape)} "
                              f"valid_parts~{vpc:.2f} "
                              f"logit(mean={logit.mean():.3f} std={logit_std:.3e} "
                              f"min={logit.min():.3f} max={logit.max():.3f}) "
                              f"score(mean={sp.mean():.3f} std={sp.std():.3e}) "
                              f"lr={cur_lr:.2e}")
                        print(f"[debug-grad] ep{epoch} b{nb} grad_norms: {gline}")
                        if logit_std < 1e-5:
                            print("[debug-grad] WARN: 同 batch logits std < 1e-5 "
                                  "(输出近常数, 疑似 SAGPN 式退化)")
                        main_keys = [k for k in gnorms
                                     if any(t in k for t in
                                            ("encoder", "trunk", "feas", "score",
                                             "phi", "in_proj", "mp", "convs", "layers"))]
                        main_zero = main_keys and all(gnorms[k] < 1e-12 for k in main_keys)
                        if main_zero:
                            zero_grad_streak += 1
                        else:
                            zero_grad_streak = 0
                        if zero_grad_streak >= 5:
                            raise RuntimeError(
                                "[debug-grad] 主要分支 grad norm 连续 5 次为 0, "
                                "训练链路断裂 (梯度未回传)。")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                for k, v in res["logs"].items():
                    ep_logs[k] = ep_logs.get(k, 0.0) + float(v)
                nb += 1
            for k in ep_logs:
                ep_logs[k] /= max(nb, 1)

            metrics = evaluate(model, val_loader, device, gen, topk=topk)
            sel = metrics.get(early_stop_metric, metrics["composite"])
            if sel is None or np.isnan(sel):
                sel = metrics["composite"]
            scheduler.step(sel)

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in ep_logs.items()}, **metrics}
            history.append(row)
            line = (f"epoch {epoch:03d} | loss={ep_logs.get('total', 0):.4f} "
                    f"cls={ep_logs.get('l_cls', 0):.4f} score={ep_logs.get('l_score', 0):.4f} "
                    f"rank={ep_logs.get('l_rank', 0):.4f} fail={ep_logs.get('l_fail', 0):.4f} "
                    f"xy={ep_logs.get('l_xy', 0):.4f} st={ep_logs.get('l_station', 0):.4f} "
                    f"| roc={metrics['roc_auc']:.3f} pr={metrics['pr_auc']:.3f} "
                    f"rec@{topk}={metrics['recall_at_k']:.3f} "
                    f"prec@{topk}={metrics['precision_at_k']:.3f} "
                    f"topkAvg={metrics['topk_avg_score_norm']:.3f} "
                    f"sp={metrics['score_spearman']:.3f} enr={metrics['enrichment']:.3f} "
                    f"| comp={metrics['composite']:.4f} [{early_stop_metric}={sel:.4f}]")
            logf.write(line + "\n")
            logf.flush()
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
                print(line)

            if sel > best_metric + min_delta:
                best_metric = sel
                best_epoch = epoch
                best_metrics = dict(metrics)
                since_improve = 0
                ckpt_config = {
                    "model_name": model_name,
                    "feature_version": feature_version,
                    "split_mode": split_mode,
                    "training_seed": seed,
                    "hidden_dim": int((model_kwargs or {}).get("hidden", 64)),
                    "dropout": float((model_kwargs or {}).get("dropout", 0.2)),
                    "rank_weight": float(weights.rank_weight),
                    "fail_weight": float(weights.fail_weight),
                    "use_focal": bool(weights.use_focal),
                    "focal_gamma": float(weights.focal_gamma),
                    "dataset_path": os.path.abspath(dataset_path),
                    "dataset_sample_count": len(samples),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "best_epoch": epoch,
                    "best_metric": float(sel),
                    "select_metric": early_stop_metric,
                    "git_commit": _git_commit(),
                    "model_kwargs": model_kwargs or {},
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "topk": topk,
                    "epochs": epochs,
                    "early_stop_patience": early_stop_patience,
                    "effective_pos_weight": float(weights.pos_weight),
                    "n_params": n_params,
                }
                torch.save({
                    **ckpt_config,
                    "state_dict": model.state_dict(),
                    "flat_dim": flat_dim,
                    "max_parts": max_parts,
                    "is_generator": gen,
                    "epoch": epoch,
                    "metric": float(sel),
                    "metrics": metrics,
                    "requested_pos_weight": requested_pos_weight,
                    "train_positive_count": int(n_pos),
                    "train_negative_count": int(n_neg),
                }, best_path)
            else:
                since_improve += 1

            if early_stop_patience > 0 and since_improve >= early_stop_patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch} "
                          f"(no improve of '{early_stop_metric}' for {early_stop_patience}); "
                          f"best={best_metric:.4f} @ epoch {best_epoch}")
                logf.write(f"[early stop] epoch {epoch}, best={best_metric:.4f} @ {best_epoch}\n")
                break

    wall_time_s = time.time() - t0
    sample_ids = [s.get("sample_id", i) for i, s in enumerate(samples)]
    config_payload = {
        "model_name": model_name,
        "feature_version": feature_version,
        "split_mode": split_mode,
        "training_seed": seed,
        "hidden_dim": int((model_kwargs or {}).get("hidden", 64)),
        "dropout": float((model_kwargs or {}).get("dropout", 0.2)),
        "rank_weight": float(weights.rank_weight),
        "fail_weight": float(weights.fail_weight),
        "use_focal": bool(weights.use_focal),
        "focal_gamma": float(weights.focal_gamma),
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_sample_count": len(samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "select_metric": early_stop_metric,
        "git_commit": _git_commit(),
        "model_kwargs": model_kwargs or {},
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "topk": topk,
        "epochs": epochs,
        "early_stop_patience": early_stop_patience,
        "effective_pos_weight": float(weights.pos_weight),
        "n_params": n_params,
        "checkpoint_path": best_path,
    }
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "select_metric": early_stop_metric,
        "wall_time_s": wall_time_s,
        "n_params": n_params,
        **{k: float(v) if v is not None and not (isinstance(v, float) and np.isnan(v))
           else None for k, v in best_metrics.items()},
    }
    _save_training_artifacts(
        save_dir, model_name,
        config=config_payload,
        metrics=metrics_payload,
        history=history,
        train_indices=train_idx,
        val_indices=val_idx,
        sample_ids=sample_ids,
    )
    summary = {
        "model_name": model_name,
        "select_metric": early_stop_metric,
        "best_metric": float(best_metric),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_path": best_path,
        "split_mode": split_mode,
        "feature_version": feature_version,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "train_feasible_rate": (n_pos / max(len(train_idx), 1)),
        "train_positive_count": int(n_pos),
        "train_negative_count": int(n_neg),
        "requested_pos_weight": requested_pos_weight,
        "effective_pos_weight": float(weights.pos_weight),
        "n_params": n_params,
        "wall_time_s": wall_time_s,
        "config": config_payload,
        "final_metrics": history[-1] if history else {},
        "history": history,
    }
    with open(os.path.join(save_dir, f"{model_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[train] done. best {early_stop_metric}={best_metric:.4f} "
              f"@ epoch {best_epoch} -> {best_path}")
    return summary
