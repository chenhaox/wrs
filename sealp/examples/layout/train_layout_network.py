"""统一训练脚本 (CLI)。

用法示例:
    python -m sealp.examples.layout.train_layout_network \
        --dataset sealp/examples/layout/_output/layout_dataset_v2.jsonl \
        --model mlp --epochs 200 --batch-size 32 --lr 1e-3 \
        --split-mode stratified --early-stop-patience 30 \
        --save-dir checkpoints/layout_models_debug

支持模型: mlp / deepsets / set_transformer / gcn / gat / transformer /
          pointnet / cvae / diffusion / sagpn / all

训练目标:
    L = L_cls + alpha*L_score_feasible + beta*L_xy_elite + gamma*L_station_elite
    - L_cls          : 所有样本;
    - L_score        : 只 feasible;
    - L_xy / L_station: 只 elite feasible (score >= 分位阈值)。

验证指标: ROC-AUC / PR-AUC / recall@K / precision@K / top-K avg true score /
          score MAE·RMSE·Spearman / enrichment。
best checkpoint 用 composite metric。
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning.losses import LossWeights
from layout_learning.models import MODEL_NAMES, size_kwargs
from layout_learning.train import train_model


_METRIC_CHOICES = ["composite", "pr_auc", "roc_auc", "auc", "recall_at_k",
                   "precision_at_k", "topk_hit", "topk_avg_score_norm",
                   "score_spearman", "feas_acc"]


def _parse_score_threshold(spec: str):
    """返回 (elite_quantile or None, abs_threshold)。

    - "quantile:0.70" -> (0.70, 0.0)  按训练集 feasible 分数分位数
    - "0.4"           -> (None, 0.4)  绝对阈值
    """
    spec = str(spec).strip()
    if spec.lower().startswith("quantile:"):
        return float(spec.split(":", 1)[1]), 0.0
    return None, float(spec)


def _parse_pos_weight(spec: str):
    """解析 --pos-weight: 'auto' 或正浮点数。"""
    text = str(spec).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pos-weight 必须是 'auto' 或正浮点数，例如 2.5"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("--pos-weight 必须大于 0")
    return value


def _parse_args():
    p = argparse.ArgumentParser(description="Layout learning trainer")
    p.add_argument("--dataset", required=True, help="layout_dataset.jsonl 路径")
    p.add_argument("--model", default="mlp",
                   help=f"模型名 (可选: {MODEL_NAMES + ['all']})")
    p.add_argument("--model-size", default=None, choices=[None, "small", "base"],
                   help="生成式模型尺寸 (sagpn/cvae/diffusion); scorer 忽略")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (默认自动)")
    p.add_argument("--save-dir", default=os.path.join("checkpoints", "layout_models"))
    p.add_argument("--topk", type=int, default=10, help="recall@K / precision@K 的 K")
    # 特征版本: v1=旧固定归一化(兼容旧ckpt/MLP baseline); v2=自适应归一化(面向迁移)
    p.add_argument("--feature-version", default="v1", choices=["v1", "v2"])
    # split & early stop
    p.add_argument("--split-mode", default="stratified",
                   choices=["stratified", "random", "seed_holdout", "region_holdout"])
    p.add_argument("--region-holdout-mode", default="auto",
                   choices=["auto", "rc", "xy_grid"],
                   help="region_holdout 分桶方式: rc(3x3网格)/xy_grid(连续站位网格)/auto")
    p.add_argument("--region-xy-grid", type=int, default=3,
                   help="xy_grid 分桶的网格划分数")
    p.add_argument("--early-stop-patience", type=int, default=30,
                   help="0 表示关闭 early stopping")
    p.add_argument("--early-stop-metric", default="composite", choices=_METRIC_CHOICES)
    p.add_argument("--min-delta", type=float, default=1e-4)
    # loss 权重
    p.add_argument("--alpha", type=float, default=1.0, help="score 回归权重")
    p.add_argument("--beta", type=float, default=1.0, help="xy proposal 权重")
    p.add_argument("--gamma", type=float, default=0.5, help="station/region 权重")
    p.add_argument("--kl", type=float, default=0.01, help="CVAE KL 权重")
    p.add_argument(
        "--pos-weight", type=_parse_pos_weight, default="auto",
        help="可行样本正类加权: 'auto'=按当前训练 split 的 neg/pos 自动计算，或指定正浮点数")
    p.add_argument("--score-threshold", default="quantile:0.70",
                   help="elite 阈值: 'quantile:0.70' 或绝对值如 '0.4'")
    # ---- seqrel 专用 loss / 调试 (其它模型默认关闭, 不受影响) ----
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="pair-ranking loss 权重 (seqrel 建议 0.5); 0=关闭")
    p.add_argument("--fail-weight", type=float, default=0.0,
                   help="fail 辅助分类权重 (seqrel 建议 0.2); 0=关闭")
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-min-score-gap", type=float, default=0.05)
    p.add_argument("--rank-pairs-per-batch", type=int, default=256)
    p.add_argument("--use-focal", action="store_true",
                   help="L_cls 使用 focal loss (默认 BCEWithLogits)")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--debug-grad", action="store_true",
                   help="每 epoch 前几个 batch 打印 logits/梯度诊断并做自动检查")
    p.add_argument("--debug-batches", type=int, default=3)
    p.add_argument("--limit-samples", type=int, default=0,
                   help=">0 时分层截断数据集 (overfit smoke test 用)")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="随机置换标签的 sanity check (指标应崩到随机水平)")
    return p.parse_args()


def main():
    args = _parse_args()
    elite_quantile, abs_thr = _parse_score_threshold(args.score_threshold)
    # auto 模式的真正权重会在 train_model 完成 train/val split 后计算。
    # 这里使用 1.0 作为临时占位，避免把字符串传入 loss dataclass。
    initial_pos_weight = 1.0 if args.pos_weight == "auto" else float(args.pos_weight)
    weights = LossWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                          kl=args.kl, pos_weight=initial_pos_weight,
                          score_threshold=abs_thr,
                          rank_weight=args.rank_weight,
                          fail_weight=args.fail_weight,
                          rank_margin=args.rank_margin,
                          rank_min_score_gap=args.rank_min_score_gap,
                          rank_pairs_per_batch=args.rank_pairs_per_batch,
                          use_focal=args.use_focal,
                          focal_gamma=args.focal_gamma)
    models = MODEL_NAMES if args.model == "all" else [args.model]
    for name in models:
        mkwargs = size_kwargs(name, args.model_size)   # 解析后的显式构造参数
        print("=" * 70)
        print(f"[train] training model = {name}  size={args.model_size or 'default'} "
              f"kwargs={mkwargs}")
        print("=" * 70)
        train_model(
            dataset_path=args.dataset,
            model_name=name,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            loss_weights=weights,
            pos_weight=args.pos_weight,
            model_kwargs=mkwargs,
            topk=args.topk,
            split_mode=args.split_mode,
            region_holdout_mode=args.region_holdout_mode,
            region_xy_grid=args.region_xy_grid,
            feature_version=args.feature_version,
            elite_quantile=elite_quantile,
            early_stop_patience=args.early_stop_patience,
            early_stop_metric=args.early_stop_metric,
            min_delta=args.min_delta,
            debug_grad=args.debug_grad,
            debug_batches=args.debug_batches,
            limit_samples=args.limit_samples,
            shuffle_labels=args.shuffle_labels,
            verbose=True,
        )


if __name__ == "__main__":
    main()
