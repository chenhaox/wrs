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
    p.add_argument("--pos-weight", type=float, default=3.0, help="可行样本正类加权")
    p.add_argument("--score-threshold", default="quantile:0.70",
                   help="elite 阈值: 'quantile:0.70' 或绝对值如 '0.4'")
    return p.parse_args()


def main():
    args = _parse_args()
    elite_quantile, abs_thr = _parse_score_threshold(args.score_threshold)
    weights = LossWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                          kl=args.kl, pos_weight=args.pos_weight,
                          score_threshold=abs_thr)
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
            verbose=True,
        )


if __name__ == "__main__":
    main()
