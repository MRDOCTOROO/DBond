#!/usr/bin/env python3
"""传统机器学习基线（RF / XGBoost / HistGB / LogReg）：5 折训练 + DBond-GT 同口径评测。

动机：MiPD513 语料只有 513 条唯一序列（47.7 万谱图行，93% 标签方差在序列内部），
小样本低序列多样性场景下传统树模型可能不逊于深度模型——本脚本在同一特征、
同一折、同一指标下直接检验该假设。

协议对齐（ludbond/_5fold_common.py + DBond-GT train_5fold.py）：
  - 5 个 sequence-level fold：1222/2252/3514/6072/9075（train/test 序列不相交）
  - seed = base_seed(42) + fold_index
  - val：train 行级随机 20%（sklearn train_test_split，与 ludbond 的
    torch random_split 口径一致、RNG 实现不同）
  - 测试：固定阈值 0.5，f1_micro(≡lab_f1_mi) + accuracy/hamming + q_* 全套
    （graph_transform/training/metrics.BinaryBondMetrics，与 GT/film 同一实现）
  - 汇总：mean ± std(ddof=0)（对齐 aggregate_5fold）

特征：sklearn_baseline/features.py，严格 parity 口径（15 理论维 + charge +
pep_mass + nce，对齐 film 臂 pre_synthesis_fold1222_theory_film.yaml）。

输出（对齐 ludbond result/cv 结构，自包含在 sklearn_baseline/result/cv 下）：
  {out_root}/{model}/{ts}/fold_{id}/pred/test.pred.csv      dbond_s 变长键级长表，
      可直接喂 graph_transform/scripts/q_metrics_from_pred_csv.py 交叉校验
  {out_root}/{model}/{ts}/fold_{id}/metric/test_metric.csv  metric,value 两列
  {out_root}/{model}/{ts}/fold_{id}/metric/feature_importance.csv（模型支持时）
  {out_root}/{model}/{ts}/5fold_metrics.csv + 5fold_summary.csv

用法（仓库根目录）：
  python sklearn_baseline/run_tree_baselines.py --models rf,histgb,xgb,logreg
  python sklearn_baseline/run_tree_baselines.py --models xgb --xgb_device cuda   # pod
  python sklearn_baseline/run_tree_baselines.py --models rf --folds 1222 --limit_rows 20000  # 冒烟
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # 仓库根目录
for p in (str(HERE), str(ROOT), str(ROOT / "graph_transform")):
    if p not in sys.path:
        sys.path.insert(0, p)

# torch 依赖仅用于 theory_features（纯序列计算，CPU 即可）与 metrics（同 q_metrics_from_pred_csv 的 import 方式）
from training.metrics import BinaryBondMetrics, batch_group_keys  # noqa: E402

from features import FEATURE_NAMES, BondFrame, build_bond_frame, parse_multi  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FOLD_IDS = ["1222", "2252", "3514", "6072", "9075"]  # 与 _5fold_common.FOLD_IDS 一致
TRAIN_SUFFIX = ".train.fbr.shuffle.multi.csv"
TEST_SUFFIX = ".test.fbr.multi.csv"
DEFAULT_BASE_SEED = 42

# 与 q_metrics_from_pred_csv.py 落盘键一致（test_metric.csv 里额外保留 compute() 全量键）
Q_EVAL_KEYS = [
    "f1_micro", "accuracy", "hamming_loss",
    "q_brier", "q_mae", "q_spearman",
    "q_spearman_pep", "q_spearman_pep_cond", "q_spearman_pep_seq",
    "q_top10_enrichment", "q_top10_enrichment_cond", "q_top10_enrichment_seq",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *a: (datetime.datetime.now(datetime.timezone.utc)
                                         + datetime.timedelta(hours=8)).timetuple()
log = logging.getLogger("sklearn_baseline")


def beijing_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)


# ---------------------------------------------------------------------------
# q 软标签（expected-behavior 真值）内联计算
# ---------------------------------------------------------------------------

def compute_test_q(frame: pd.DataFrame) -> list[np.ndarray]:
    """测试折内按 (seq,charge,nce) 组内逐键均值求 q。

    逐行逐键复刻 graph_transform/scripts/precompute_soft_labels.py::
    compute_group_soft（groupby sort=False + parse + mean，%.4f 取整），
    不直接 import 它是因为其模块级 import 链会拉起 train_graph_model
    （torch_geometric）。pod 端可用 precompute_soft_labels.py 生成
    dataset/5fold_soft 后跑 q_metrics_from_pred_csv.py 做全链路对拍。
    """
    q_per_row: list[np.ndarray] = [None] * len(frame)  # type: ignore
    keys = ["seq", "charge", "nce"]
    for group_key, idxs in frame.groupby(keys, sort=False).indices.items():
        idxs = np.asarray(idxs)
        mats = np.stack([parse_multi(v) for v in frame["true_multi"].iloc[idxs]])
        q = np.round(mats.mean(axis=0), 4)  # 与 compute_group_soft 的 %.4f 落盘一致
        for i in idxs:
            q_per_row[int(i)] = q
    if any(v is None for v in q_per_row):
        raise ValueError("存在未覆盖 q 的测试行，分组逻辑异常")
    return q_per_row


# ---------------------------------------------------------------------------
# 评测（镜像 q_metrics_from_pred_csv.py 的喂入方式）
# ---------------------------------------------------------------------------

def evaluate_test_probs(bf: BondFrame, probs: np.ndarray, q_per_row: list[np.ndarray]) -> dict:
    """键级概率 → BinaryBondMetrics 全量指标。

    compute() 固定按 logits 过 sigmoid（metrics.py:446），因此概率先做 logit
    逆变换再喂入（与 q_metrics_from_pred_csv.py / ensemble_inference.py 同做法）。
    """
    frame = bf.frame
    n_bonds = bf.n_bonds_per_row
    R, L = len(frame), int(n_bonds.max())
    P = np.zeros((R, L), dtype=np.float32)
    Y = np.zeros((R, L), dtype=np.int32)
    Q = np.zeros((R, L), dtype=np.float32)
    MASK = np.zeros((R, L), dtype=bool)
    starts = np.concatenate([[0], np.cumsum(n_bonds)])
    seqs = [str(s) for s in frame["seq"]]
    for i in range(R):
        b = int(n_bonds[i])
        if b == 0:
            continue
        s, e = int(starts[i]), int(starts[i + 1])
        P[i, :b] = probs[s:e]
        Y[i, :b] = bf.y[s:e]
        Q[i, :b] = q_per_row[i]
        MASK[i, :b] = True

    eps = 1e-7
    logits = np.log(np.clip(P, eps, 1.0 - eps) / (1.0 - np.clip(P, eps, 1.0 - eps))).astype(np.float32)

    metrics = BinaryBondMetrics({"threshold": 0.5, "threshold_strategy": "fixed"})
    metrics.update(
        logits, Y,
        label_mask=MASK,
        soft_targets=Q,
        sequences=seqs,
        group_keys=batch_group_keys({
            "sequences": seqs,
            "charges": frame["charge"].to_numpy(),
            "nces": frame["nce"].to_numpy(),
        }),
    )
    return metrics.compute()


def write_pred_csv(bf: BondFrame, probs: np.ndarray, out_path: Path) -> None:
    """dbond_s 变长键级长表（q_metrics_from_pred_csv.py 可直接消费）。"""
    df = pd.DataFrame({
        "evaluation_id": bf.row_of_bond.astype(np.int64),
        "threshold": 0.5,
        "bond_index": bf.bond_index.astype(np.int64),
        "true": bf.y.astype(np.int64),
        "pred": (probs > 0.5).astype(np.int64),
        "pred_prob": probs.astype(np.float64),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def make_model(name: str, seed: int, args) -> object:
    if name == "rf":
        # max_samples: 每棵树 bootstrap 的行数上限（float<=1 为比例）。10M 键行
        # 全量 bootstrap 在 ~16GB 本机会 OOM；pod 大内存可保持默认 None（全量）
        max_samples = args.rf_max_samples
        if max_samples is not None and max_samples > 1:
            max_samples = int(max_samples)
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators, n_jobs=args.n_jobs,
            max_samples=max_samples, random_state=seed)
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.1, random_state=seed)
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed))
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost 未安装（pyproject 已声明，pod 端 uv pip install xgboost）")
        # 迭代数由 val logloss early stopping 决定（自定义 F1 eval 的入参形式
        # 在不同 xgboost 版本间有 margin/概率歧义，不冒险）；val F1 照样记录
        return XGBClassifier(
            n_estimators=3000, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", device=args.xgb_device,
            eval_metric="logloss", early_stopping_rounds=50,
            random_state=seed, n_jobs=args.n_jobs, verbosity=1)
    raise ValueError(f"未知模型: {name}")


def fit_model(model, X_fit, y_fit, X_val, y_val, name: str) -> dict:
    """拟合 + 返回元信息（xgb 用 val early stopping，其余模型 val 仅记录用）。"""
    t0 = time.time()
    extra = {}
    if name == "xgb":
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=200)
        extra["best_iteration"] = int(getattr(model, "best_iteration", -1))
    else:
        model.fit(X_fit, y_fit)
    extra["fit_seconds"] = round(time.time() - t0, 1)
    return extra


def predict_proba_safe(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# 5 折主流程
# ---------------------------------------------------------------------------

def aggregate_5fold_local(per_fold_metrics: list, out_dir: Path) -> tuple:
    """mean±std(ddof=0)+min/max/num_folds，复刻 ludbond/_5fold_common.aggregate_5fold
    （fold_id 等元信息列不参与聚合）。"""
    metrics_df = pd.DataFrame(per_fold_metrics)
    metrics_df.to_csv(out_dir / "5fold_metrics.csv", index=False)
    rows = []
    for metric in [c for c in metrics_df.columns if c not in ("fold_id",)]:
        series = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
        if len(series) == 0:
            continue
        rows.append({
            "metric": metric,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "num_folds": int(series.shape[0]),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "5fold_summary.csv", index=False)
    return metrics_df, summary_df


def run_one_fold(model_name: str, fold_id: str, fold_index: int, args,
                 theory_cache: dict, run_root: Path) -> dict:
    seed = args.base_seed + fold_index
    fold_dir = Path(args.fold_dir)
    train_path = fold_dir / f"{fold_id}{TRAIN_SUFFIX}"
    test_path = fold_dir / f"{fold_id}{TEST_SUFFIX}"
    metric_dir = run_root / "metric"  # run_root 已是 fold_{id} 目录
    pred_dir = run_root / "pred"

    metric_csv = metric_dir / "test_metric.csv"
    if metric_csv.exists() and not args.force_new:
        log.info(f"[{model_name}] fold {fold_id} 已有结果, 跳过: {metric_csv}")
        ref = pd.read_csv(metric_csv)
        row = dict(zip(ref["metric"], ref["value"]))
        row["fold_id"] = fold_id
        return row

    log.info(f"[{model_name}] fold {fold_id}: 构建键级特征 (train={train_path.name}, test={test_path.name})")
    train_bf = build_bond_frame(train_path, theory_cache=theory_cache, limit_rows=args.limit_rows)
    test_bf = build_bond_frame(test_path, theory_cache=theory_cache, limit_rows=args.limit_rows)
    log.info(f"[{model_name}] fold {fold_id}: train {len(train_bf.y):,} 键行 / "
             f"test {len(test_bf.y):,} 键行, 特征 {train_bf.X.shape[1]} 维")

    # val：train 行级随机 20%（协议对齐 ludbond random_split；RNG 实现不同但口径一致）
    fit_idx, val_idx = train_test_split(
        np.arange(len(train_bf.y)), test_size=args.val_fraction, random_state=seed)
    X_fit, y_fit = train_bf.X[fit_idx], train_bf.y[fit_idx]
    X_val, y_val = train_bf.X[val_idx], train_bf.y[val_idx]

    model = make_model(model_name, seed, args)
    extra = fit_model(model, X_fit, y_fit, X_val, y_val, model_name)

    val_prob = predict_proba_safe(model, X_val)
    val_f1 = float(f1_score(y_val, (val_prob > 0.5).astype(int), zero_division=0))
    log.info(f"[{model_name}] fold {fold_id}: val_f1={val_f1:.4f} {extra}")

    test_prob = predict_proba_safe(model, test_bf.X)
    q_per_row = compute_test_q(test_bf.frame)
    out = evaluate_test_probs(test_bf, test_prob, q_per_row)

    # 落盘：pred 长表 + metric 两列
    if args.save_pred:
        write_pred_csv(test_bf, test_prob, pred_dir / "test.pred.csv")
        log.info(f"[{model_name}] fold {fold_id}: pred -> {pred_dir / 'test.pred.csv'}")
    metric_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"metric": list(out.keys()), "value": [float(out[k]) for k in out]}
                 ).to_csv(metric_csv, index=False)

    # 特征重要性（模型支持时）
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"feature": FEATURE_NAMES,
                            "importance": est.feature_importances_})
        imp.to_csv(metric_dir / "feature_importance.csv", index=False)

    row = {k: float(out[k]) for k in out}
    row.update({
        "fold_id": fold_id, "seed": seed, "best_val_f1": val_f1,
        "n_train_bond_rows": int(len(train_bf.y)), "n_test_bond_rows": int(len(test_bf.y)),
        **extra,
    })
    log.info(f"[{model_name}] fold {fold_id}: test f1_micro={row.get('f1_micro', float('nan')):.4f} "
             f"q_spearman_pep_seq={row.get('q_spearman_pep_seq', float('nan')):.4f}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="传统 ML 基线 5 折（RF/XGBoost/HistGB/LogReg）")
    ap.add_argument("--models", default="rf,histgb,xgb",
                    help="逗号分隔: rf,histgb,xgb,logreg")
    ap.add_argument("--folds", default="all", help="all 或逗号分隔 fold id 子集")
    ap.add_argument("--fold_dir", default=str(ROOT / "dataset" / "5fold"))
    ap.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--xgb_device", default="cpu", help="cpu|cuda（pod 5090 可用 cuda）")
    ap.add_argument("--rf_n_estimators", type=int, default=300)
    ap.add_argument("--rf_max_samples", type=float, default=None,
                    help="RF 每树 bootstrap 行数上限（float<=1 比例 / >1 绝对行数 / None 全量）。"
                         "内存吃紧时降载用（本机 16GB 建议 0.3）")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_root", default=str(HERE / "result" / "cv"))
    ap.add_argument("--force_new", action="store_true", help="已有 test_metric.csv 也强制重跑")
    ap.add_argument("--save_pred", action=argparse.BooleanOptionalAction, default=True,
                    help="保存 test.pred.csv 长表（q_metrics_from_pred_csv.py 交叉校验用；"
                         "约 95MB/折/模型，--no-save-pred 关闭）")
    ap.add_argument("--limit_rows", type=int, default=None,
                    help="调试用：每折 CSV 只读前 N 谱图行")
    args = ap.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m == "xgb" and not HAS_XGB:
            raise SystemExit("xgboost 未安装：本机跳过 xgb 或先安装（pod: uv pip install xgboost）")
    folds = FOLD_IDS if args.folds == "all" else [f.strip() for f in args.folds.split(",")]

    timestamp = beijing_now().strftime("%Y%m%d_%H%M%S")
    for model_name in model_names:
        cv_root = Path(args.out_root) / model_name / timestamp
        cv_root.mkdir(parents=True, exist_ok=True)
        log.info(f"[{model_name}] 5fold cv_root: {cv_root}")
        theory_cache: dict = {}
        per_fold = []
        for fold_index, fold_id in enumerate(folds):
            per_fold.append(run_one_fold(model_name, fold_id, fold_index, args,
                                         theory_cache, cv_root / f"fold_{fold_id}"))
        _, summary_df = aggregate_5fold_local(per_fold, cv_root)
        log.info(f"[{model_name}] 5fold summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
