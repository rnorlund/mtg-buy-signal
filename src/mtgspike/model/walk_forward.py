"""Walk-forward validation: retrain the model on rolling 1-year folds and report stability.

For each fold we re-fit XGBoost from scratch on data <= train_end, calibrate on a 1-year
validation block, then evaluate on a 1-year held-out test block. Reports PR-AUC, ROC-AUC,
and precision@K per gain head per fold, at H=12 months (most consistently resolvable
across folds). The metric we trust most across folds is PR-AUC — top-20 P@K is noisier
because each fold's test block has its own number of opportunities.

Output: outputs/walk_forward.json
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from mtgspike.config import GAIN_LADDER, OUTPUTS_DIR, PROCESSED_DIR
from mtgspike.model.dataset import feature_columns, gated_split, to_long
from mtgspike.model.train import _matrix, _precision_at_k

log = logging.getLogger(__name__)

# Each fold: train ends before val_start; val is [val_start, test_start); test is [test_start, test_end)
# Five rolling folds covering test years 2020-2024. Labels at H=12 mo resolve within each fold.
FOLDS = [
    {"name": "test=2020", "val_start": date(2019, 1, 1), "test_start": date(2020, 1, 1), "test_end": date(2021, 1, 1)},
    {"name": "test=2021", "val_start": date(2020, 1, 1), "test_start": date(2021, 1, 1), "test_end": date(2022, 1, 1)},
    {"name": "test=2022", "val_start": date(2021, 1, 1), "test_start": date(2022, 1, 1), "test_end": date(2023, 1, 1)},
    {"name": "test=2023", "val_start": date(2022, 1, 1), "test_start": date(2023, 1, 1), "test_end": date(2024, 1, 1)},
    {"name": "test=2024", "val_start": date(2023, 1, 1), "test_start": date(2024, 1, 1), "test_end": date(2025, 1, 1)},
]
EVAL_HORIZON = 12  # report at H=12 mo for cross-fold consistency


def _train_one_head(Xtr, ytr, Xva, yva, feats, device="cpu"):
    spw = (ytr == 0).sum() / max(ytr.sum(), 1)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feats)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "tree_method": "hist", "device": device,
              "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85,
              "colsample_bytree": 0.85, "min_child_weight": 5,
              "scale_pos_weight": spw, "verbosity": 0}
    bst = xgb.train(params, dtr, num_boost_round=800,
                    evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(bst.predict(dva), yva)
    return bst, cal


def run_fold(long: pl.DataFrame, feats: list[str], fold: dict, eval_horizon: int = 12) -> dict:
    val_start, test_start, test_end = fold["val_start"], fold["test_start"], fold["test_end"]
    eval_data_end = test_end + timedelta(days=int(eval_horizon * 30.44) + 5)
    split = gated_split(long, val_start=val_start, test_start=test_start, data_end=eval_data_end)
    # Restrict test snapshots to within this fold's window
    test = split.test.filter(pl.col("snapshot_date") < pl.lit(test_end))
    train, val = split.train, split.val
    log.info("[%s] train=%d val=%d test=%d", fold["name"], train.height, val.height, test.height)
    if not train.height or not val.height or not test.height:
        return {"note": "empty split"}

    Xtr, Xva, Xte = _matrix(train, feats), _matrix(val, feats), _matrix(test, feats)
    h_mask = (test["horizon_months"] == eval_horizon).to_numpy()
    out: dict = {"n_test_h": int(h_mask.sum())}
    if h_mask.sum() < 50:
        log.info("  H=%dmo: too few test rows (%d)", eval_horizon, h_mask.sum())
        return out

    for g in GAIN_LADDER:
        ytr = (train["target_mult"] >= g).cast(pl.Int8).to_numpy()
        yva = (val["target_mult"] >= g).cast(pl.Int8).to_numpy()
        yte = (test["target_mult"] >= g).cast(pl.Int8).to_numpy()
        if ytr.sum() == 0 or (ytr == 0).sum() == 0 or yva.sum() == 0:
            continue
        bst, cal = _train_one_head(Xtr, ytr, Xva, yva, feats)
        p_te = cal.predict(bst.predict(xgb.DMatrix(Xte, feature_names=feats)))
        # Restrict to the consistent eval horizon
        yh, ph = yte[h_mask], p_te[h_mask]
        if yh.sum() == 0:
            continue
        m = {
            "base_rate": float(yh.mean()),
            "pr_auc": float(average_precision_score(yh, ph)),
            "roc_auc": float(roc_auc_score(yh, ph)) if 0 < yh.sum() < len(yh) else None,
            "prec_at_20": _precision_at_k(yh, ph, 20),
            "prec_at_50": _precision_at_k(yh, ph, 50),
        }
        out[f"g{g}"] = m
        log.info("  g=%.1f  base=%.1f%%  PR-AUC=%.3f  P@20=%.0f%%  P@50=%.0f%%",
                 g, 100 * m["base_rate"], m["pr_auc"], 100 * m["prec_at_20"], 100 * m["prec_at_50"])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feats_df = pl.read_parquet(PROCESSED_DIR / "features.parquet")
    long = to_long(feats_df)
    feats = feature_columns(long)
    log.info("Long rows: %d  |  features: %d", long.height, len(feats))

    results = {"eval_horizon": EVAL_HORIZON, "folds": {}}
    for fold in FOLDS:
        log.info("=" * 70)
        results["folds"][fold["name"]] = run_fold(long, feats, fold, EVAL_HORIZON)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "walk_forward.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", OUTPUTS_DIR / "walk_forward.json")


if __name__ == "__main__":
    main()
