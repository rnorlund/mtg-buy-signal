"""Head-to-head: our model vs the heuristics real MTG-finance tools / simple specs use.

All methods are scored on the SAME held-out test rows, the SAME (g, H) targets, and the
SAME metrics — precision@K, PR-AUC, and rolling realised sustained-peak return of the
top-K buys. Baselines mirror what the popular tools actually do:

  momentum_30 / momentum_90  — "trending up" continuation (EchoMTG/MTGStocks movers)
  most_expensive             — buy the chase cards
  edhrec_popularity          — buy what's most-played in Commander
  buy_the_dip                — biggest drawdown from 1y high (mean-reversion spec)
  random                     — random liquid card

This is built to be fair: the baselines use real signals (the same features our model sees),
and nothing about the test split or labels differs between methods.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from mtgspike.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, data_end
from mtgspike.model.dataset import feature_columns, gated_split, to_long
from mtgspike.model.train import _precision_at_k
from mtgspike.predict import _load
from sklearn.metrics import average_precision_score

log = logging.getLogger(__name__)


def _model_scores(test: pl.DataFrame, feat_names, calibrators, boosters) -> dict[float, np.ndarray]:
    X = test.select([c if c in test.columns else pl.lit(None).alias(c) for c in feat_names]).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    dm = xgb.DMatrix(X, feature_names=feat_names)
    cal = {g: calibrators[g].predict(boosters[g].predict(dm)) for g in boosters}
    rungs = sorted(cal)
    for i in range(1, len(rungs)):
        cal[rungs[i]] = np.minimum(cal[rungs[i]], cal[rungs[i - 1]])
    return cal


def _baseline_scores(test: pl.DataFrame) -> dict[str, np.ndarray]:
    def col(name, fill):
        return test[name].fill_null(fill).to_numpy() if name in test.columns else np.full(test.height, fill)
    rng = np.random.default_rng(0)
    return {
        "momentum_30": col("f_ret_30d", -1.0),
        "momentum_90": col("f_ret_90d", -1.0),
        "most_expensive": col("f_log_price", -10.0),
        "edhrec_popularity": -col("edhrec_rank", 10_000_000).astype(float),  # lower rank = better
        "buy_the_dip": -col("f_drawdown_365", 0.0),                          # deeper dip = better
        "random": rng.random(test.height),
    }


def benchmark(
    *, features_path: Path = PROCESSED_DIR / "features.parquet", k: int = 20,
    months: int = 24, gains=(1.5, 2.0, 3.0), test_start: date = date(2023, 1, 1),
    models_dir: Path = MODELS_DIR, reserved_only: bool = False,
) -> dict:
    long = to_long(pl.read_parquet(features_path))
    feats = feature_columns(long)
    split = gated_split(long, val_start=date(2020, 6, 1), test_start=test_start, data_end=data_end())
    test = split.test.filter(pl.col("horizon_months") == months)
    tag = "RESERVED-LIST ONLY" if reserved_only else "ALL CARDS"
    if reserved_only:
        test = test.filter(pl.col("is_reserved_list") == 1)
    log.info("Benchmark [%s] on %d held-out rows (H=%dmo, test>=%s)", tag, test.height, months, test_start)
    if test.height < k * 3:
        log.warning("  too few rows for a stable benchmark (%d)", test.height)

    feat_names, calibrators, boosters = _load(models_dir)
    model_cal = _model_scores(test, feat_names, calibrators, boosters)
    baselines = _baseline_scores(test)

    def _pct_rank(x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
        order = np.argsort(np.argsort(x))
        return order / max(len(order) - 1, 1)

    # ensemble: per-gain blend of model and momentum (weights tuned on validation; else equal)
    mom = baselines["momentum_30"]
    bw = {}
    bw_path = models_dir / "blend_weights.json"
    if bw_path.exists():
        bw = {float(kk): vv for kk, vv in json.loads(bw_path.read_text()).items()}
    ensemble = {}
    for g in model_cal:
        w = bw[min(bw, key=lambda gg: abs(gg - g))] if bw else 0.5
        ensemble[g] = w * _pct_rank(model_cal[g]) + (1 - w) * _pct_rank(mom)

    def ranked_return(score: np.ndarray) -> float:
        """Rolling: each snapshot, buy top-K by score, mean sustained return; average."""
        t = test.with_columns(pl.Series("s", score))
        per = []
        for _snap, g in t.group_by("snapshot_date"):
            if g.height < k:
                continue
            top = g.sort("s", descending=True).head(k)
            per.append(float((top["target_mult"] - 1).mean()))
        return float(np.mean(per)) if per else float("nan")

    results = {"config": {"k": k, "months": months, "test_start": str(test_start), "n_rows": test.height},
               "by_g": {}}
    for g in gains:
        y = (test["target_mult"] >= g).cast(pl.Int8).to_numpy()
        row = {"base_rate": float(y.mean())}
        methods = {"OURS": model_cal[g], "OURS+mom": ensemble[g], **baselines}
        for name, score in methods.items():
            valid = np.isfinite(score)
            ys, ss = y[valid], score[valid]
            row[name] = {
                "pr_auc": float(average_precision_score(ys, ss)) if ys.sum() else None,
                "prec_at_k": _precision_at_k(ys, ss, k),
                "ranked_return": ranked_return(score),
            }
        results["by_g"][f"g{g}"] = row

    results["universe"] = tag
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fname = "benchmark_rl.json" if reserved_only else "benchmark.json"
    (OUTPUTS_DIR / fname).write_text(json.dumps(results, indent=2))

    # pretty print
    for gk, row in results["by_g"].items():
        base = row["base_rate"]
        log.info("")
        log.info("=== [%s] target %s within %dmo  (base rate %.1f%%) ===", tag, gk, months, 100 * base)
        log.info("  %-18s  %6s  %8s  %12s", "method", "P@%d" % k, "PR-AUC", "topK return")
        ordered = sorted([m for m in row if m != "base_rate"],
                         key=lambda m: -(row[m]["prec_at_k"] if row[m]["prec_at_k"] == row[m]["prec_at_k"] else -1))
        for m in ordered:
            d = row[m]
            star = "  <-- OURS" if m == "OURS" else ""
            log.info("  %-18s  %5.0f%%  %8s  %+11.1f%%%s", m, 100 * d["prec_at_k"],
                     f"{d['pr_auc']:.3f}" if d["pr_auc"] else "NA", 100 * d["ranked_return"], star)
    log.info("\nWrote %s", OUTPUTS_DIR / "benchmark.json")
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    benchmark()


if __name__ == "__main__":
    main()
