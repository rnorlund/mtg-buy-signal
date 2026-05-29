"""Shared dataset utilities: long-form stacking and leakage-safe time gating.

The feature matrix has one row per (oracle_id, snapshot_date) with per-horizon label
columns. We reshape to LONG form — one row per (oracle_id, snapshot_date, H) with H as
a feature — so a single model spans all holding windows. Binary targets for a gain
multiple g are then `sustained_mult_H >= g`.

Time gating (no leakage): a row's label peeks H months forward, so a row may only be in
a split if its label fully resolves *before the next split begins*. This auto-embargoes
each row by its own H.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import polars as pl

from spikepred.config import GAIN_LADDER, HORIZONS_MONTHS

log = logging.getLogger(__name__)

_DAYS_PER_MONTH = 30.44

# columns that are bookkeeping or labels — never features
_BOOKKEEPING = ("oracle_id", "name", "snapshot_date", "entry_price", "first_obs", "current_rarity")
_LABEL_PREFIXES = ("sustain_mult_", "peak_mult_", "hold_mult_", "resolved_")


def feature_columns(df: pl.DataFrame) -> list[str]:
    """All usable feature columns (numeric, non-label, non-bookkeeping) + horizon_months."""
    feats = []
    for c, dt in df.schema.items():
        if c in _BOOKKEEPING or c == "horizon_months":
            continue
        if any(c.startswith(p) for p in _LABEL_PREFIXES) or c in (
            "target_mult", "hold_mult", "peak_mult", "resolved", "y"
        ):
            continue
        if dt.is_numeric() or dt == pl.Boolean:
            feats.append(c)
    return ["horizon_months", *feats]


def to_long(features: pl.DataFrame) -> pl.DataFrame:
    """Stack horizons into long form with horizon_months as a column."""
    base_drop = [c for c in features.columns
                 if any(c.startswith(p) for p in _LABEL_PREFIXES)]
    base = features.drop(base_drop)
    frames = []
    for h in HORIZONS_MONTHS:
        fr = base.with_columns(
            pl.lit(h).cast(pl.Int32).alias("horizon_months"),
            features[f"sustain_mult_{h}"].alias("target_mult"),
            features[f"hold_mult_{h}"].alias("hold_mult"),
            features[f"peak_mult_{h}"].alias("peak_mult"),
            features[f"resolved_{h}"].alias("resolved"),
        )
        frames.append(fr)
    long = pl.concat(frames, how="vertical")
    return long


@dataclass
class Split:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def gated_split(long: pl.DataFrame, *, val_start: date, test_start: date, data_end: date) -> Split:
    """Leakage-safe split. A row resolves at snapshot + H months; it joins a split only
    if it resolves before the next split starts (train→val_start, val→test_start)."""
    long = long.with_columns(
        (pl.col("snapshot_date")
         + pl.duration(days=(pl.col("horizon_months").cast(pl.Float64) * _DAYS_PER_MONTH).round(0).cast(pl.Int64)))
        .alias("resolve_date")
    ).filter(pl.col("resolved"))

    train = long.filter(
        (pl.col("snapshot_date") < val_start) & (pl.col("resolve_date") <= val_start)
    )
    val = long.filter(
        (pl.col("snapshot_date") >= val_start) & (pl.col("snapshot_date") < test_start)
        & (pl.col("resolve_date") <= test_start)
    )
    test = long.filter(
        (pl.col("snapshot_date") >= test_start) & (pl.col("resolve_date") <= data_end)
    )
    log.info("Gated split: train=%d  val=%d  test=%d", train.height, val.height, test.height)
    return Split(train=train, val=val, test=test)


def add_labels(df: pl.DataFrame, g: float) -> pl.DataFrame:
    return df.with_columns((pl.col("target_mult") >= g).cast(pl.Int8).alias("y"))


__all__ = ["feature_columns", "to_long", "gated_split", "add_labels", "Split",
           "GAIN_LADDER", "HORIZONS_MONTHS"]
