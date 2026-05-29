"""Build the monthly observation panel and the buy-target labels.

Each row is a *buy decision*: `(oracle_id, snapshot_date)` where we could have bought
the card at `entry_price` (trailing-30d median). For each holding window H we record
the **sustained-peak multiple** reached within the next H months — the highest price
level that held for >= SUSTAIN_DAYS consecutive days, i.e. a level you realistically
could have sold into. Binary labels for any gain multiple `g` are derived downstream
by thresholding these multiples, so the model can be queried at any `(g, H)`.

Method (point-in-time, no leakage):
  1. Daily mean price per oracle; keep oracles whose price ever clears the floor.
  2. Reindex to a gapless daily grid per oracle (forward-fill); this makes row-based
     rolling windows equal calendar windows.
  3. entry_price = trailing-30d median.  sustain_level = trailing-14d min (a level
     held for 14 consecutive days).
  4. Forward stats over each H: max sustained_level (sustained-peak), max raw price
     (optimistic peak), and price exactly at t+H (hold-to-deadline) — all as multiples
     of entry_price.  `resolved_H` flags whether the full window is observed.
  5. Keep first-of-month snapshots that clear the liquidity floor and history minimum.

Output: data/processed/panel.parquet
"""
from __future__ import annotations

import logging

import polars as pl

from spikepred.config import (
    ENTRY_MEDIAN_DAYS,
    HORIZONS_MONTHS,
    INTERIM_DIR,
    LIQUIDITY_FLOOR_USD,
    MIN_HISTORY_DAYS,
    PANEL_START,
    PROCESSED_DIR,
    SUSTAIN_DAYS,
)

log = logging.getLogger(__name__)

_DAYS_PER_MONTH = 30.44


def _horizon_days(h_months: int) -> int:
    return round(h_months * _DAYS_PER_MONTH)


def build_panel(
    *,
    interim_dir=INTERIM_DIR,
    processed_dir=PROCESSED_DIR,
    sample_oracles: int | None = None,
    candidate_floor: float | None = None,
) -> pl.DataFrame:
    """Construct the panel. `sample_oracles` limits to N cards for smoke tests."""
    floor = LIQUIDITY_FLOOR_USD if candidate_floor is None else candidate_floor
    gf_path = interim_dir / "prices.parquet"
    log.info("Loading daily card prices from %s", gf_path)
    gf = pl.read_parquet(gf_path).select(["oracle_id", "date", "price_usd"])

    # 1) daily mean price per oracle (across printings/sets)
    daily = (
        gf.group_by(["oracle_id", "date"])
        .agg(pl.col("price_usd").mean().alias("price"))
        .sort(["oracle_id", "date"])
    )

    data_end = daily["date"].max()
    (interim_dir / "_data_end.txt").write_text(str(data_end))
    log.info("Price history ends %s", data_end)

    # keep only oracles that ever clear the floor (a card never >= floor can never
    # have entry_price >= floor, so it can never enter the universe)
    ever = daily.group_by("oracle_id").agg(pl.col("price").max().alias("pmax"))
    keep = ever.filter(pl.col("pmax") >= floor)["oracle_id"]
    if sample_oracles is not None:
        keep = keep.sort()[:sample_oracles]
    daily = daily.filter(pl.col("oracle_id").is_in(keep.implode()))
    log.info("Candidate oracles (price ever >= $%.2f): %d", floor, daily["oracle_id"].n_unique())

    # 2) gapless daily grid per oracle (within each card's [first, last]); forward-fill
    daily = (
        daily.upsample(time_column="date", every="1d", group_by="oracle_id", maintain_order=True)
        .with_columns(pl.col("price").forward_fill().over("oracle_id"))
    )

    # first/last observed date per oracle (for age + censoring) — BEFORE the trailing extension
    bounds = daily.group_by("oracle_id").agg(
        pl.col("date").min().alias("first_obs"),
        pl.col("date").max().alias("last_obs"),
    )

    # 2b) Extend each oracle's grid forward to the GLOBAL data_end, forward-filling its last
    # known price. This lets the "today" snapshot include every liquid card, even ones we didn't
    # refresh in the latest scrape. `bounds` still uses each oracle's actual last_obs, so
    # `resolved_H` correctly flags forward windows as not yet observed for the trailing dates.
    gmax = daily["date"].max()
    to_extend = (daily.group_by("oracle_id").agg(
        pl.col("date").max().alias("_last"),
        pl.col("price").last().alias("_last_price"))
        .filter(pl.col("_last") < gmax))
    if to_extend.height:
        ext = (to_extend
               .with_columns(pl.date_ranges(pl.col("_last") + pl.duration(days=1),
                                            pl.lit(gmax), "1d").alias("date"))
               .explode("date")
               .select(["oracle_id", "date", pl.col("_last_price").alias("price")]))
        log.info("  forward-filling %d oracles up to %s (+%d trailing rows)",
                 to_extend.height, gmax, ext.height)
        daily = pl.concat([daily, ext], how="vertical_relaxed").sort(["oracle_id", "date"])

    # 3) trailing entry price + sustain level
    daily = daily.with_columns(
        pl.col("price").rolling_median(window_size=ENTRY_MEDIAN_DAYS, min_samples=10)
        .over("oracle_id").alias("entry_price"),
        pl.col("price").rolling_min(window_size=SUSTAIN_DAYS, min_samples=SUSTAIN_DAYS)
        .over("oracle_id").alias("sustain_level"),
    )

    # 4) forward stats per horizon (reverse + rolling-max trick = forward window)
    fwd_exprs = []
    for h in HORIZONS_MONTHS:
        hd = _horizon_days(h)
        fwd_exprs.append(
            pl.col("sustain_level").reverse().rolling_max(window_size=hd, min_samples=1)
            .reverse().over("oracle_id").alias(f"fwd_sustain_{h}")
        )
        fwd_exprs.append(
            pl.col("price").reverse().rolling_max(window_size=hd, min_samples=1)
            .reverse().over("oracle_id").alias(f"fwd_peak_{h}")
        )
        fwd_exprs.append(
            pl.col("price").shift(-hd).over("oracle_id").alias(f"price_at_{h}")
        )
    daily = daily.with_columns(fwd_exprs)

    # 5) snapshots: first-of-month + the latest available date (for "today" forecasts)
    latest = daily["date"].max()
    snaps = (
        daily.filter((pl.col("date").dt.day() == 1) | (pl.col("date") == latest))
        .filter(pl.col("date") >= PANEL_START)
        .join(bounds, on="oracle_id", how="left")
    )
    snaps = snaps.with_columns(
        (pl.col("date") - pl.col("first_obs")).dt.total_days().alias("age_days"),
        (pl.col("last_obs") - pl.col("date")).dt.total_days().alias("remaining_days"),
    )
    snaps = snaps.filter(
        (pl.col("entry_price") >= LIQUIDITY_FLOOR_USD)
        & (pl.col("age_days") >= MIN_HISTORY_DAYS)
    )

    # assemble output: multiples + resolved flags per horizon
    out_cols = [
        pl.col("oracle_id"),
        pl.col("date").alias("snapshot_date"),
        pl.col("entry_price"),
        pl.col("age_days"),
        pl.col("first_obs"),
    ]
    for h in HORIZONS_MONTHS:
        hd = _horizon_days(h)
        out_cols += [
            (pl.col(f"fwd_sustain_{h}") / pl.col("entry_price")).alias(f"sustain_mult_{h}"),
            (pl.col(f"fwd_peak_{h}") / pl.col("entry_price")).alias(f"peak_mult_{h}"),
            (pl.col(f"price_at_{h}") / pl.col("entry_price")).alias(f"hold_mult_{h}"),
            (pl.col("remaining_days") >= hd).alias(f"resolved_{h}"),
        ]
    panel = snaps.select(out_cols).sort(["oracle_id", "snapshot_date"])

    out_path = processed_dir / "panel.parquet"
    panel.write_parquet(out_path, compression="zstd")
    log.info(
        "Wrote %s: %d rows, %d oracles, snapshots %s..%s",
        out_path, panel.height, panel["oracle_id"].n_unique(),
        panel["snapshot_date"].min(), panel["snapshot_date"].max(),
    )
    return panel


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_panel()


if __name__ == "__main__":
    main()
