"""Build the point-in-time feature matrix for the buy-target model.

For every `(oracle_id, snapshot_date)` in the panel, compute features knowable ONLY
from data dated `<= snapshot_date`. Feature blocks:

  price_dynamics   — trailing momentum / volatility / drawdown / CAGR / RSI / spikiness
  supply_scarcity  — printings count, recency, age, recent-reprint, reserved-list, float
  reprint_risk     — sibling-model P(reprint) (low risk => spikes sustain)
  demand           — EDHREC inclusion / rank / salt
  intrinsics       — rarity, type, colors, cmc, keywords (reserved-list flag)
  cross_sectional  — log-price z-score within rarity bucket at the snapshot
  calendar         — month-of-year cyclical, year

Output: data/processed/features.parquet  (panel labels + features, one row per obs).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from spikepred.config import (
    ENTRY_MEDIAN_DAYS,
    INTERIM_DIR,
    PROCESSED_DIR,
    REPRINT_RISK_CSV,
)

log = logging.getLogger(__name__)

_PRIMARY_TYPES = (
    "Creature", "Instant", "Sorcery", "Artifact",
    "Enchantment", "Planeswalker", "Land", "Battle",
)
_RARITIES = ("common", "uncommon", "rare", "mythic", "special", "bonus")


# --------------------------------------------------------------------------
# Block 1: trailing price dynamics (the core signal), as-of snapshot_date
# --------------------------------------------------------------------------

def _price_dynamics(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    log.info("Computing trailing price-dynamics features...")
    gf = pl.read_parquet(interim_dir / "prices.parquet").select(
        ["oracle_id", "date", "price_usd"]
    )
    # only need oracles that appear in the panel
    oracles = panel.select("oracle_id").unique()
    daily = (
        gf.join(oracles, on="oracle_id", how="inner")
        .group_by(["oracle_id", "date"])
        .agg(pl.col("price_usd").mean().alias("price"))
        .sort(["oracle_id", "date"])
        .upsample(time_column="date", every="1d", group_by="oracle_id", maintain_order=True)
        .with_columns(pl.col("price").forward_fill().over("oracle_id"))
    )
    # Extend each oracle's grid forward to GLOBAL data_end (same as build_panel does), so
    # features ARE computed at the latest snapshot for every card. Without this, stale cards
    # get NaN features at the latest snapshot, which the model treats as missing and predict's
    # `f_ret_30d.fill_null(-1.0)` then forces ret_30d to -100% — corrupting the buy ranking.
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
        log.info("  forward-filling %d oracles to %s (+%d trailing feature rows)",
                 to_extend.height, gmax, ext.height)
        daily = pl.concat([daily, ext], how="vertical_relaxed").sort(["oracle_id", "date"])

    d = daily.with_columns(
        pl.col("price").log().alias("logp"),
        pl.col("price").pct_change().over("oracle_id").alias("ret1d"),
    )
    # trailing shifts for momentum
    for n in (30, 90, 180, 365):
        d = d.with_columns(pl.col("price").shift(n).over("oracle_id").alias(f"p_{n}"))
    for y in (3, 5):
        d = d.with_columns(pl.col("price").shift(365 * y).over("oracle_id").alias(f"p_y{y}"))
    # trailing rolling stats
    d = d.with_columns(
        pl.col("ret1d").rolling_std(window_size=90, min_samples=20).over("oracle_id").alias("vol_90"),
        pl.col("price").rolling_max(window_size=365, min_samples=20).over("oracle_id").alias("hi_365"),
        pl.col("price").rolling_mean(window_size=30, min_samples=10).over("oracle_id").alias("ma_30"),
        pl.col("price").rolling_mean(window_size=90, min_samples=20).over("oracle_id").alias("ma_90"),
        pl.col("price").rolling_mean(window_size=180, min_samples=30).over("oracle_id").alias("ma_180"),
        pl.col("price").rolling_mean(window_size=365, min_samples=60).over("oracle_id").alias("mean_365"),
        pl.col("price").rolling_std(window_size=365, min_samples=60).over("oracle_id").alias("std_365"),
        pl.col("price").rolling_mean(window_size=7, min_samples=3).over("oracle_id").alias("ma_7"),
    )
    # RSI(14): from up/down moves
    d = d.with_columns(
        pl.when(pl.col("ret1d") > 0).then(pl.col("ret1d")).otherwise(0.0).alias("up"),
        pl.when(pl.col("ret1d") < 0).then(-pl.col("ret1d")).otherwise(0.0).alias("dn"),
    ).with_columns(
        pl.col("up").rolling_mean(window_size=14, min_samples=5).over("oracle_id").alias("avg_up"),
        pl.col("dn").rolling_mean(window_size=14, min_samples=5).over("oracle_id").alias("avg_dn"),
    )
    # realized-spike count: day price > 1.25 * trailing 7d mean
    d = d.with_columns(
        ((pl.col("price") > 1.25 * pl.col("ma_7")).cast(pl.Int8))
        .rolling_sum(window_size=180, min_samples=1).over("oracle_id").alias("spikes_180"),
        ((pl.col("price") > 1.25 * pl.col("ma_7")).cast(pl.Int8))
        .rolling_sum(window_size=365, min_samples=1).over("oracle_id").alias("spikes_365"),
    )
    # days since trailing-365 high
    d = d.with_columns(
        (pl.col("price") >= pl.col("hi_365") * 0.999).cast(pl.Int8).alias("is_hi"),
    )

    feats = d.select(
        "oracle_id", pl.col("date").alias("snapshot_date"),
        "price", "logp", "vol_90", "hi_365", "ma_30", "ma_90", "ma_180",
        "mean_365", "std_365", "avg_up", "avg_dn", "spikes_180", "spikes_365",
        "p_30", "p_90", "p_180", "p_365", "p_y3", "p_y5",
    )
    # derive the final point-in-time features
    feats = feats.with_columns(
        pl.col("logp").alias("f_log_price"),
        (pl.col("price") / pl.col("p_30") - 1).alias("f_ret_30d"),
        (pl.col("price") / pl.col("p_90") - 1).alias("f_ret_90d"),
        (pl.col("price") / pl.col("p_180") - 1).alias("f_ret_180d"),
        (pl.col("price") / pl.col("p_365") - 1).alias("f_ret_365d"),
        ((pl.col("price") / pl.col("p_y3")) ** (1.0 / 3) - 1).alias("f_cagr_3y"),
        ((pl.col("price") / pl.col("p_y5")) ** (1.0 / 5) - 1).alias("f_cagr_5y"),
        pl.col("vol_90").alias("f_vol_90d"),
        (pl.col("price") / pl.col("hi_365") - 1).alias("f_drawdown_365"),
        (pl.col("price") / pl.col("ma_30") - 1).alias("f_dist_ma30"),
        (pl.col("price") / pl.col("ma_90") - 1).alias("f_dist_ma90"),
        (pl.col("price") / pl.col("ma_180") - 1).alias("f_dist_ma180"),
        ((pl.col("price") - pl.col("mean_365")) / pl.col("std_365")).alias("f_zscore_365"),
        (pl.col("avg_up") / (pl.col("avg_up") + pl.col("avg_dn"))).alias("f_rsi_14"),
        pl.col("spikes_180").alias("f_spikes_180d"),
        pl.col("spikes_365").alias("f_spikes_365d"),
    )
    fcols = [c for c in feats.columns if c.startswith("f_")]
    feats = feats.select("oracle_id", "snapshot_date", *fcols)

    # exact join (panel snapshots are first-of-month, present in the daily grid)
    return panel.join(feats, on=["oracle_id", "snapshot_date"], how="left")


# --------------------------------------------------------------------------
# Block 1b: comprehensive archetype/group flags + peer-group momentum
# --------------------------------------------------------------------------

# Validated on the data: 0-cost artifacts double ~33% vs 25% baseline and co-move ~6x more
# than random (mean pairwise return corr 0.19 vs 0.03). Cards in a group spike TOGETHER, so a
# heating group is a leading indicator for its laggards. We tag each card with many overlapping
# group flags (type / mana / lands / engines / combo / commander / ~40 tribes), pick its most
# cohesive group, and compute the leave-one-out mean trailing return of that group per snapshot.
# All point-in-time (uses only trailing returns already known at snapshot_date).

# Top creature subtypes (tribal demand spikes when a new tribal commander prints).
_TRIBES = (
    "Dragon", "Elf", "Goblin", "Zombie", "Vampire", "Human", "Wizard", "Merfolk", "Angel",
    "Demon", "Beast", "Soldier", "Warrior", "Knight", "Spirit", "Cat", "Dog", "Snake", "Sliver",
    "Dinosaur", "Hydra", "Elemental", "Giant", "Cleric", "Rogue", "Faerie", "Treefolk", "Wolf",
    "Bird", "Insect", "Construct", "Golem", "Horror", "Eldrazi", "Phoenix", "Minotaur", "Ooze",
    "Pirate", "Ninja", "Samurai", "Shaman", "Druid", "Assassin", "Spider", "Fungus",
)

# Functional groups via regex on lowercased oracle text.
_TEXT_RULES = {
    "tutor": r"search your library",
    "card_draw": r"\bdraw\b",
    "draw_engine": r"draw .*(whenever|at the beginning|each)",
    "counterspell": r"counter target",
    "spot_removal": r"(destroy|exile) target",
    "board_wipe": r"(destroy|exile) all|each (creature|player|opponent)",
    "recursion": r"from your graveyard",
    "reanimator": r"return target .*creature card .*from .*graveyard to the battlefield",
    "extra_turn": r"extra turn",
    "extra_combat": r"additional combat",
    "stax_tax": r"can't|cost.* more|don't untap|skip (your|that)",
    "token_maker": r"create .*token",
    "sac_outlet": r"sacrifice (a|another|an)",
    "lifegain": r"gain .*life",
    "mill": r"mill|put .* into .*graveyard",
    "burn": r"deals? \d+ damage to (any target|target player|each)",
    "discard": r"discard",
    "land_destruction": r"destroy target land",
    "protection": r"hexproof|indestructible|protection from|shroud|\bward\b",
    "free_cast": r"without paying its mana cost",
    "untapper": r"untap (target|all|another)",
    "copy_effect": r"copy (target|that)|create a (token that's a )?copy",
    "cost_reducer": r"cost .*less to cast",
    "storm_cascade": r"\bstorm\b|\bcascade\b",
    "ramp": r"search your library for .*land|add \{[wubrgc]",
    "lord_anthem": r"(other )?creatures you control get \+",
}


def _group_flags(o: pl.DataFrame) -> pl.DataFrame:
    tl = pl.col("type_line").fill_null("")
    txt = pl.col("oracle_text").fill_null("").str.to_lowercase()
    mc = pl.col("mana_cost").fill_null("")
    cmc = pl.col("cmc")
    is_art = tl.str.contains("Artifact", literal=True)
    is_land = tl.str.contains("Land", literal=True)
    is_crea = tl.str.contains("Creature", literal=True)

    exprs: list = []
    # --- type flags ---
    for t, pat in [("creature", "Creature"), ("instant", "Instant"), ("sorcery", "Sorcery"),
                   ("artifact", "Artifact"), ("enchantment", "Enchantment"), ("land", "Land"),
                   ("planeswalker", "Planeswalker"), ("battle", "Battle"), ("equipment", "Equipment"),
                   ("aura", "Aura"), ("vehicle", "Vehicle"), ("saga", "Saga"), ("class", "Class"),
                   ("kindred", "Kindred")]:
        exprs.append(tl.str.contains(pat, literal=True).fill_null(False).cast(pl.Int8).alias(f"grp_{t}"))
    # --- mana / fast-mana flags ---
    exprs += [
        (cmc == 0).cast(pl.Int8).alias("grp_zero_mana"),
        ((cmc == 0) & is_art).cast(pl.Int8).alias("grp_zero_artifact"),
        (cmc == 1).cast(pl.Int8).alias("grp_one_drop"),
        ((is_art | is_land) & (cmc <= 1) & txt.str.contains(r"add \{")).cast(pl.Int8).alias("grp_fast_mana"),
        (is_art & txt.str.contains(r"\{t\}: add")).cast(pl.Int8).alias("grp_mana_rock"),
        (is_crea & txt.str.contains(r"\{t\}: add")).cast(pl.Int8).alias("grp_mana_dork"),
        mc.str.contains(r"\{X\}", literal=False).cast(pl.Int8).alias("grp_x_spell"),
        mc.str.contains("/P", literal=True).cast(pl.Int8).alias("grp_phyrexian"),
        (mc.str.contains("/", literal=True) & ~mc.str.contains("/P", literal=True)).cast(pl.Int8).alias("grp_hybrid"),
    ]
    # --- land subtypes ---
    exprs += [
        (is_land & ~tl.str.contains("Basic", literal=True)).cast(pl.Int8).alias("grp_nonbasic_land"),
        (is_land & txt.str.contains("search your library") & txt.str.contains("land")).cast(pl.Int8).alias("grp_fetch_land"),
        (is_land & txt.str.contains(r"add \{[wubrg]\}\{[wubrg]\}|add \{[wubrg]\} or \{[wubrg]\}")).cast(pl.Int8).alias("grp_dual_land"),
        (is_land & txt.str.contains("becomes a") & txt.str.contains("creature")).cast(pl.Int8).alias("grp_creature_land"),
    ]
    # --- functional text rules ---
    for name, pat in _TEXT_RULES.items():
        exprs.append(txt.str.contains(pat).fill_null(False).cast(pl.Int8).alias(f"grp_{name}"))
    # --- commander flags ---
    is_legend = tl.str.contains("Legendary", literal=True)
    exprs += [
        is_legend.cast(pl.Int8).alias("grp_legendary"),
        (is_legend & is_crea).cast(pl.Int8).alias("grp_legend_creature"),
        txt.str.contains("can be your commander").cast(pl.Int8).alias("grp_can_be_commander"),
        txt.str.contains(r"\bpartner\b").cast(pl.Int8).alias("grp_partner"),
        tl.str.contains("Background", literal=True).cast(pl.Int8).alias("grp_background"),
    ]
    # --- color-identity buckets ---
    cil = pl.col("color_identity").list.len()
    exprs += [
        (cil == 0).cast(pl.Int8).alias("grp_colorless"),
        (cil == 1).cast(pl.Int8).alias("grp_mono"),
        (cil == 2).cast(pl.Int8).alias("grp_two_color"),
        (cil >= 3).cast(pl.Int8).alias("grp_three_plus_color"),
    ]
    # --- tribal multi-hot ---
    for tribe in _TRIBES:
        exprs.append((is_crea & tl.str.contains(tribe, literal=True)).fill_null(False).cast(pl.Int8).alias(f"grp_tribe_{tribe.lower()}"))

    o = o.with_columns(exprs)

    # primary tribe (priority = list order) for the cohesive grouping
    chain = None
    for tribe in _TRIBES:
        cond = pl.col(f"grp_tribe_{tribe.lower()}") == 1
        chain = pl.when(cond).then(pl.lit(f"tribe_{tribe.lower()}")) if chain is None else chain.when(cond).then(pl.lit(f"tribe_{tribe.lower()}"))
    tribe_primary = chain.otherwise(None)

    # functional cohesive group (priority order) when no tribe applies
    func = (
        pl.when(pl.col("grp_zero_artifact") == 1).then(pl.lit("zero_artifact"))
        .when(pl.col("grp_fast_mana") == 1).then(pl.lit("fast_mana"))
        .when(pl.col("grp_fetch_land") == 1).then(pl.lit("fetch_land"))
        .when(pl.col("grp_dual_land") == 1).then(pl.lit("dual_land"))
        .when(pl.col("grp_creature_land") == 1).then(pl.lit("creature_land"))
        .when(pl.col("grp_nonbasic_land") == 1).then(pl.lit("nonbasic_land"))
        .when(pl.col("grp_mana_rock") == 1).then(pl.lit("mana_rock"))
        .when(pl.col("grp_mana_dork") == 1).then(pl.lit("mana_dork"))
        .when(pl.col("grp_tutor") == 1).then(pl.lit("tutor"))
        .when(pl.col("grp_planeswalker") == 1).then(pl.lit("planeswalker"))
        .when(pl.col("grp_equipment") == 1).then(pl.lit("equipment"))
        .when(pl.col("grp_legend_creature") == 1).then(pl.lit("legend_creature"))
        .when(pl.col("grp_enchantment") == 1).then(pl.lit("enchantment"))
        .when(pl.col("grp_instant") == 1).then(pl.lit("instant"))
        .when(pl.col("grp_sorcery") == 1).then(pl.lit("sorcery"))
        .otherwise(pl.lit("other"))
    )
    return o.with_columns(pl.coalesce([tribe_primary, func]).alias("cohesive_group"))


def _archetype_peer_features(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    """Requires f_ret_30d / f_ret_90d from the price-dynamics block to already exist."""
    log.info("Computing archetype flags + peer-group momentum...")
    o = pl.read_parquet(interim_dir / "cards_oracle.parquet").select(
        ["oracle_id", "cmc", "type_line", "oracle_text", "mana_cost", "color_identity"]
    )
    o = _group_flags(o)
    flag_cols = [c for c in o.columns if c.startswith("grp_")]
    panel = panel.join(o.select(["oracle_id", "cohesive_group", *flag_cols]), on="oracle_id", how="left")
    panel = panel.with_columns(pl.col("cohesive_group").fill_null("other"))
    log.info("  %d group flags + cohesive peer-momentum on %d distinct groups",
             len(flag_cols), panel["cohesive_group"].n_unique())

    # leave-one-out peer mean of trailing return within (snapshot, cohesive_group)
    for ret in ("f_ret_30d", "f_ret_90d"):
        grp = ["snapshot_date", "cohesive_group"]
        peer = ((pl.col(ret).sum().over(grp) - pl.col(ret).fill_null(0))
                / (pl.col(ret).count().over(grp) - pl.col(ret).is_not_null().cast(pl.Int32)).clip(1, None))
        panel = panel.with_columns(peer.alias(f"f_peer{ret[1:]}"))  # f_peer_ret_30d / 90d
    return panel


# --------------------------------------------------------------------------
# Block 2: supply / scarcity (point-in-time from printings)
# --------------------------------------------------------------------------

def _supply_scarcity(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    log.info("Computing supply/scarcity features...")
    prn = pl.read_parquet(interim_dir / "printings.parquet").select(
        ["oracle_id", "released_date", "set_type", "rarity", "reserved"]
    ).filter(pl.col("released_date").is_not_null())

    j = (
        panel.select(["oracle_id", "snapshot_date"])
        .join(prn, on="oracle_id", how="inner")
        .filter(pl.col("released_date") <= pl.col("snapshot_date"))
    )
    agg = (
        j.group_by(["oracle_id", "snapshot_date"]).agg(
            pl.len().alias("n_prior_printings"),
            pl.col("released_date").min().alias("first_print"),
            pl.col("released_date").max().alias("last_print"),
            # recent reprint: any printing in trailing 365d
            ((pl.col("snapshot_date") - pl.col("released_date")).dt.total_days() <= 365)
            .any().cast(pl.Int8).alias("recent_reprint_365"),
        )
    )
    agg = agg.with_columns(
        ((pl.col("snapshot_date") - pl.col("last_print")).dt.total_days() / 30.44)
        .alias("months_since_last_print"),
        ((pl.col("snapshot_date") - pl.col("first_print")).dt.total_days() / 30.44)
        .alias("months_since_first_print"),
    ).drop(["first_print", "last_print"])

    # point-in-time rarity (most recent printing as of snapshot)
    last_rar = (
        j.sort("released_date").group_by(["oracle_id", "snapshot_date"])
        .agg(pl.col("rarity").last().alias("current_rarity"))
    )
    out = panel.join(agg, on=["oracle_id", "snapshot_date"], how="left")
    out = out.join(last_rar, on=["oracle_id", "snapshot_date"], how="left")
    out = out.with_columns(
        pl.col("n_prior_printings").fill_null(1),
        pl.col("recent_reprint_365").fill_null(0),
        pl.col("months_since_last_print").fill_null(0.0),
        pl.col("months_since_first_print").fill_null(pl.col("age_days") / 30.44),
    )
    return out


# --------------------------------------------------------------------------
# Block 3: intrinsics + reserved-list (time-invariant from cards_oracle)
# --------------------------------------------------------------------------

def _intrinsics(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    log.info("Computing card-intrinsic features (incl. reserved-list)...")
    o = pl.read_parquet(interim_dir / "cards_oracle.parquet")
    o = o.with_columns(
        pl.col("cmc").cast(pl.Float32).fill_null(0.0),
        pl.col("reserved").fill_null(False).cast(pl.Int8).alias("is_reserved_list"),
        pl.col("type_line").str.contains("Legendary", literal=True).fill_null(False).cast(pl.Int8).alias("is_legendary"),
    )
    for c in ("W", "U", "B", "R", "G"):
        o = o.with_columns(
            pl.col("color_identity").list.contains(c).cast(pl.Int8).alias(f"ci_{c}")
        )
    o = o.with_columns(pl.col("color_identity").list.len().cast(pl.Int8).alias("n_color_identity"))
    for t in _PRIMARY_TYPES:
        o = o.with_columns(
            pl.col("type_line").str.contains(t, literal=True).fill_null(False).cast(pl.Int8).alias(f"is_{t.lower()}")
        )
    # top-25 keyword multi-hots
    kw_counts: dict[str, int] = {}
    for kws in o["keywords"].to_list():
        for k in (kws or []):
            kw_counts[k] = kw_counts.get(k, 0) + 1
    top_kws = [k for k, _ in sorted(kw_counts.items(), key=lambda x: -x[1])[:25]]
    for kw in top_kws:
        slug = kw.lower().replace(" ", "_").replace("-", "_")
        o = o.with_columns(pl.col("keywords").list.contains(kw).cast(pl.Int8).alias(f"kw_{slug}"))

    keep = [
        "oracle_id", "name", "cmc", "is_reserved_list", "is_legendary", "n_color_identity",
        *[f"ci_{c}" for c in "WUBRG"],
        *[f"is_{t.lower()}" for t in _PRIMARY_TYPES],
        *[f"kw_{kw.lower().replace(' ', '_').replace('-', '_')}" for kw in top_kws],
    ]
    return panel.join(o.select(keep), on="oracle_id", how="left")


# --------------------------------------------------------------------------
# Block 4: demand (EDHREC) + reprint-risk score
# --------------------------------------------------------------------------

def _demand_and_reprintrisk(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    log.info("Joining demand (EDHREC) + reprint-risk score...")
    edh_path = interim_dir / "edhrec.parquet"
    if edh_path.exists():
        edh = pl.read_parquet(edh_path)
        cols = [c for c in ["oracle_id", "edhrec_inclusion", "edhrec_inclusion_rate",
                            "edhrec_salt", "edhrec_rank"] if c in edh.columns]
        panel = panel.join(edh.select(cols), on="oracle_id", how="left")
    if REPRINT_RISK_CSV.exists():
        rr = pl.read_csv(REPRINT_RISK_CSV)
        rr_cols = [c for c in ["oracle_id", "p_reprint_12mo", "p_reprint_6mo"] if c in rr.columns]
        if len(rr_cols) > 1:
            panel = panel.join(rr.select(rr_cols), on="oracle_id", how="left")
    return panel


# --------------------------------------------------------------------------
# Block 5: cross-sectional + calendar
# --------------------------------------------------------------------------

def _cross_sectional_and_calendar(panel: pl.DataFrame) -> pl.DataFrame:
    log.info("Computing cross-sectional + calendar features...")
    # log-price z within (snapshot_date, rarity)
    panel = panel.with_columns(
        ((pl.col("f_log_price") - pl.col("f_log_price").mean().over(["snapshot_date", "current_rarity"]))
         / (pl.col("f_log_price").std().over(["snapshot_date", "current_rarity"]) + 1e-6))
        .alias("f_logp_z_in_rarity")
    )
    m = pl.col("snapshot_date").dt.month()
    panel = panel.with_columns(
        (2 * np.pi * m / 12).sin().alias("f_month_sin"),
        (2 * np.pi * m / 12).cos().alias("f_month_cos"),
        pl.col("snapshot_date").dt.year().cast(pl.Int32).alias("f_year"),
    )
    # one-hot current_rarity
    for r in _RARITIES:
        panel = panel.with_columns(
            (pl.col("current_rarity") == r).fill_null(False).cast(pl.Int8).alias(f"rarity_{r}")
        )
    return panel


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def _semantic_cluster_peer(panel: pl.DataFrame, interim_dir, k: int = 40) -> pl.DataFrame:
    """Use oracle-text embeddings to *discover* co-moving groups (semantic clusters),
    then add peer-group momentum within each cluster — the robust, low-dimensional way to
    use the embeddings (vs 32 raw dims, which overfit). Leakage-free: clustering is
    unsupervised on static card text. Requires f_ret_30d/90d to already exist.
    """
    path = interim_dir / "oracle_text_embeddings.parquet"
    if not path.exists():
        return panel
    from sklearn.cluster import KMeans

    emb = pl.read_parquet(path)
    txt_cols = [c for c in emb.columns if c.startswith("f_txt_")]
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(emb.select(txt_cols).to_numpy())
    lab = emb.select("oracle_id").with_columns(pl.Series("sem_cluster", labels))
    log.info("Semantic clusters: %d groups from text embeddings; peer-momentum within each", k)
    panel = panel.join(lab, on="oracle_id", how="left").with_columns(pl.col("sem_cluster").fill_null(-1))
    for ret in ("f_ret_30d", "f_ret_90d"):
        grp = ["snapshot_date", "sem_cluster"]
        peer = ((pl.col(ret).sum().over(grp) - pl.col(ret).fill_null(0))
                / (pl.col(ret).count().over(grp) - pl.col(ret).is_not_null().cast(pl.Int32)).clip(1, None))
        panel = panel.with_columns(peer.alias(f"f_semclus{ret[1:]}"))
    return panel.drop("sem_cluster")  # drop the raw label (meaningless as an ordinal feature)


# Commander "Game Changers" list = WotC's de-facto "in danger of being banned" watchlist.
_GAME_CHANGERS = {
    "Drannith Magistrate", "Consecrated Sphinx", "Ad Nauseam", "Gamble", "Enlightened Tutor",
    "Cyclonic Rift", "Bolas's Citadel", "Jeska's Will", "Humility", "Fierce Guardianship",
    "Braids, Cabal Minion", "Underworld Breach", "Smothering Tithe", "Force of Will", "Demonic Tutor",
    "Teferi's Protection", "Gifts Ungiven", "Imperial Seal", "Intuition", "Necropotence",
    "Mystical Tutor", "Opposition Agent", "Narset, Parter of Veils", "Orcish Bowmasters", "Rhystic Study",
    "Tergrid, God of Fright", "Thassa's Oracle", "Vampiric Tutor", "Crop Rotation",
    "Grand Arbiter Augustin IV", "Chrome Mox", "Serra's Sanctum", "Natural Order", "Notion Thief",
    "Grim Monolith", "Gaea's Cradle", "Seedborn Muse", "Aura Shards", "Lion's Eye Diamond",
    "Ancient Tomb", "Survival of the Fittest", "Coalition Victory", "Mana Vault", "Field of the Dead",
    "Worldly Tutor", "Mox Diamond", "Glacial Chasm", "Panoptic Mirror", "Mishra's Workshop",
    "The One Ring", "The Tabernacle at Pendrell Vale",
}
_BAN_FORMATS = ("modern", "legacy", "vintage", "pioneer", "pauper", "commander", "standard")


def _banlist_features(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    """Current banned/restricted status per format + the Game-Changers 'in danger' flag.

    Powerful, format-warping cards (banned in eternal formats, on the watchlist) carry demand
    pressure. Legality is the *current* snapshot (a known approximation for historical rows).
    """
    log.info("Computing banlist + watchlist features...")
    prn = pl.read_parquet(interim_dir / "printings.parquet")
    legal_cols = [f"legal_{f}" for f in _BAN_FORMATS if f"legal_{f}" in prn.columns]
    latest = (prn.filter(pl.col("released_date").is_not_null())
              .sort("released_date", descending=True).unique(subset=["oracle_id"], keep="first")
              .select(["oracle_id", *legal_cols]))
    out = panel.join(latest, on="oracle_id", how="left")
    banned_flags = []
    for f in _BAN_FORMATS:
        col = f"legal_{f}"
        if col in out.columns:
            out = out.with_columns((pl.col(col) == "banned").fill_null(False).cast(pl.Int8).alias(f"banned_{f}"))
            banned_flags.append(f"banned_{f}")
    if "legal_vintage" in out.columns:
        out = out.with_columns((pl.col("legal_vintage") == "restricted").fill_null(False).cast(pl.Int8).alias("restricted_vintage"))
    out = out.with_columns(pl.sum_horizontal(banned_flags).cast(pl.Int8).alias("n_formats_banned")).drop(legal_cols)

    # Game-Changers watchlist (by name)
    cards = pl.read_parquet(interim_dir / "cards_oracle.parquet").select(["oracle_id", "name"])
    gc_ids = cards.filter(pl.col("name").is_in(list(_GAME_CHANGERS)))["oracle_id"]
    out = out.with_columns(pl.col("oracle_id").is_in(gc_ids.implode()).cast(pl.Int8).alias("is_game_changer"))
    return out


def _market_regime(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-snapshot market state (mean trailing return across all cards) + the card's EXCESS
    return vs market. Lets the model know hot vs cold regimes — our biggest generalization gap.
    Point-in-time: only trailing returns known at the snapshot."""
    log.info("Computing market-regime features...")
    for ret in ("f_ret_30d", "f_ret_90d", "f_ret_365d"):
        if ret in panel.columns:
            mkt = pl.col(ret).mean().over("snapshot_date")
            panel = panel.with_columns(
                mkt.alias(f"f_mkt{ret[1:]}"),
                (pl.col(ret) - mkt).alias(f"f_excess{ret[1:]}"),
            )
    return panel


def _text_embeddings(panel: pl.DataFrame, interim_dir) -> pl.DataFrame:
    """Join PCA-reduced oracle-text embeddings (synergy semantics) if present."""
    path = interim_dir / "oracle_text_embeddings.parquet"
    if not path.exists():
        log.info("No oracle_text_embeddings.parquet — skipping text features")
        return panel
    log.info("Joining oracle-text embedding features...")
    return panel.join(pl.read_parquet(path), on="oracle_id", how="left")


def build_features(*, interim_dir=INTERIM_DIR, processed_dir=PROCESSED_DIR,
                   with_text: bool = False, with_clusters: bool = False) -> pl.DataFrame:
    # NOTE: text embeddings (with_text=True) were tested and *hurt* held-out generalization
    # on this small, regime-shifting dataset (ensemble ≥1.5x P@20 0.90 -> 0.80, ≥2x 0.65 -> 0.45).
    # Kept opt-in only; production excludes them. Same lesson as Optuna tuning: more complexity
    # overfits here. Revisit if/when the panel has many more cards or a regime-robust scheme.
    panel = pl.read_parquet(processed_dir / "panel.parquet")
    log.info("Panel: %d rows", panel.height)
    panel = _price_dynamics(panel, interim_dir)
    panel = _archetype_peer_features(panel, interim_dir)   # 107 meaningful group flags + peer momentum
    if with_clusters:  # embedding-discovered groups — tested mixed/neutral, opt-in
        panel = _semantic_cluster_peer(panel, interim_dir)
    if with_text:
        panel = _text_embeddings(panel, interim_dir)       # raw 32-d (opt-in; overfits — see notes)
    panel = _supply_scarcity(panel, interim_dir)
    panel = _intrinsics(panel, interim_dir)
    panel = _demand_and_reprintrisk(panel, interim_dir)
    panel = _banlist_features(panel, interim_dir)
    panel = _market_regime(panel)
    panel = _cross_sectional_and_calendar(panel)

    # sanitize inf -> null (pct ratios can blow up when a prior price was ~0)
    num_cols = [c for c, dt in panel.schema.items()
                if dt in (pl.Float32, pl.Float64)]
    panel = panel.with_columns(
        [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in num_cols]
    )
    out_path = processed_dir / "features.parquet"
    panel.write_parquet(out_path, compression="zstd")
    log.info("Wrote %s: %d rows, %d cols", out_path, panel.height, len(panel.columns))
    return panel


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_features()


if __name__ == "__main__":
    main()
