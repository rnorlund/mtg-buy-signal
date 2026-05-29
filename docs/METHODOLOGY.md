# spikepred methodology — high-level

This document explains *what the model does* and *how it is validated*, without revealing the
trained weights, the proprietary feature-engineering recipes, or the live data pipeline. The
full scientific writeup with figures, calibration curves, walk-forward tables, and the
substitution-event validation lives in [REPORT.pdf](REPORT.pdf).

## 1. Problem framing

For each MTG card on each day, predict the probability the card's market price will reach
**at least g × today's price within H months**.

- Gain targets g ∈ {1.5×, 2×, 3×, 5×}
- Horizons H ∈ {3, 6, 12, 18, 24, 36} months

This is a *targeted* spike-detection problem, not a generic forecast. We don't try to predict
the next price; we predict the probability of crossing a specific threshold within a specific
window. That framing matches what an investor actually needs to decide whether to buy.

## 2. Data

- ~15.7 years of daily MTG card prices (TCG-mid, retail-side benchmark)
- 6,944 liquid cards (entry price ≥ , ≥ 12 months of price history)
- Card metadata from Scryfall: set, frame, type, oracle text, foil-availability, Reserved-List flag
- ~235 engineered features per (card, day)

The live price-ingest pipeline that produces this panel is **not** part of this repo — the
upstream data source's terms don't permit redistribution.

## 3. Two-stage architecture

### Stage 1 — Card-level

For each (card, snapshot) the model emits four calibrated probabilities — one per gain
target — using a gradient-boosted ensemble. Features cover:

- Price dynamics over multiple windows (returns, momentum, drawdown, volatility, range-position)
- Cross-card peer comparisons inside the same set / colour / type
- Game-rules signals derived from oracle text
- Supply-side signals (Reserved List, set type, age, number of printings)

Outputs are isotonically calibrated against a held-out validation block so that
"the model says 70%" really does mean ≈70% empirical hit-rate.

### Stage 2 — Printing-level

Each card name (the *oracle*) can exist in many printings — Beta vs Unlimited vs the modern
reprint. Stage 2 ranks those printings *within an oracle the Stage-1 model already likes*.
A second gradient-boosted ensemble scores per-printing features:

- Per-printing trailing dynamics
- Set type (Core, Expansion, Masters, Commander product, Promo)
- Frame era / age
- Foil vs nonfoil
- Within-oracle context (this printing's price relative to the cheapest available printing)

The result is a *which-version-to-buy* answer, not just a card name. The clearest example is
Berserk: in a hot regime Stage 2 ranks the Beta original (P(≥1.5×) ≈ 90%) far above the
modern Conspiracy reprint (≈ 22%), even though both share the name.

## 4. Validation — sealed, dated, walk-forward

The dataset is split **once** by time:

| Block | Years | Use |
|---|---|---|
| Train | early history → 2020-05 | learn |
| Validation | 2020-06 → 2022-12 | early-stopping, calibrator fitting, hyperparameter tuning |
| **Test** | **2023-01 → 2026-05** | **never touched until the end — held out for final scoring** |

We do not look at the test block while developing. Everything reported as "held-out
precision" comes from the sealed window above.

### Walk-forward stress test

We additionally re-run the entire training/validation/test pipeline as **five rolling 1-year
folds** (test = 2020, 2021, 2022, 2023, 2024). This surfaces regime risk honestly:

- Folds 2020–2022, 2024: top-20 precision @ 1.5× = 60–95%
- Fold 2023: 25% — the COVID hangover / WotC product-flood era — exposes the only
  fold where the model degrades materially. The 2024 recovery to 95% shows that once
  cooling-era data enters the training set, the model adapts.

This is documented in REPORT.pdf §13. We don't hide the bad fold.

## 5. Production scoring

For the daily forecasts we publish:

1. Score every liquid card with Stage 1 for the target window
2. Blend the calibrated Stage-1 probability with a separately-validated 30-day price-momentum
   signal (per-target weight, tuned on validation only)
3. Apply the persistence streak overlay (★N = number of consecutive prior forecasts the
   card has appeared in — confidence that the model isn't flickering)
4. Apply the capacity tag (HIGH / MED / LOW based on entry price × number of printings ×
   Reserved-List flag — how many copies the market can absorb before a recommendation
   front-runs itself)
5. For each picked oracle, run Stage 2 and emit the recommended specific printing

## 6. Net-of-fees realism

We don't report gross peak returns as if they were realisable. The PDF and our backtests
report three exit assumptions:

| Exit | What it models |
|---|---|
| Gross sustained-peak | The card *could* have been sold at this level, ignoring all costs |
| Net retail sale | TCG / eBay retail sale: 12.5% platform+payment fee, $1.50 shipping |
| Net buylist sale | Card Kingdom / Cardmarket dealer pays ~55% of retail, $1.50 ship |

The third row is the conservative one — and the one a casual seller will likely realise.
Random-baseline picks are net-negative under that exit, which is a meaningful moat.

## 7. What is open in this repo

- The framing, features-by-category list, validation protocol, calibration approach
- Time-blocked split / gating logic ([`src/spikepred_public/model/dataset.py`](../src/spikepred_public/model/dataset.py))
- Walk-forward harness ([`src/spikepred_public/model/walk_forward.py`](../src/spikepred_public/model/walk_forward.py))
- The benchmark / leakage-check skeleton ([`src/spikepred_public/model/benchmark.py`](../src/spikepred_public/model/benchmark.py))
- Public feature-build skeleton ([`src/spikepred_public/features/build.py`](../src/spikepred_public/features/build.py))
- The full scientific report ([REPORT.pdf](REPORT.pdf))
- Every dated forecast we have ever published ([../forecasts/](../forecasts/))

## 8. What is not open in this repo

- Trained model weights and calibrators
- Stage-2 per-printing model
- Production scoring blend weights
- The live price-ingest pipeline (paid data source)
- Premium MTGGoldfish auth (operationally critical, contractually private)

These are commercial assets. Contact Cameraderie Cards if you want access for a
non-personal use case.

## 9. Known caveats

- **Calibration is regime-dependent.** Absolute probabilities are optimistic in the calmest
  parts of the cycle. The **ranking** (this card is more likely than that card) is what to
  trust, not the raw probability number. We report this transparently in REPORT.pdf.
- **Capacity matters.** A top pick that's only printed once and trades 5 copies a week can be
  moved by a single newsletter. We tag picks HIGH / MED / LOW for this reason.
- **Set coverage on deep history.** The earliest sets (Alpha/Beta, dual lands, full Vintage
  staples) have spotty price coverage in our deep-history source. This affects ~29% of the
  set universe by count but a much smaller fraction of liquid trading volume. Documented
  in our internal data-coverage notes.
