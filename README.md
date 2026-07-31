# Predict Strength

A from-scratch machine learning project investigating: **how predictable is strength
progression from competition data alone?**

## How to Run

```bash
# set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # installs pandas/numpy/sklearn/matplotlib/xgboost + pytest/torch
```

Get the data: place `openpowerlifting.csv` (from
[OpenPowerlifting's public dataset](https://openpowerlifting.gitlab.io/opl-csv/)) in
`data/`. The first run of `load_clean_data()` will clean it and cache the result to
`data/openpowerlifting_processed.csv`.

```bash
# run the tests (validates every from-scratch model against sklearn/xgboost)
pytest tests/

# run the full pipeline end-to-end: clean data -> features -> train -> evaluate
python -m src.pipeline

# individual phase scripts (exploration, sweeps, comparisons) live in scripts/,
# numbered in the order they were built - e.g.:
python scripts/06_boosting.py     # gradient boosting vs. XGBoost, loss curve
python scripts/07_cv_tuning.py    # grouped cross-validation hyperparameter search
python scripts/08_evaluation.py   # subgroup MAE breakdown, bootstrap CI, residual plots
```

Plots get saved to `outputs/`; full number reference for every result in this writeup
lives in `docs/results-summary.md`.

## Research Question

I set out to determine how predictable strength progression is from previous
competition data alone. This question matters for data-driven coaching decisions — it
explores whether a lifter's competition history (past totals, attempts, and results)
contains enough signal to meaningfully predict their future performance.

## Method

Starting from [OpenPowerlifting's public dataset](https://openpowerlifting.gitlab.io/opl-csv/),
I pulled competition data for lifters across federations, weight classes, and meets.
While the data is relatively well-structured, it has real limitations — some
federations, for example, don't report attempt-by-attempt selections at all.

The core prediction task: given a lifter's meet at time *t*, predict their total at
their next meet, *t+1*. Two rules prevent leakage throughout: (a) predicting meet
*t+1* only ever uses data available up to and including meet *t*, and (b) every
train/test split is grouped by lifter — a single lifter's meets are never split across
train and test.

Models tried, in order:

- Historical-average baseline
- Linear regression
- Ridge regression
- Random Forest (default, then tuned)
- Gradient Boosting
- Gradient Boosting, with the fully engineered feature set

## Results

| Model | Features | Config | Test MAE |
|---|---|---|---|
| Historical-average baseline | — | avg_past_delta + population fallback | 38.45 |
| Linear regression (scratch, validated vs. sklearn) | 2 (TotalKg, avg_past_delta) | normal equation | 33.43 |
| Ridge (scratch, validated vs. sklearn) | 9 | λ=1.0 | 33.19 |
| Random Forest (sklearn) | 9 | default (100 trees, unlimited depth) | 33.29 |
| Random Forest (sklearn, tuned) | 9 | n_estimators=300, max_depth=15, min_samples_leaf=10 | 31.90 |
| Gradient Boosting (scratch, validated vs. XGBoost) | 9 | n_estimators=100, lr=0.1, max_depth=3 | 31.99 |
| XGBoost (same config, for comparison) | 9 | n_estimators=100, lr=0.1, max_depth=3 | 32.04 |
| **Gradient Boosting (scratch) — final pipeline model** | **20 (full engineered set)** | **n_estimators=100, lr=0.2, max_depth=3** | **32.00** |

The final model improved on the naive historical baseline by roughly **16.8%**
(38.45kg → 32.00kg MAE).

## Finding 1: The Models Converge

Going from a 2-feature linear model to a 20-feature gradient-boosted ensemble only
moved MAE from 33.43 to 32.00 — about a 4% improvement, despite a massive increase in
both model flexibility and feature count.

Progressing from Ridge → Random Forest → Gradient Boosting and watching them all
converge to around 32-34kg established that improving the model was never going to be
the answer. Linear regression made a significant jump over the naive baseline, but
every model after that produced improvement closer to noise than to a meaningful gain.

## Finding 2: Feature Engineering Didn't Move the Needle

Four separate, real hypotheses were tested for what might explain the remaining
error — attempt-selection behavior, drug-tested status, equipment category and
equipment switches, and relative competitive standing (placement/Dots percentile).
All four produced null results: no change larger than the model's own measurement
noise (the bootstrap confidence interval width).

While one or two null results might be chalked up to poor modeling choices or a
limited feature set, four independent nulls trend toward genuine evidence rather than
coincidence.

## Finding 3: Advanced Lifters Are Genuinely Less Predictable

Grouping predictions by beginner/advanced and by age band showed advanced lifters are
wider on every measure (std, IQR, full range, large-miss rate) — signifying they are
less predictable, not just noisier by chance.

My hypothesis going in was that advanced lifters, being more strategic, would take
riskier attempts for better placing — which should show up as fat-tailed residuals
(usually accurate, with occasional large misses). Instead, kurtosis was *higher* for
beginners, meaning the advanced group's extra error is spread uniformly across the
whole distribution, not concentrated in the tails the way an occasional
high-risk-attempt story would predict. The data supports advanced lifters being
genuinely noisier — just not for the specific reason I expected, based on competition
data alone.

## Limitations

The key limitation: this project has zero training data. Every feature is
reconstructed from past competition results — the model has no way to account for
actual training process, which plausibly matters more than competition outcomes
themselves. Specifically missing:

- No RPE, volume, or intensity data
- No injury or health data
- No coaching, program, or gym-change data
- No true nutrition/weight-cut context
- No meet-intent data
- Survivorship/selection bias — only lifters who competed in sanctioned meets appear at all
- Non-random, meet-level data gaps (some federations/meets simply don't report certain fields)

## Conclusion

Strength progression is predictable to within roughly **32kg** using competition data
alone — a **~16.8% improvement** over the naive historical baseline. Whether that
counts as "pretty predictable" or "mostly noise" is a judgment call: relative to a
lifter's total, 32kg can range from roughly 2-3% (elite/super-heavyweight totals) to
around 10% (lighter weight classes). It's not a precise forecast, but it's a
meaningful, better-than-naive guideline. Closing the remaining gap would most likely
require real training data — RPE, volume, injury history — which this project, by
design, never had access to.
