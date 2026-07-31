# Results Summary — Reference Sheet for the Phase 10 Writeup

All MAE values are in kg, on `next_TotalKg`. All numbers below use the leakage-safe,
lifter-grouped train/test split from Phase 0 unless marked "CV" (grouped k-fold, Phase 7).

---

## 1. Model comparison (headline numbers)

| Model | Features | Config | Test MAE |
|---|---|---|---|
| Historical-average baseline (Phase 1) | — | avg_past_delta + population fallback | 38.45 |
| Linear regression (scratch, validated vs. sklearn) | 2 (TotalKg, avg_past_delta) | normal equation | 33.43 |
| Ridge (scratch, validated vs. sklearn) | 9 | λ=1.0 | 33.19 |
| Random Forest (sklearn) | 9 | default (100 trees, unlimited depth) | 33.29 |
| Random Forest (sklearn, tuned) | 9 | **n_estimators=300, max_depth=15, min_samples_leaf=10** | **31.90** |
| Gradient Boosting (scratch, validated vs. XGBoost) | 9 | n_estimators=100, lr=0.1, max_depth=3 | 31.99 |
| XGBoost (same config, for comparison) | 9 | n_estimators=100, lr=0.1, max_depth=3 | 32.04 |
| **Gradient Boosting (scratch) — final pipeline model** | **20 (full engineered set)** | **n_estimators=100, lr=0.2, max_depth=3** | **32.00** |

**Key takeaway:** going from a 2-feature linear model to a 20-feature gradient-boosted
ensemble only moved MAE from 33.43 → 32.00 — about 4% improvement despite a massive
increase in both model flexibility and feature count.

---

## 2. Random Forest hyperparameter sweep (9 features)

| n_estimators | max_depth | min_samples_leaf | MAE |
|---|---|---|---|
| 100 | None | 1 (default) | 33.29 |
| 200 | 10 | 20 | 32.07 |
| 300 | 20 | 5 | 32.13 |
| 200 | None | 50 | 31.93 |
| **300** | **15** | **10** | **31.90 (best)** |

## 3. Ridge λ-sweep (9 features, single split)

| λ | validation MAE |
|---|---|
| 0 | 33.1856 (best) |
| 0.1 | 33.1856 |
| 1 | 33.1856 |
| 10 | 33.1865 |
| 100 | 33.1948 |
| 1000 | 33.2889 |
| 10000 | 34.8880 (underfitting) |

Essentially flat from λ=0 to λ=100 — evidence there's no meaningful overfitting for
Ridge to correct in this data/feature regime (9 features, ~1M+ rows); regularization
only starts to hurt (underfit) once λ gets very large.

## 4. Grouped 5-fold CV results (Phase 7, `07_cv_tuning.py`)

**Ridge:**
| λ | CV MAE |
|---|---|
| 0.1 | 33.8556 (best) |
| 1 | 33.8557 |
| 10 | 33.8565 |
| 100 | 33.8646 |

**Gradient Boosting:**
| learning_rate | n_estimators | CV MAE |
|---|---|---|
| 0.05 | 50 | 39.1056 (underfit — too few rounds at low lr) |
| 0.05 | 100 | 33.5573 |
| 0.1 | 50 | 33.5406 |
| 0.1 | 100 | 32.6449 |
| 0.2 | 50 | 32.7857 |
| 0.2 | 100 | **32.5487 (best)** |

**Single-split vs. CV note:** the single train/test split MAE for (lr=0.1, n_est=100)
was 31.99, but the 5-fold CV average for the *same* config was 32.6449 — a ~0.65kg
gap, illustrating that a single split can be optimistic. The bootstrap CI (below)
independently confirms wobble of similar magnitude (~0.5kg).

---

## 5. Bootstrap 90% confidence interval (final model, 1000 resamples)

**[31.74, 32.27]** — width ≈ 0.53kg. The model's own performance estimate is stable to
about half a kilogram; treat any comparison smaller than this as noise, not a real effect.

---

## 6. Subgroup analysis (final 20-feature model)

**Beginner (meet_number ≤ 8) vs. advanced (> 8):**
*Note: true beginners (1-4 meets) are structurally unobservable — features like
`rolling_avg_change_last_3_meets` require ≥5 meets of history, so every row surviving
`dropna` already has meet_number ≥ 5.*

| group | MAE | count |
|---|---|---|
| beginner | 29.48 | 36,577 |
| advanced | 34.70 | 34,078 |
| **gap** | **~5.2kg** | |

**Age bands:**

| band | MAE | count |
|---|---|---|
| <20 | 31.53 | 13,776 |
| 20-29 | 33.27 | 24,632 |
| 30-39 | 33.57 | 14,363 |
| 40+ | 29.35 | 17,884 |

**Residual distribution shape, beginner vs. advanced** (before attempt-level features
were added, but shape is representative):

| | beginner | advanced |
|---|---|---|
| std | 50.0 | 58.6 |
| IQR (25-75) | 34.6 | 38.6 |
| 1st-99th pct range | 300.3 | 353.1 |
| excess kurtosis | 21.1 | 17.7 |
| % \|residual\| > 100kg | 4.4% | 6.4% |
| % \|residual\| > 200kg | 1.0% | 1.5% |

Advanced lifters are wider on every measure (std, IQR, full range, large-miss rate) —
genuinely less predictable, not just noisier by chance. But kurtosis is *higher* for
beginners, not advanced — meaning the advanced group's extra error is spread fairly
uniformly across the whole distribution, not concentrated in occasional extreme
"high-risk-attempt" blowups the way a fat-tail story would predict.

---

## 7. Feature engineering experiment log — four rounds, all null

Started from the 9-feature model and added engineered features in four separate
rounds, each testing a distinct hypothesis for why advanced lifters are harder to
predict. All results use Gradient Boosting (lr=0.2, n_estimators=100, max_depth=3).

| Round | Added | # features | advanced MAE | beginner MAE | gap | overall CI |
|---|---|---|---|---|---|---|
| 0 (start) | — | 9 | 34.748 | 29.480 | 5.268 | [31.76, 32.27] |
| 1 | attempt-level (missed attempts, opener→best % gain, has_attempt_data) | 12 | 34.740 | 29.473 | 5.267 | [31.74, 32.28] |
| 2 | is_tested | 13 | 34.762 | 29.435 | 5.327 | [31.73, 32.28] |
| 3 | equipment (one-hot + equipment_changed) | 18 | 34.706 | 29.451 | 5.256 | [31.71, 32.25] |
| 4 (final) | place_last_meet, dots_percentile_in_meet | 20 | 34.698 | 29.483 | 5.215 | [31.74, 32.27] |

**Net change, round 0 → round 4:** overall CI essentially unchanged
([31.76, 32.27] → [31.74, 32.27]); beginner/advanced gap unchanged within noise
(5.268 → 5.215kg, both well inside the ~0.53kg CI width repeated many times over —
i.e., not a real trend). Every individual round's change was smaller than the
bootstrap CI's own width.

**Hypotheses tested and rejected (as *the* explanation, at least via these specific
features):**
- Attempt-selection / risk-taking behavior (missed attempts, opener-to-final jump size)
- Drug-tested status
- Equipment category and equipment switches between meets
- Relative competitive standing (placement, Dots percentile within meet)

---

## 8. Bottom line for the writeup

- Best model: **Gradient Boosting, MAE ≈ 32.0kg**, 90% CI **[31.74, 32.27]**.
- Improvement over the historical-average baseline (38.45kg): **~16.8%**. This is
  the fair comparison per the Phase 1 self-check — the "predict zero change" baseline
  would be an even lower, less meaningful bar to clear, since it ignores every bit of
  per-lifter signal the historical-average baseline already uses.
- Three structurally different model families (Ridge, Random Forest, Gradient
  Boosting) all converge to ~32-34kg MAE regardless of flexibility.
- Four independent rounds of feature engineering, each backed by a specific
  real-world hypothesis, produced zero net movement beyond noise.
- Advanced lifters are measurably (not just apparently) less predictable than
  beginners (~5.2kg MAE gap), but the *shape* of that extra error (uniform
  inflation, not fat tails) doesn't match a "high-risk-attempt" story, and no
  tested feature explains it away.
- **Combined, this is convergent evidence for a real noise floor around 32kg** —
  not a limitation of model choice or feature engineering effort, but a ceiling on
  what competition-result data alone can predict about future strength progression.

---

## 9. Limitations of the feature set

Worth stating explicitly in the writeup, not just implied: this project has **zero
training data**. Every feature, from Phase 1's `avg_past_delta` through the final
20-column set, is reconstructed entirely from public competition results — nothing
here observes the actual training process. This isn't a gap that better feature
engineering can close; it's a ceiling on what this *kind* of data can ever capture.
Specifically missing, with no proxy available anywhere in this dataset:

- **No RPE, volume, or intensity data.** No sense of how hard a training block was,
  how close to failure sets were taken, or how training load was structured/periodized
  leading into a meet.
- **No injury or health data.** An injury, illness, or nagging joint issue could
  easily explain a stalled or declined total — completely invisible here. This is a
  particularly strong candidate for *part* of the "advanced lifters are less
  predictable" gap (Section 6): more experienced lifters training at higher
  intensities for longer plausibly accumulate more injury history, but this dataset
  has no way to confirm or rule that out.
- **No coaching, program, or gym-change data.** A new coach, a new program, or a gym
  switch could all produce a real jump or plateau that looks like unexplained noise.
- **No true nutrition/weight-cut context.** `bodyweight_change` is a crude proxy — it
  can't distinguish a deliberate weight-class cut from natural fluctuation, or muscle
  gain from fat gain.
- **No meet-intent data.** Whether a given meet was an all-out peak attempt or a
  low-stakes "opener-only" tune-up isn't recorded anywhere directly. The attempt-level
  features (Section 7, round 1) were a best-effort proxy for this and produced a null
  result — but a null result from an imperfect proxy doesn't rule out that intent
  matters, only that *this specific* measurable version of it didn't help.
- **Survivorship / selection bias in who appears at all.** The dataset only contains
  people who competed *and* got recorded by OpenPowerlifting. Lifters who trained
  seriously but never competed, or who stopped competing entirely (injury, life
  circumstances, loss of interest), are invisible — the population studied here is
  specifically "people who kept competing," not "lifters" in general.
- **Non-random, meet-level data gaps.** Directly discovered in this project (Section
  7): attempt-level data, `Tested` status, and similar fields are missing in
  meet-sized clusters, not at random. Some subpopulations (certain federations,
  true beginners with <5 meets) are structurally less observable than others in this
  dataset, independent of anything about their actual predictability.

The honest framing for the writeup: this project answers *"how predictable is future
strength total from public competition history alone,"* not *"how predictable is
strength progression"* in the fully general sense. The ~32kg noise floor found here is
a floor on that narrower, data-source-constrained question — a genuinely different,
and answerable, question from "how predictable would progression be with real training
data," which this project cannot speak to either way.
