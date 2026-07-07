# Strength Predictability Project — Feature Engineering Plan

## Objective
Transform sparse meet data into features that approximate a lifter's hidden training state.

We only observe competitions, so all features are **inference-based proxies**.

---

## 1. Core Strength Features

These describe current performance state.

- current_total
- current_squat
- current_bench
- current_deadlift
- bodyweight
- DOTS score (optional normalization)
- strength_to_weight_ratio

---

## 2. Momentum Features

Approximate training progress over time.

- prev_total_change = total_t - total_t-1
- rolling_avg_change_last_2_meets
- rolling_avg_change_last_3_meets
- linear_trend_slope_last_N_meets
- is_improving (binary)

---

## 3. Experience Features

Capture where lifter is in their career.

- meet_number (career count)
- age
- years_active (from first recorded meet)
- age_squared (nonlinear effect)

---

## 4. Time Gap Features

Proxy for training block length.

- days_since_last_meet
- meets_per_year
- short_gap_flag (< 60 days)

---

## 5. Bodyweight Dynamics

Important signal for strength changes.

- bodyweight_current
- bodyweight_change_since_last_meet
- weight_class_change
- relative_bodyweight_change (%)

---

## 6. Stability / Consistency Features

Measure volatility in performance.

- std_last_3_totals
- attempt_success_rate_last_meet
- failed_attempts_last_meet
- consistency_score (inverse variance)

---

## 7. Fatigue Proxies

Indirect signals of recovery/training stress.

- meets_last_6_months
- back_to_back_meets_flag
- short_recovery_flag

---

## 8. Relative Ranking Features (optional)

If computable:

- percentile_in_weight_class
- percentile_in_federation
- competition_strength_index

---

## 9. Feature Construction Rule

For every feature:
> Only use information available BEFORE the prediction meet.

Avoid leakage:
- never use future meets
- never use future totals

---

## 10. Minimal Viable Feature Set (recommended start)

Start simple:

- current_total
- bodyweight
- age
- prev_total_change
- meet_number
- days_since_last_meet
- bodyweight_change

Then iterate upward in complexity.

---

## Core Insight

These features do NOT represent training.

They represent:

> reconstructed signals of training state inferred from competition outcomes.