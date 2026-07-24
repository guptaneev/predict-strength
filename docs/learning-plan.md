# Strength Predictability Project — Learning-First Implementation Plan

## Purpose of This Document

This is not a "ship it fast" plan. The goal is **understanding**, not velocity. Every
phase below maps to something you already decided to build (see `general-structure.md`
and `feature-engineering.md`), broken into: the underlying concept explained in plain
language, what to implement with your own hands to actually learn it, where AI should
carry the load instead, resources to learn from, and how the phase connects back to your
actual research question — *how predictable is strength progression from competition
data alone?*

**The rule of thumb used throughout:**

| Do it yourself | Let AI do it |
|---|---|
| Any math derivation (gradients, loss functions, splitting rules) | Repetitive pandas column generation, once you've written the pattern once by hand |
| The core algorithm of whatever model you're studying that phase | Boilerplate: plotting code, project scaffolding, environment setup |
| Interpreting results / deciding what they mean | Writing test scripts *after you specify what they should check* |
| Debugging your own logic errors (AI as a question-asker, not an answer key) | Explaining a stack trace, looking up a library function's arguments |
| Choosing evaluation metrics and why | Generating small synthetic datasets to validate your code against sklearn |

Every phase ends with **"Self-check before moving on"** — if you can't answer these,
don't advance yet, even if the code runs without errors. Code running is not the same
as understanding why it works.

---

## If You Need to Brush Up First

You don't need to be a math expert, but a few basics will make this much smoother.
If any of these feel shaky, spend an hour on them before Phase 2 — it'll save you many
hours later:

- **Basic algebra & what a "linear equation" is** — if `y = mx + b` isn't automatic,
  [Khan Academy: Algebra basics](https://www.khanacademy.org/math/algebra-basics) (free).
- **What a derivative is** (the slope of a curve at a point) —
  [Khan Academy: Derivatives intro](https://www.khanacademy.org/math/differential-calculus)
  or 3Blue1Brown's ["Essence of Calculus"](https://www.3blue1brown.com/topics/calculus)
  video series (much more visual/intuitive, highly recommended over Khan Academy if you
  like a visual style).
- **What a matrix/vector is, and matrix multiplication** —
  [Khan Academy: Matrices](https://www.khanacademy.org/math/algebra-home/alg-matrices),
  or 3Blue1Brown's ["Essence of Linear Algebra"](https://www.3blue1brown.com/topics/linear-algebra)
  (again, the more intuitive/visual option).
- **Comfort with pandas** (you're already using it in `src/data.py`, so you likely have
  enough — but if `.groupby()` or boolean masking feels unfamiliar,
  [pandas' own "10 minutes to pandas"](https://pandas.pydata.org/docs/user_guide/10min.html)
  is a fast refresher).

You do **not** need to master these first — you can learn them just-in-time as each
phase needs them. This section is just here so you know where to look when something
feels unfamiliar, instead of guessing.

---

## Quick Glossary (terms that show up again and again below)

- **Feature** — one input column your model uses to predict, e.g. `age`, `bodyweight`.
- **Target / label** — the thing you're trying to predict, e.g. Δtotal (change in total).
- **Loss function / cost function** — a single number measuring "how wrong" your model's
  predictions are. Training = trying to make this number smaller.
- **Gradient** — the slope/direction that tells you which way to nudge your model's
  numbers to make the loss function smaller. "Gradient descent" = repeatedly nudging in
  that direction.
- **Hyperparameter** — a setting you choose before training (like learning rate, or tree
  depth) as opposed to something the model learns on its own.
- **Closed-form solution** — a formula you can plug numbers into directly to get the
  exact answer, no trial-and-error looping required (as opposed to gradient descent,
  which iterates toward an answer).
- **Overfitting** — a model that memorized quirks of the training data instead of
  learning the real pattern, so it does great on data it's seen and poorly on new data.
- **Regularization** — a penalty added during training that discourages the model from
  becoming too complex/large, specifically to fight overfitting.
- **Leakage** — accidentally letting information "from the future" (relative to the
  thing you're predicting) sneak into your features or your train/test split.
- **Residual** — the leftover error for one prediction: `actual − predicted`.
- **Cross-validation (CV)** — repeatedly splitting your data into train/test in
  different ways and averaging the results, to get a more trustworthy performance
  estimate than a single split gives you.
- **Ensemble** — a model made of many smaller models combined (e.g., Random Forest =
  many decision trees combined).
- **Bootstrap** — a statistical trick: resample your existing data (with replacement)
  many times to see how much your answer would wobble if you'd collected slightly
  different data. Used to estimate uncertainty.

---

## How to Use AI Well in This Project

AI can always produce a correct-*looking* answer, which makes it easy to feel like you
understand something you don't. Guard against that specifically.

**Prompts that help you learn:**
- "Explain the gradient of MSE loss step by step, but don't write code — I want to
  derive it on paper first and check my work against your steps."
- "Here's my from-scratch linear regression. Without fixing it, tell me which function
  most likely has the bug, given that gradient descent isn't converging."
- "Quiz me with 5 questions on bias-variance tradeoff before I move to random forests."
- "Generate a tiny synthetic dataset with a known linear relationship so I can check my
  normal-equation code recovers the true coefficients."

**Prompts that skip the learning (avoid these):**
- "Write me a linear regression from scratch." (the whole point of this plan is that
  *you* do this)
- "Why isn't my accuracy good, fix it." (teaches you nothing about *why*)
- Pasting an error and asking for a fix before you've formed your own hypothesis first.

**Where it's completely fine to just let AI do the work, no guilt attached:** the pandas
cleaning boilerplate already in `src/data.py`, repetitive feature columns once the
*pattern* is something you designed (write `rolling_avg_change_last_2_meets` yourself,
then let AI generate the `last_3_meets`/`last_4_meets` variants), plotting code,
docstrings/README polish, and environment/dependency setup. None of that teaches you ML —
it's plumbing, and offloading it is exactly what "save repeatable work" means.

---

## Project Structure & Pipeline Conventions

Read this once before Phase 1 — every phase below tells you exactly which file to add
code to, and it all fits into one growing structure:

```
predict-strength/
├── data/
│   └── openpowerlifting.csv
├── scripts/                  # exploration, plots, "does this match sklearn" checks
│   ├── 00_explore_pairs.py
│   ├── 01_baseline.py
│   ├── 02_linear_regression.py
│   ├── 04_ridge.py
│   ├── 05_trees.py
│   ├── 06_boosting.py
│   ├── 07_cv_tuning.py
│   ├── 08_evaluation.py
│   └── 09_neural_net.py
├── outputs/                  # plots saved here (e.g. plt.savefig(...)), gitignored
├── src/
│   ├── __init__.py
│   ├── data.py                 # (exists) raw CSV -> cleaned dataframe
│   ├── pairs.py                # Phase 0 — build meet-t -> meet-t+1 rows, grouped split
│   ├── metrics.py              # Phase 1 — MAE, bootstrap CI
│   ├── features.py             # Phase 3 — feature engineering
│   ├── splits.py               # Phase 7 — grouped K-fold
│   ├── evaluate.py             # Phase 8 — subgroup analysis
│   ├── pipeline.py             # Phase 10 — runs the whole thing end to end
│   └── models/
│       ├── __init__.py
│       ├── baseline.py         # Phase 1
│       ├── linear_regression.py# Phase 2
│       ├── ridge.py            # Phase 4
│       ├── tree.py             # Phase 5
│       ├── boosting.py         # Phase 6
│       └── neural_net.py       # Phase 9 (optional)
├── tests/
│   ├── test_linear_regression.py
│   ├── test_ridge.py
│   ├── test_tree.py
│   └── test_boosting.py
├── pyproject.toml
└── README.md
```

**Two conventions worth adopting from the start, both because they teach good habits,
not just because they're tidy:**

1. **Scripts (plain `.py` files) are for exploring and plotting; `src/` is for anything
   you'll reuse.** When you're sweeping a hyperparameter and eyeballing a plot, do it in
   a short script under `scripts/` — it's fine for that code to be messy and disposable,
   and fine to re-run it from scratch each time (loading the CSV takes a few seconds,
   not long enough to matter). The moment you've written a function you'll call again
   from a different phase (a model class, a metric, a feature), it belongs in `src/`,
   imported into the script, not copy-pasted between scripts. This is the actual reason
   production ML code and research code look different — reuse is the dividing line,
   not "cleanliness." (No Jupyter/notebooks anywhere in this plan — see the callout
   right after this list for why, and what you're trading away by skipping them.)
2. **Give every model the same interface: a class with `.fit(X, y)` and `.predict(X)`**,
   mirroring how sklearn's own models work. It costs nothing extra to write it this way,
   and it means your Phase 7/8 evaluation code (grouped CV, subgroup MAE) can be written
   *once*, against that interface, and reused unchanged for every model you build from
   Phase 2 onward — you're not rewriting the evaluation loop five times.

### A note on notebooks (Jupyter) — and why this plan doesn't use them

Short answer: **no, you don't need them, and this plan is written entirely around plain
`.py` scripts instead.** Every "run this and look at the output" step below means
"run `python scripts/whatever_it_is.py` in your terminal," full stop — nothing here
requires installing or learning Jupyter. (Running the file by its path like this — not
`python -m` — works fine even though these filenames start with numbers, and your
`from src... import ...` lines will resolve correctly as long as you've run
`pip install -e .` from Phase 0, step 2.)

Longer answer, for context (skip this if you don't care why): a notebook (a `.ipynb`
file, opened via `jupyter lab` or an editor's notebook mode) lets you write code in
numbered "cells" and run them one at a time, keeping variables in memory between runs —
so you load your 3.9M-row CSV once, then re-run just the plotting cell twenty times while
you tweak it, instead of reloading the CSV every time. Plots also render directly under
the cell that made them. That's genuinely convenient for fast, throwaway iteration, and
it's the standard tool most working ML practitioners reach for during this kind of
exploration — which is the only reason it showed up in this plan in the first place.

The tradeoff you're accepting by skipping it: your plain scripts will re-run the full
load-and-clean step every time (a few seconds — not the 3.9M full CSV parse repeated
uselessly, since `load_clean_data()` is fast enough that this is a minor annoyance, not
a real cost), and instead of a plot appearing inline, you'll save it to a file and open
it yourself: `plt.savefig("outputs/gradient_descent_cost.png")`, then open that PNG in
your file viewer (or `plt.show()` if you'd rather have a window pop up while the script
runs — either works, `savefig` just leaves you a permanent copy to compare against later
runs). Neither of these costs is significant for a project this size — it's a genuine,
reasonable trade in exchange for one less tool to learn while you're focused on ML
concepts. If you ever find yourself wishing you could tweak-and-rerun a plot faster
without reloading data, that's the moment notebooks start paying for themselves — worth
revisiting then, not before.

Each phase below has a **"Where the code goes"** section telling you exactly which of
these files to touch, so you don't need to re-derive this structure each time.

---

## Phase 0 — Environment & Data Discipline

**Where this fits:** everything downstream depends on your train/test split being
trustworthy. Get this wrong and every later "my model works!" result is a mirage.

**The concept behind this phase, up front:** your dataset isn't a pile of independent
rows — it's *sequences per lifter* (meet 1 → meet 2 → meet 3...). The usual "randomly
shuffle and split" approach leaks information here: if a lifter's meet-3→meet-4 pair
ends up in training and their meet-4→meet-5 pair ends up in test, the model has
effectively already seen that lifter's "future" during training. Everything below exists
to prevent that.

### Step-by-step

1. **Write a one-paragraph note first, before any code** (a comment at the top of a
   scratch file is fine) explaining in your own words why splitting by *row* rather
   than by *lifter* would leak information here. You genuinely need this reasoning
   solid before step 8 below, and you'll need it again in Phase 7.
2. **Fill in `pyproject.toml`'s dependencies**: `pandas`, `numpy`, `scikit-learn`,
   `matplotlib`, `xgboost`. Fine to have AI just write this line for you — it's not a
   learning task. Then install: `pip install -e .` from the project root.
3. **Turn `src/data.py`'s script body into a function.** Right now it's a script that
   prints things as it runs; wrap the same logic into
   `def load_clean_data() -> pd.DataFrame: ... return df` so other files can
   `from src.data import load_clean_data` and get the cleaned dataframe back, instead
   of copy-pasting the cleaning steps everywhere you need them.
4. **Create a new file `src/pairs.py`.** This is where the next two functions go.
5. **Write `build_meet_pairs(df)` — turns one-row-per-meet into one-row-per-transition.**
   Right now each row is "one lifter at one meet." You want each row to become "one
   lifter's transition from meet t to meet t+1": meet t's info stays in the row, plus a
   new column holding meet t+1's total (the thing you're trying to predict). Do it in
   this order:
   - Sort chronologically within each lifter, so "the next row" really does mean "the
     next meet": `df = df.sort_values(['Name', 'Date'])`.
   - Group by lifter: `grouped = df.groupby('Name')`.
   - Pull each lifter's *next* meet's total into the *current* row:
     `df['next_TotalKg'] = grouped['TotalKg'].shift(-1)`. What `.shift(-1)` actually
     does: it looks one row ahead *within each lifter's group* and copies that value
     backward — so on a lifter's 2nd meet, `next_TotalKg` becomes their 3rd meet's
     total; on their 3rd meet, it becomes their 4th meet's total.
   - Drop rows that don't have a next meet to point to: every lifter's *last* recorded
     meet will have `next_TotalKg = NaN`, since there's nothing after it to shift in.
     Drop those: `df = df.dropna(subset=['next_TotalKg'])`.
   - Return `df`. That's the whole function — what's left over after the drop *is* your
     pairs dataframe.
6. **Verify step 5 actually worked before trusting it.** In a new file
   `scripts/00_explore_pairs.py`: load and clean the data, call `build_meet_pairs`, then
   pick one lifter with 3+ meets and print their rows from *before* and *after* the
   transformation side by side. Confirm by eye: row 1's `next_TotalKg` should equal row
   2's `TotalKg`; row 2's `next_TotalKg` should equal row 3's `TotalKg`; and so on. Don't
   move to step 7 until this looks right for at least one real lifter.
7. **Back in `src/pairs.py`, write
   `train_test_split_by_lifter(pairs_df, test_size=0.2, random_state=42)`.** Plain
   `sklearn.model_selection.train_test_split` shuffles individual rows randomly — the
   exact leakage problem from step 1. `GroupShuffleSplit` does the same job but
   guarantees every row belonging to the same lifter lands entirely on one side. Its API
   looks a little different the first time you see it — it hands back index
   *positions* to slice with, not dataframes directly:
   - `from sklearn.model_selection import GroupShuffleSplit`
   - `splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`
   - Call `splitter.split(pairs_df, groups=pairs_df['Name'])` — the `groups` argument is
     what tells it "these rows belong together, keep them on the same side."
   - `.split()` returns a *generator*, not indices directly (it supports producing
     multiple different splits, which is what you'll use for real cross-validation in
     Phase 7). Since `n_splits=1` here, pull out the one result with `next(...)`:
     `train_idx, test_idx = next(splitter.split(pairs_df, groups=pairs_df['Name']))`.
   - Slice: `train_df = pairs_df.iloc[train_idx]`, `test_df = pairs_df.iloc[test_idx]`.
   - Return `train_df, test_df`.
8. **Verify step 7 the same way, every single time you use it, from now on:**
   `assert set(train_df['Name']) & set(test_df['Name']) == set()`. If this ever fails,
   something upstream is broken — treat it as a smoke alarm, not an optional extra.
9. **Before moving to Phase 1, answer this out loud (no code needed):** if a lifter has
   5 meets (so 4 meet-t→meet-t+1 pairs), should any of those 4 pairs end up split across
   train and test? Why not? If you're not fully confident, re-read step 1's note you
   wrote — that's the exact reasoning this question is checking.

### Resources (reference material, not additional steps)
- [sklearn: `GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
  and [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
  docs — short, concrete, and you'll use these directly in Phase 7.

---

## Phase 1 — The Baseline Model

**Where this fits:** you can't claim "my model is predictive" unless you know what
"predictive" is being compared against. This number is the yardstick for every later
phase — literally the "Baseline comparison" line in `general-structure.md`'s evaluation
section.

**The concept behind this phase, up front:** before building anything fancy, build the
dumbest reasonable model and measure it. Later, "my XGBoost model has an MAE of 8kg" is
meaningless on its own — "my XGBoost model beats the naive baseline's MAE of 12kg by
33%" is a real result. Everything below builds exactly two things: that dumb model, and
the metric to measure it with.

### Step-by-step

1. **Create `src/metrics.py` with one function: `mae(actual, predicted) -> float`.**
   Build it as three explicit lines first, so each piece is visible:
   - `errors = actual - predicted` (elementwise subtraction — works directly on pandas
     Series or numpy arrays)
   - `abs_errors = np.abs(errors)`
   - `return np.mean(abs_errors)`

   Once you're comfortable with it, these three lines are just
   `np.mean(np.abs(actual - predicted))`. Every later phase imports this one function
   instead of reaching for `sklearn.metrics.mean_absolute_error`, so you always know
   exactly what number you're looking at.
2. **Decide what the baseline actually predicts, in plain words, before writing the
   class.** For a lifter you've already seen compete before: their next total = their
   current total, plus whatever they've *typically* changed by in their past meets. For
   a lifter on their very first recorded transition (no past changes to look at yet):
   fall back to the average change across everyone.
3. **Create `src/models/baseline.py`.** It needs a class `BaselineModel`, plus a small
   *module-level helper function*, `_add_delta_features(df) -> pd.DataFrame`, that steps
   4-5 below will fill in. This helper lives in `baseline.py`, not `pairs.py` — `pairs.py`
   is Phase 0's concern only (building pairs, doing the leakage-safe split), and
   `delta`/`avg_past_delta` are this specific baseline's own invention, not a property of
   "pairs" in general. `fit()` and `predict()` will each call this helper internally
   (steps 7-8), so anyone using `BaselineModel` just passes in the raw pairs dataframe
   and never needs to know these columns exist.
4. **Inside `_add_delta_features`, add a `delta` column** — the size of *this*
   transition: `df['delta'] = df['next_TotalKg'] - df['TotalKg']`. (Same quantity as
   your prediction target, just expressed as a change instead of an absolute total.)
5. **In the same function, add an `avg_past_delta` column** — for each row, the average
   of that *same lifter's* delta values from transitions strictly *before* this one
   (never including the current row's own delta — using the answer to predict itself
   would be leakage, the same mistake Phase 0 is about, just showing up inside a single
   lifter's history instead of across the train/test boundary):
   ```python
   def _add_delta_features(df):
       df = df.copy()
       df['delta'] = df['next_TotalKg'] - df['TotalKg']
       df['avg_past_delta'] = (
           df.groupby('Name')['delta']
           .transform(lambda s: s.shift(1).expanding().mean())
       )
       return df
   ```
   Two things are happening in the `avg_past_delta` line: `.shift(1)` moves each
   lifter's delta values down by one row, so row *i* sees row *i-1*'s delta instead of
   its own; `.expanding().mean()` then averages everything seen so far at each point —
   combined with the shift, that gives you "the average of this lifter's deltas
   strictly before the current one." A lifter's very first row will have nothing to
   shift in, so `avg_past_delta` is `NaN` there — expected, and exactly the case step 8
   handles. (The `df.copy()` avoids silently mutating whatever dataframe was passed in —
   worth doing any time a function modifies columns on data it didn't create itself.)
6. **Verify step 5 by hand before moving on**: call `_add_delta_features` directly on a
   small dataframe for one lifter with 3+ transitions, and manually check that their 3rd
   row's `avg_past_delta` equals the plain average of their 1st and 2nd rows' `delta`
   values.
7. **Write `BaselineModel.fit(self, train_df)`**:
   - `train_df = _add_delta_features(train_df)`
   - `self.population_avg_delta = train_df['delta'].mean()` — the one number that's
     genuinely "learned" from training data here.
8. **Write `BaselineModel.predict(self, df)`**:
   - `df = _add_delta_features(df)`
   - `predicted_delta = df['avg_past_delta'].fillna(self.population_avg_delta)` (use
     the lifter's own average where it exists, the population average where it's `NaN`)
   - `return df['TotalKg'] + predicted_delta`
9. **Wire it all together in a new file `scripts/01_baseline.py`**:
   ```python
   from src.data import load_clean_data
   from src.pairs import build_meet_pairs, train_test_split_by_lifter
   from src.models.baseline import BaselineModel
   from src.metrics import mae

   df = load_clean_data()
   pairs = build_meet_pairs(df)
   train_df, test_df = train_test_split_by_lifter(pairs)

   model = BaselineModel()
   model.fit(train_df)          # internally adds delta / avg_past_delta to train_df
   predictions = model.predict(test_df)   # internally adds them to test_df too

   print(mae(test_df['next_TotalKg'], predictions))
   ```
   Notice this script never mentions `delta` or `avg_past_delta` at all — that's the
   payoff of step 3's decision to keep that logic inside `BaselineModel`.
10. **Run it**: `python scripts/01_baseline.py` from the project root. Whatever MAE
    prints out is your yardstick number — write it down somewhere, you'll compare every
    future model against it.
  
  ### 38.45 kgs

11. **Before moving to Phase 2, answer this (no code needed):** what MAE would a
    "predict zero change" baseline get, versus your historical-average baseline? Which
    of the two is the fairer thing to compare your real model against later, and why?

Note for later: you'll see this exact `.shift(1).expanding()` (or `.shift(1).rolling()`
for a fixed window instead of "everything so far") pattern again in Phase 3 —
`prev_total_change` and `rolling_avg_change_last_2_meets` are built the same way.

### AI use here
Ask AI to explain *why* MAE (rather than MSE, which squares errors) is the better fit
for this specific problem — think about what a single lifter with a 50kg jump between
meets (possibly a data-entry error) would do to each metric. Don't ask AI to write the
baseline itself — every step above is a few lines of `pandas`/`numpy` you should type
yourself.

### Resources (reference material, not additional steps)
- ISLR (*An Introduction to Statistical Learning*, free PDF —
  [statlearning.com](https://www.statlearning.com/)), Ch. 2 — covers exactly this idea of
  measuring model quality and what a baseline buys you, in very approachable language.

---

## Phase 2 — Linear Regression From Scratch

This is the most important phase in the whole plan. Ridge regression (Phase 4), your
trend-slope feature (Phase 3), and even the leaves of a gradient-boosted tree (Phase 6)
all lean on the same idea you'll build here.

**Where this fits:** linear regression is your first *real* model of "does strength
progression follow a pattern, or is it noise" — the coefficients it learns are a direct,
readable answer to "how much does each factor (age, bodyweight change, etc.) actually
move the needle."

### Concept, in plain terms
You're fitting a straight-line-ish relationship: `predicted_Δtotal = w1*feature1 +
w2*feature2 + ... + b`. "Training" means finding the weights (`w1, w2, ...`) that make
the predictions as close as possible to the real Δtotal values, measured by a loss
function (here, squared error).

There are two different ways to find those weights, and you'll implement both:
1. **The normal equation** — a formula that jumps straight to the exact best-fit weights
   in one shot, using matrix algebra: `w = (XᵀX)⁻¹ Xᵀy`. Think of it like solving "2
   equations, 2 unknowns" from algebra class, just generalized to many features at once.
2. **Gradient descent** — instead of solving directly, you make a guess, measure how
   wrong it is, and nudge the weights a little in the direction that reduces the error
   (the gradient), repeating thousands of times until it converges.

### Build yourself, in this order
1. **On paper, before any code:** derive the gradient of the squared-error loss with
   respect to the weights. This is one application of the chain rule (the calculus rule
   for taking derivatives of "functions of functions"). Don't skip this step — typing
   out the derivative in code without having derived it on paper is the single most
   common way to fake-understand this phase.
2. Implement the **normal equation** with raw numpy. Hint structure (fill in the numpy
   calls yourself):
   - build your feature matrix `X` (don't forget a column of 1s for the bias/intercept term)
   - compute `Xᵀ X`, invert it, multiply by `Xᵀ y`
   - Validate: compare your weights against `sklearn.linear_model.LinearRegression` on
     the same toy data — they should match to several decimal places.
3. Implement **gradient descent** from scratch. Hint structure:
   - start with weights at zero (or small random values)
   - loop: compute predictions → compute the gradient (your paper derivation from step 1)
     → update weights by `weights -= learning_rate * gradient` → repeat
   - track the loss value each iteration in a list, and plot it (ask AI for the
     matplotlib boilerplate — this part isn't a learning task) — you should see it
     decrease and flatten out.
4. **Deliberately break it**: run gradient descent with one unscaled feature sitting next
   to a small-scale one (e.g., raw `days_since_last_meet`, which can be in the hundreds,
   next to `age`). Watch it converge slowly or diverge entirely. Seeing this failure
   firsthand is the fastest way to actually internalize *why* feature scaling matters,
   rather than just accepting it as a rule.

### Where the code goes
- Create `src/models/linear_regression.py` with a class `LinearRegressionScratch`
  matching the `fit(X, y)` / `predict(X)` interface from Phase 1's baseline. Give it a
  `method="normal_equation"` vs. `method="gradient_descent"` option (or two separate
  classes if that's clearer to you) so both approaches from steps 2 and 3 above live
  side by side and you can compare them directly.
- Create `tests/test_linear_regression.py` — this is where the "compare against sklearn
  on synthetic data" validation from steps 2/3 becomes a real, permanent pytest test
  (`def test_normal_equation_matches_sklearn(): ...`) instead of a one-off print
  statement you'll forget about. Every from-scratch model in this plan gets a test file
  like this — it's the automated version of "did I actually implement this correctly."
- Do the cost-curve plot and the "break it with unscaled features" experiment (step 4)
  in `scripts/02_linear_regression.py` — this is exploratory/visual (save plots with
  `plt.savefig("outputs/...")`), so it belongs in a throwaway script, not `src/`.
- In Phase 3 you'll import `LinearRegressionScratch` from this file again to build the
  trend-slope feature — this file doesn't get thrown away after this phase.

### Resources
- **Andrew Ng's "Machine Learning Specialization"** (Coursera, free to audit) — the
  clearest, most beginner-friendly walkthrough of linear regression and gradient descent
  that exists; start here, not with denser academic notes.
- ISLR, Ch. 3 — same topic from a more statistics-flavored angle (it also covers
  confidence intervals on coefficients, which will matter again in Phase 8).
- 3Blue1Brown's "Essence of Calculus" (if you skipped the brush-up section above, do it
  now — the gradient descent visualization in particular makes this phase click).
- *Optional, more advanced*: Stanford CS229 lecture notes (free PDF) cover the same
  material more rigorously/densely — good as a second pass once Ng's course makes sense,
  not as your first exposure.

### AI use here
- "Check my derivation of ∂MSE/∂w" (paste your paper work, photo is fine).
- "Generate a 20-row synthetic dataset with a known linear relationship and some noise,
  so I can confirm my normal equation recovers coefficients close to the true ones."
- Do **not** ask AI to write `fit()` / `predict()` for you — that's the entire exercise.

### Self-check
- Write the normal equation from memory, without looking it up.
- In one sentence: why does gradient descent need a learning rate, but the normal
  equation doesn't need any hyperparameter at all?
- Why does the normal equation become impractical with a very large number of features
  (hint: think about the cost of inverting a matrix as it grows)?

---

## Phase 3 — Feature Engineering

**Where this fits:** this is where your project's core hypothesis gets encoded into
data — `feature-engineering.md` is explicitly about approximating a lifter's *hidden*
training state from what little you can observe. Bad features here cap your model's
performance no matter how good later phases are.

### Concept, in plain terms
A feature is only useful if it's built purely from information you'd actually have
*before* the meet you're predicting. The recurring risk (leakage, from Phase 0) shows up
here in a sneakier form: it's easy to accidentally compute a rolling average that
includes the target meet itself.

### Build yourself
- Implement the "Minimal Viable Feature Set" from `feature-engineering.md` by hand:
  `current_total`, `bodyweight`, `age`, `prev_total_change`, `meet_number`,
  `days_since_last_meet`, `bodyweight_change`.
- Implement the **trend slope feature** (`linear_trend_slope_last_N_meets`) using your
  *own* Phase 2 linear regression code (fit a line through a lifter's last N totals over
  time, take the slope) rather than reaching for `numpy.polyfit`. This is the actual
  payoff of building linear regression by hand — you now have a real use for it.
- For at least 3 features, write a one-line comment explaining exactly why it's
  leakage-safe (what data would sneak in if you weren't careful about the meet-t
  boundary?).

### Where the code goes
Create `src/features.py` with a single function `build_features(pairs_df) -> pd.DataFrame`
that takes the output of `src/pairs.py`'s `build_meet_pairs()` and adds each engineered
column. Build it feature-by-feature — write one, sanity-check it in a quick script
(print a few rows, eyeball them), add it to the function, repeat. Import
`LinearRegressionScratch` from `src/models/linear_regression.py`
for the trend-slope feature specifically — this is the concrete link between Phase 2 and
Phase 3 the plan keeps mentioning. From here on, `src/pipeline.py` (Phase 10) will call
your functions in this order: `load_clean_data()` → `build_meet_pairs()` →
`build_features()` → `train_test_split_by_lifter()`.

### AI use here
Once you've written `rolling_avg_change_last_2_meets` by hand and understand the
`groupby().rolling()` pattern, it's completely fine to have AI generate the analogous
`last_3_meets` / `last_4_meets` variants and the fatigue/consistency features from
sections 6-7 of your doc — these are repetitive applications of a pattern you already
understand, not new concepts.

### Resources
- Your own `feature-engineering.md`, section 9 ("Feature Construction Rule") — re-read
  it after implementing 5 features; you'll likely think of an edge case it doesn't
  mention yet.

### Self-check
- Pick 3 features and explain, without checking the doc, why each is leakage-safe.
- Why is `linear_trend_slope` likely more informative than `prev_total_change` alone,
  for a lifter with 4+ meets?

---

## Phase 4 — Regularization (Ridge Regression)

**Where this fits:** once you have 15-20 engineered features but a limited number of
meets per lifter, plain linear regression will start fitting noise instead of signal —
which is exactly the noise-vs-signal question your project cares about. Regularization
is your first tool for telling the two apart.

### Concept, in plain terms
**Overfitting** happens when a model gets so flexible it starts matching quirks of the
specific training examples rather than the general pattern — great training score,
lousy score on new data. **Ridge regression** fixes this by adding a penalty for having
large weights, controlled by a knob called `λ` (lambda): bigger `λ` → smaller weights →
simpler, less overfit model, but too much `λ` underfits (the model gets too simple to
capture real patterns). Finding the right `λ` is a genuine **bias-variance tradeoff**:
too little regularization = high variance (overfits), too much = high bias (underfits).

### Build yourself
- On paper, notice that Ridge's closed-form solution is just Phase 2's normal equation
  with one term added: `w = (XᵀX + λI)⁻¹ Xᵀy`. Convince yourself why adding `λI` (λ times
  the identity matrix) shrinks the weights — it's a small, satisfying "aha" once you see
  it's a one-line change from something you already built.
- Implement Ridge from scratch; validate against `sklearn.linear_model.Ridge`.
- Run a small experiment: sweep `λ` from 0 to something large, plot train MAE and
  validation MAE at each value, and find the classic U-shaped validation curve yourself.

### Where the code goes
- Create `src/models/ridge.py` with `RidgeScratch`, same `fit(X, y)` / `predict(X)`
  interface as before (you could even have it subclass or reuse pieces of
  `LinearRegressionScratch`, since the only difference is the one added term —
  worth trying if you want the code to visibly mirror the one-line math change).
- Add `tests/test_ridge.py` comparing against `sklearn.linear_model.Ridge`, same pattern
  as `tests/test_linear_regression.py`.
- Do the λ-sweep plot in `scripts/04_ridge.py`, using `build_features()` from
  Phase 3 and the train/test split from Phase 0 as inputs.

### Resources
- ISLR, Ch. 6 — the standard, approachable treatment of Ridge/Lasso and the
  bias-variance tradeoff, with the same simple examples throughout.
- StatQuest, "Ridge vs Lasso Regression" (YouTube) — watch this *before* the math above
  for visual intuition; StatQuest is consistently the clearest "why does this work"
  explainer for exactly these topics.

### Self-check
- Sketch (even roughly, on paper) train-error and validation-error curves as `λ`
  increases, and explain what's happening in each region.
- Why would an L1 penalty (Lasso, not covered above but worth knowing about) tend to zero
  out some coefficients entirely, while Ridge's L2 penalty only shrinks them toward zero?

---

## Phase 5 — Decision Trees & Random Forest

**Where this fits:** trees let your model naturally capture "it depends" patterns
(e.g., "young lifters with short meet gaps behave differently from older lifters with
long gaps") that a single straight-line model like Phase 2/4 can't represent well. This
is a step toward matching the real structure in strength progression, if structure
exists.

### Concept, in plain terms
A **regression tree** repeatedly splits your data into two groups based on a feature
threshold (e.g., "`age < 25`?"), picking whichever split makes each resulting group as
internally similar as possible (technically: the split that reduces the *variance*
inside each group the most). It keeps splitting until it hits a stopping rule (max depth,
or too few examples left to split further), then predicts the average target value
within whichever final group ("leaf") a new example lands in.

A single deep tree overfits easily. **Random Forest** fixes this the way Phase 4 fixed
overfitting in linear regression, but with a different trick: build many trees, each on
a random subset of the data and features (this randomness is called **bagging** —
bootstrap aggregating), then average their predictions. Averaging many high-variance,
somewhat-wrong trees cancels out their individual noise and leaves the shared signal.

### Build yourself
1. Implement a **minimal regression tree from scratch** on a small toy dataset (not the
   full powerlifting data — too slow to debug against). A recursive function that:
   - tries splitting on each feature/threshold combination
   - picks whichever split most reduces variance in the two resulting groups
   - recurses on each half, stopping at a max depth or minimum group size
2. Validate: on the same toy dataset, compare your tree's predictions against
   `sklearn.tree.DecisionTreeRegressor` with the same max depth.
3. Once that works, run it (or sklearn's — being realistic that a from-scratch tree on
   3.9M rows will be painfully slow) on a small sample of your real features. Manually
   look at a few of the splits it chose — do they make physical sense (e.g., does it
   split on `days_since_last_meet` roughly where you'd expect)?
4. **Don't hand-write Random Forest itself.** Wrapping bagging + random feature
   selection around your own tree is a fine stretch goal if you're enjoying this, but
   it's completely reasonable to use `sklearn.ensemble.RandomForestRegressor` here and
   spend your effort on understanding *why* averaging many overfit trees reduces
   variance — that's a statistics insight, not a coding one, and re-implementing the
   bagging loop teaches you comparatively little once you've built one tree.

### Where the code goes
- Create `src/models/tree.py` with a class `RegressionTreeScratch(max_depth=,
  min_samples_leaf=)`, same `fit(X, y)` / `predict(X)` interface. Internally, a recursive
  helper (e.g. `_best_split(X, y)` and `_build_node(X, y, depth)`) keeps the recursion
  readable — this is a natural place to split logic into small private methods.
- Build and debug it against the toy dataset in `scripts/05_trees.py` first — full
  data is too slow for a from-scratch tree to iterate against while debugging.
- Add `tests/test_tree.py`: generate a small synthetic dataset, assert your tree's
  predictions match (or are very close to) `sklearn.tree.DecisionTreeRegressor` with
  identical `max_depth`.
- For Random Forest itself, no new `src/` file needed — just use
  `sklearn.ensemble.RandomForestRegressor` directly in `scripts/05_trees.py`, called on
  the output of `build_features()`.

### Resources
- StatQuest, "Decision Trees" and "Random Forest" video series (YouTube) — start here;
  genuinely the clearest visual explanation of splitting criteria and bagging anywhere.
- ISLR, Ch. 8 (Tree-Based Methods) — the same material once you want the written/math
  version, still approachable.

### AI use here
- "I wrote this recursive split function and it never terminates — before telling me the
  fix, ask me questions to help me find my own bug." (Missing/wrong base cases in
  recursion are extremely common here, and working through it yourself is worth more
  than being handed the fix.)
- Boilerplate for visualizing a tree (e.g., `sklearn.tree.plot_tree`) — not a learning
  task, fine to have AI wire this up.

### Self-check
- Explain "pick the split that reduces variance the most" in your own words, without
  using the word "impurity."
- Why does averaging predictions from many overfit (deep) trees reduce variance instead
  of just averaging their overfitting too? (This is the core reason Random Forest works —
  make sure you can explain the role of randomness in which data/features each tree sees.)

---

## Phase 6 — Gradient Boosting / XGBoost

This is your project's designated "primary" model (`general-structure.md`), so it's
worth the deepest implementation effort after linear regression.

**Where this fits:** this is likely your best-performing model, and the one you'll use
for the headline result — so understanding *why* it works (not just that it scores well)
matters for trusting and explaining your final conclusions.

### Concept, in plain terms
Random Forest (Phase 5) builds many trees *independently* and averages them —
this is called **bagging**. **Boosting** instead builds trees *one after another*, where
each new tree's whole job is to fix the mistakes (the **residuals** — actual minus
current prediction) that the trees before it made. Each tree's contribution gets scaled
down by a small **learning rate** before being added in, so no single tree can swing the
prediction too far. XGBoost is a highly-optimized, regularized version of this same idea.

### Build yourself
1. Implement **simplified gradient boosting from scratch**, using
   `sklearn.tree.DecisionTreeRegressor` as the small "weak learner" tree at each step
   (you already understand trees from Phase 5 — no need to re-derive tree-building; the
   new concept here is the *boosting loop* around it):
   - start with a constant prediction (the overall mean of the target)
   - fit a shallow tree to the current residuals (`actual − current_prediction`)
   - update your running prediction: `prediction += learning_rate * tree.predict(X)`
   - repeat for N rounds, recording the training error at each round
2. Plot your training error dropping across boosting rounds (AI can give you the
   plotting boilerplate) — watching this happen is far more convincing than reading
   about it.
3. Compare your final predictions/error against real `xgboost.XGBRegressor` trained with
   the same number of rounds on the same data. They won't match exactly — real XGBoost
   uses a more sophisticated update rule (it uses both the slope *and* the curvature of
   the loss — a "second-order" approximation — plus extra penalty terms on the trees) —
   but the overall error trend should land in the same ballpark.

### Where the code goes
- Create `src/models/boosting.py` with `GradientBoostingScratch(n_estimators=,
  learning_rate=, max_depth=)`, same `fit(X, y)` / `predict(X)` interface. Internally it
  builds a list of `sklearn.tree.DecisionTreeRegressor` instances (one per round) — you
  don't need `RegressionTreeScratch` from Phase 5 here, since the learning goal this
  phase is the boosting loop itself, not re-proving the tree works.
- Add `tests/test_boosting.py`: on a small dataset, assert training error strictly
  decreases round-over-round (a weak but genuinely useful correctness check for a
  boosting loop).
- Do the round-by-round error plot and the real-`xgboost.XGBRegressor` comparison in
  `scripts/06_boosting.py`.

### Resources
- StatQuest's "Gradient Boost" series (4 parts, YouTube) — do this *before* the paper
  below; it builds the intuition the paper assumes you already have.
- ISLR, Ch. 8 (same chapter as Phase 5, boosting section) — for the written explanation
  once the videos click.
- *Optional, read last*: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System"
  (2016, free on arXiv) — the actual algorithm's paper. Save this for after you have a
  working simplified version; the regularization terms and the curvature-based update
  will actually mean something to you by then, instead of being abstract notation.

### AI use here
- "Here's my boosting loop — walk through what should happen on round 1 vs. round 5, so
  I can check my mental model matches what the code is actually doing."
- Have AI scaffold a side-by-side comparison (your model's MAE vs. real XGBoost's MAE,
  plotted together) — a repeatable reporting task, not a learning one.

### Self-check
- Why does gradient boosting fit trees to *residuals* instead of the original target
  each round?
- What goes wrong if `learning_rate` is too high? Too low? (Try both and watch, rather
  than just predicting the answer.)
- In one sentence, what does real XGBoost add on top of the simplified version you built?

---

## Phase 7 — Cross-Validation Done Right

**Where this fits:** a quick but essential correctness check — every performance number
in Phases 4-6 is only trustworthy if it was measured with a split that respects the
lifter-grouping rule from Phase 0.

### Concept, in plain terms
**Cross-validation** gives you a more reliable performance estimate than a single
train/test split by repeating the split several times and averaging the results. But it
has to respect the same rule as Phase 0: a given lifter's meets must always land
entirely in one fold, never split across folds — otherwise you're back to leaking
information about a lifter across train/validation.

### Build yourself
- Implement a manual grouped-K-fold loop by hand, once: split the *lifters* (not rows)
  into K groups, then iterate holding one group out as validation each time. Doing this
  once by hand means the mechanics are never a black box to you.
- After that, switch to `sklearn.model_selection.GroupKFold` for convenience — no need
  to keep hand-rolling this once you've verified you understand it.
- Run a small grid search over your Ridge `λ` and your boosting `learning_rate` /
  number-of-rounds, using this grouped CV setup.

### Where the code goes
- Create `src/splits.py` with `grouped_kfold(pairs_df, k=5, group_col="Name")` — your
  hand-rolled version, yielding `(train_idx, val_idx)` pairs the same way sklearn's
  splitters do, so it's a drop-in stand-in while you're building it.
- Once validated, the grid search itself is a good fit for its own script
  (`scripts/07_cv_tuning.py`), looping your `RidgeScratch` and
  `GradientBoostingScratch` (both already share the `fit`/`predict` interface, so the
  same loop works for both) over `sklearn.model_selection.GroupKFold`.

### Resources
- sklearn's [Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) —
  read the "Cross-validation iterators for grouped data" section specifically; short and
  directly applicable.

### Self-check
- Why would ordinary (ungrouped) K-fold CV give you an overly optimistic error estimate
  on this dataset specifically?

---

## Phase 8 — Evaluation Deep-Dive: Answering the Actual Research Question

This is where you answer the core question from `general-structure.md`: *is strength
progression structured, or mostly noise?* Treat this as the most important phase for
the project's actual purpose, even though it involves the least new ML machinery.

**Where this fits:** everything before this phase built the tools; this phase is where
you point them at the real question and produce the project's actual deliverable.

### Concept, in plain terms
A single overall MAE number hides a lot. "Signal vs. noise" isn't something you read off
one metric — it's something you build a case for by looking at *where* your model beats
the baseline convincingly, and where that advantage shrinks or vanishes. If a subgroup's
error is barely better than the naive baseline, that's evidence the outcomes in that
subgroup are close to unpredictable from meet data alone (i.e., mostly noise, or driven
by things you can't observe, like actual training details).

### Build yourself
- Compute MAE **and** the full error distribution (plot the residuals — don't just
  report one number) **split by subgroup**: beginner vs. advanced lifters (using
  `meet_number`), and by age band.
- Implement a simple **bootstrap confidence interval** for your MAE by hand: resample
  your test set with replacement N times (e.g., 1000), recompute MAE each time, and
  report the spread (e.g., the 5th-95th percentile of those 1000 MAE values). This
  directly answers "how confident are we in this number?" — a question most beginner ML
  projects skip entirely, and a small, high-value exercise to have done by hand once.
- Build your actual signal-vs-noise argument: compare your model's MAE against the
  Phase 1 baseline's MAE, *per subgroup*. Wherever your model's advantage over the
  baseline shrinks or disappears, that's your evidence of a noise floor in that subgroup.

### Where the code goes
- Create `src/evaluate.py` with `subgroup_mae(actual, predicted, group_labels) ->
  pd.DataFrame` and `bootstrap_mae_ci(actual, predicted, n_boot=1000) -> (low, high)`.
  These get called once per model (baseline, ridge, boosting, ...) so writing them once
  here and importing them is what actually pays off the "same `fit`/`predict` interface
  everywhere" decision from the Project Structure section.
- Do the residual plots and subgroup comparison tables in
  `scripts/08_evaluation.py`, saving plots to `outputs/` and printing the subgroup
  tables to the terminal — this script's output is essentially your project's results
  section.

### Resources
- StatQuest, "Bootstrapping Main Ideas" (YouTube) — quick, intuitive intro before you
  implement it.
- *Practical Statistics for Data Scientists* (Bruce, Bruce & Gedeck) — has an
  approachable chapter on the bootstrap and resampling methods aimed at practitioners,
  not statisticians; a gentler read than a full statistics textbook.
- ISLR, Ch. 5 — covers the bootstrap alongside cross-validation, consistent with the
  rest of your reading list.

### Self-check
- If beginner lifters have low MAE and advanced lifters have high MAE, is that
  necessarily evidence of "less predictability" for advanced lifters — or could it be
  explained by something else (e.g., fewer advanced-lifter examples in your data, or
  genuinely higher variance in their true Δtotal)? Try to argue both sides before
  settling on a conclusion.

---

## Phase 9 (Optional Stretch) — Neural Network From Scratch

Not required by your project's model list — skip this if your main interest is
finishing the powerlifting analysis. Do it if you specifically want deeper intuition for
how gradients flow through more complex models than Phase 2's straight line.

### Concept, in plain terms
A neural network is, at its simplest, several linear-regression-style layers stacked
with a nonlinear function in between (without that nonlinearity, stacking linear layers
just collapses back into one big linear model — see the self-check below).
**Backpropagation** is just the chain rule (same one from Phase 2) applied repeatedly,
layer by layer, to figure out how much each weight in every layer contributed to the
final error.

### Build yourself
- A single-hidden-layer network in raw numpy: forward pass, MSE loss, manual backprop
  through one hidden layer, gradient descent update — same loop shape as Phase 2's
  gradient descent, just with one more layer of chain rule.
- Compare against a tiny `keras` or `pytorch` model with the same architecture, and check
  that the losses decrease similarly.

### Where the code goes
Create `src/models/neural_net.py` with the same `fit(X, y)` / `predict(X)` interface as
every other model. Keep the `keras`/`pytorch` comparison model in
`scripts/09_neural_net.py` rather than `src/` — it's a one-off validation check,
not something the pipeline needs to call.

### Resources
- Michael Nielsen, *Neural Networks and Deep Learning* (free online book,
  [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)) — the
  best from-scratch derivation available anywhere; builds exactly this kind of network
  in raw Python.
- 3Blue1Brown's neural network series — intuition companion to Nielsen's book, watch
  alongside it.
- *Optional, more advanced*: Andrej Karpathy's "micrograd" video/repo — builds
  backpropagation from a tiny scalar autograd engine up. Great once the above two clicks
  and you want the "computational graph" mental model; not the easiest starting point.

### Self-check
- Derive backpropagation through one hidden layer on paper before writing any code.
- Why does a network with only linear activations collapse to being equivalent to plain
  linear regression, no matter how many layers you stack? (This ties directly back to
  Phase 2 — the nonlinearity is what actually buys you anything.)

---

## Phase 10 — Final Analysis & Writeup

**Where this fits:** this is the deliverable named at the bottom of
`general-structure.md` — a model *and* an analysis answering the core question.

### Build yourself
- Write the final answer to the research question in `README.md`, backed by your
  Phase 8 subgroup and bootstrap results — not just a vibe, an actual number-backed claim.
- Include: baseline MAE vs. best model MAE, the per-subgroup breakdown, and an honest,
  specific statement of where the noise floor seems to sit and why you believe that.

### Where the code goes
Create `src/pipeline.py` — a single script that chains every module you've built into
one runnable end-to-end pass, roughly:

```python
df = load_clean_data()                      # src/data.py
pairs = build_meet_pairs(df)                # src/pairs.py
features = build_features(pairs)            # src/features.py
train_df, test_df = train_test_split_by_lifter(features)   # src/pairs.py

model = GradientBoostingScratch(...)        # or whichever model won in Phase 7's CV
model.fit(train_df[feature_cols], train_df["target"])
predictions = model.predict(test_df[feature_cols])

print("MAE:", mae(test_df["target"], predictions))              # src/metrics.py
print(subgroup_mae(test_df["target"], predictions, groups))     # src/evaluate.py
print(bootstrap_mae_ci(test_df["target"], predictions))         # src/evaluate.py
```

Runnable as `python -m src.pipeline`. This is the moment all ten phases visibly become
one program instead of a folder of separate exercises — and it's the script whose printed
output you're transcribing into the `README.md` writeup.

### AI use here
Entirely fine to have AI help copyedit/structure the writeup once you've drafted the
substance yourself — that's a communication task, not a learning task.

---

## Suggested Pacing (loose — adjust freely)

| Phase | Focus | Rough time (evenings/weekends) |
|---|---|---|
| 0 | Setup + leakage reasoning | 1 session |
| 1 | Baseline + metrics from scratch | 1 session |
| 2 | Linear regression from scratch | 3-4 sessions (the anchor phase — don't rush it) |
| 3 | Feature engineering | 2-3 sessions |
| 4 | Ridge regression | 1-2 sessions |
| 5 | Trees + Random Forest | 3 sessions |
| 6 | Gradient boosting + XGBoost | 3-4 sessions |
| 7 | Grouped cross-validation | 1 session |
| 8 | Evaluation / signal-vs-noise | 2-3 sessions |
| 9 | Neural net (optional) | 3+ sessions |
| 10 | Writeup | 1 session |

Don't treat this as a schedule to hit — treat it as a sequencing guarantee. Each phase's
math leans on the previous one: Ridge is Phase 2 plus one regularizing term; the
trend-slope feature reuses Phase 2's code directly; gradient boosting reuses Phase 5's
trees. Slow down wherever a self-check question exposes a gap — that's exactly the
moment AI-as-tutor (quizzing you, checking a derivation, pointing at *where* a bug likely
lives) is most valuable, and AI-as-autocomplete is most tempting and least useful.
