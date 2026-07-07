# Strength Predictability Project — General Structure

## Core Research Question
How predictable is strength progression when we only observe competition (meet) data, without training logs?

## Key Idea
We are not modeling training. We are modeling:
> Statistical patterns in strength outcomes over time.

---

## Problem Framing

### Input
A lifter at meet t:
- age
- bodyweight
- previous performance history
- meet spacing
- strength trajectory

### Output
- Δ Total (change in total from meet t → t+1)

---

## Dataset Construction

We transform OpenPowerlifting into:
> Lifter-level time transitions

Each row becomes:
(Meet t) → (Meet t+1)

So instead of independent records, we model **sequences of performance**.

---

## Modeling Approach

### Baseline Model
- Predict next improvement using average historical change

### Machine Learning Models
- Linear Regression
- Random Forest
- XGBoost (primary)
- Optional: Neural Network

---

## Evaluation

We evaluate not just accuracy, but:

- Mean Absolute Error (MAE)
- Baseline comparison
- Error distribution
- Predictability by subgroup

---

## Core Insight Goal

We are trying to measure:

> Is strength progression structured or mostly noise?

We quantify:
- signal vs noise ratio
- uncertainty in predictions
- how predictability changes across lifter experience

---

## Expected Outcome

We will likely observe:
- beginners = more predictable
- advanced lifters = less predictable
- diminishing returns in model accuracy

---

## Final Deliverable

A model + analysis answering:

> How predictable is human strength progression from competition data alone?