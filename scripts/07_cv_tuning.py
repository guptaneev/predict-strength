import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.metrics import mae
from src.pairs import build_meet_pairs
from src.features import build_features
from src.models.ridge import RidgeScratch
from src.models.boosting import GradientBoostingScratch

df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])
pairs = build_meet_pairs(df)
pairs = build_features(pairs)

FEATURE_COLS = [
    'TotalKg', 'BodyweightKg', 'Age', 'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change', 'rolling_avg_change_last_3_meets',
    'linear_trend_slope_last_N_meets',
]

pairs = pairs.dropna(subset=FEATURE_COLS + ['next_TotalKg'])

X = pairs[FEATURE_COLS].to_numpy()
y = pairs['next_TotalKg'].to_numpy()
groups = pairs['Name'].to_numpy()

K = 5
gkf = GroupKFold(n_splits=K)


def cross_val_mae(model_factory, needs_scaling):
    """Average validation MAE for one hyperparameter setting, across all K folds."""
    fold_maes = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if needs_scaling:
            mean, std = X_train.mean(axis=0), X_train.std(axis=0)
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std

        model = model_factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        fold_maes.append(mae(y_val, predictions))

    return np.mean(fold_maes)


# --- Ridge: sweep lambda ---
print("Ridge lambda sweep:")
ridge_lambdas = [0.1, 1, 10, 100]
ridge_results = {}
for lam in ridge_lambdas:
    cv_mae = cross_val_mae(lambda lam=lam: RidgeScratch(alpha=lam), needs_scaling=True)
    ridge_results[lam] = cv_mae
    print(f"  lambda={lam}: CV MAE={cv_mae:.4f}")

best_lambda = min(ridge_results, key=lambda k: ridge_results[k])
print(f"best lambda: {best_lambda} (CV MAE={ridge_results[best_lambda]:.4f})")

# --- Gradient Boosting: sweep learning_rate x n_estimators ---
print("\nGradient boosting grid search:")
learning_rates = [0.05, 0.1, 0.2]
n_estimators_options = [50, 100]
gb_results = {}
for lr, n_est in itertools.product(learning_rates, n_estimators_options):
    cv_mae = cross_val_mae(
        lambda lr=lr, n_est=n_est: GradientBoostingScratch(n_estimators=n_est, learning_rate=lr, max_depth=3),
        needs_scaling=False,
    )
    gb_results[(lr, n_est)] = cv_mae
    print(f"  learning_rate={lr}, n_estimators={n_est}: CV MAE={cv_mae:.4f}")

best_gb_params = min(gb_results, key=lambda k: gb_results[k])
print(f"best (learning_rate, n_estimators): {best_gb_params} (CV MAE={gb_results[best_gb_params]:.4f})")
