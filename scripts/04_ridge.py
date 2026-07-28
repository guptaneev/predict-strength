import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from src.metrics import mae
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features
from src.models.ridge import RidgeScratch

# use the already-cleaned CSV directly instead of re-running load_clean_data()
df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])

pairs = build_meet_pairs(df)
pairs = build_features(pairs)

train_df, test_df = train_test_split_by_lifter(pairs)

# fill missing avg_past_delta (first transition per lifter) using train-only stats
population_avg_delta = train_df['delta'].mean()
train_df = train_df.copy()
test_df = test_df.copy()
train_df['avg_past_delta'] = train_df['avg_past_delta'].fillna(population_avg_delta)
test_df['avg_past_delta'] = test_df['avg_past_delta'].fillna(population_avg_delta)

FEATURE_COLS = [
    'TotalKg', 'BodyweightKg', 'Age', 'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change', 'rolling_avg_change_last_3_meets',
    'linear_trend_slope_last_N_meets',
]

# drop rows missing any of these engineered features (e.g. not enough history yet)
train_df = train_df.dropna(subset=FEATURE_COLS + ['next_TotalKg'])
test_df = test_df.dropna(subset=FEATURE_COLS + ['next_TotalKg'])

X_train = train_df[FEATURE_COLS].to_numpy()
y_train = train_df['next_TotalKg'].to_numpy()
X_test = test_df[FEATURE_COLS].to_numpy()
y_test = test_df['next_TotalKg'].to_numpy()

# standardize (train-only mean/std, reused for test) - required for Ridge, penalty is scale-sensitive
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

alpha = 1.0

model = RidgeScratch(alpha=alpha)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

sklearn_model = Ridge(alpha=alpha)
sklearn_model.fit(X_train_scaled, y_train)
sklearn_predictions = sklearn_model.predict(X_test_scaled)

print("scratch weights:", model.weights)
print("sklearn weights:", sklearn_model.coef_, "bias:", sklearn_model.intercept_)
print("weights close:", np.allclose(model.weights[:-1], sklearn_model.coef_, atol=1e-4)
      and np.isclose(model.weights[-1], sklearn_model.intercept_, atol=1e-4))
print("scratch MAE:", mae(y_test, predictions))
print("sklearn MAE:", mae(y_test, sklearn_predictions))

# lambda sweep
lambdas = [0, 0.1, 1, 10, 100, 1000, 10000]
train_maes = []
val_maes = []

for lam in lambdas:
    sweep_model = RidgeScratch(alpha=lam)
    sweep_model.fit(X_train_scaled, y_train)
    train_maes.append(mae(y_train, sweep_model.predict(X_train_scaled)))
    val_maes.append(mae(y_test, sweep_model.predict(X_test_scaled)))

best_lambda = lambdas[int(np.argmin(val_maes))]
print("lambda sweep - train MAE:", train_maes)
print("lambda sweep - val MAE:", val_maes)
print("best lambda (min val MAE):", best_lambda)

plt.figure()
plt.plot(lambdas, train_maes, marker='o', label='train MAE')
plt.plot(lambdas, val_maes, marker='o', label='validation MAE')
plt.xscale('symlog')
plt.xlabel('lambda (alpha)')
plt.ylabel('MAE')
plt.title('Ridge: train vs validation MAE across lambda')
plt.legend()
plt.savefig('outputs/ridge_lambda_sweep.png')
