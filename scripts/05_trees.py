import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.metrics import mae
from src.models.tree import RegressionTreeScratch
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features

rng = np.random.RandomState(0)

# small synthetic dataset - piecewise-constant-ish so a shallow tree can
# actually capture the pattern, easier to sanity check against sklearn
n_samples = 200
X = rng.uniform(0, 10, size=(n_samples, 2))
y = np.where(X[:, 0] < 5, 10, 50) + np.where(X[:, 1] < 5, 0, 20) + rng.normal(0, 0.5, size=n_samples)

max_depth = 3
min_samples_leaf = 5

model = RegressionTreeScratch(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
model.fit(X, y)
predictions = model.predict(X)

sklearn_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)

print("scratch MAE:", mae(y, predictions))
print("sklearn MAE:", mae(y, sklearn_predictions))
print("predictions close:", np.allclose(predictions, sklearn_predictions, atol=1.0))
print("scratch predictions (first 10):", predictions[:10])
print("sklearn predictions (first 10):", sklearn_predictions[:10])

# --- Random Forest on real project data ---
# no scratch implementation - plan explicitly says use sklearn's RandomForestRegressor
# directly here, since re-implementing bagging teaches little once you've built one tree

df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])
pairs = build_meet_pairs(df)
pairs = build_features(pairs)

train_df, test_df = train_test_split_by_lifter(pairs)

FEATURE_COLS = [
    'TotalKg', 'BodyweightKg', 'Age', 'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change', 'rolling_avg_change_last_3_meets',
    'linear_trend_slope_last_N_meets',
]

train_df = train_df.dropna(subset=FEATURE_COLS + ['next_TotalKg'])
test_df = test_df.dropna(subset=FEATURE_COLS + ['next_TotalKg'])

X_train = train_df[FEATURE_COLS].to_numpy()
y_train = train_df['next_TotalKg'].to_numpy()
X_test = test_df[FEATURE_COLS].to_numpy()
y_test = test_df['next_TotalKg'].to_numpy()

forest = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)

print()
print("Random Forest MAE (real data):", mae(y_test, forest_predictions))
