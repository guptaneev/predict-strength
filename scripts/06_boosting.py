import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from src.metrics import mae
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features
from src.models.boosting import GradientBoostingScratch

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

n_estimators = 100
learning_rate = 0.1
max_depth = 3

model = GradientBoostingScratch(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

print("scratch gradient boosting MAE:", mae(y_test, predictions))
print("xgboost MAE:", mae(y_test, xgb_predictions))

plt.plot(model.error_history)
plt.xlabel("boosting round")
plt.ylabel("training error (MSE)")
plt.title("Gradient boosting training error over rounds")
plt.savefig("outputs/boosting_error_history.png")
