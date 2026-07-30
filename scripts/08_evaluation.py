import pandas as pd
import matplotlib.pyplot as plt

from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features
from src.models.boosting import GradientBoostingScratch
from src.evaluate import subgroup_mae, bootstrap_mae_ci

df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])
pairs = build_meet_pairs(df)
pairs = build_features(pairs)

FEATURE_COLS = [
    'TotalKg', 'BodyweightKg', 'Age', 'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change', 'rolling_avg_change_last_3_meets',
    'linear_trend_slope_last_N_meets',
    'missed_attempts_last_meet', 'opener_to_best_pct_gain', 'has_attempt_data',
    'is_tested',
    'equipment_Raw', 'equipment_Single-ply', 'equipment_Unlimited', 'equipment_Wraps',
    'equipment_changed',
    'place_last_meet', 'dots_percentile_in_meet',
]

pairs = pairs.dropna(subset=FEATURE_COLS + ['next_TotalKg'])

train_df, test_df = train_test_split_by_lifter(pairs)

X_train = train_df[FEATURE_COLS].to_numpy()
y_train = train_df['next_TotalKg'].to_numpy()
X_test = test_df[FEATURE_COLS].to_numpy()
y_test = test_df['next_TotalKg'].to_numpy()

# best combo found in scripts/07_cv_tuning.py's grid search
model = GradientBoostingScratch(n_estimators=100, learning_rate=0.2, max_depth=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

residuals = y_test - predictions

# --- beginner vs. advanced, from meet_number ---
# note: rolling_avg_change_last_3_meets / linear_trend_slope_last_N_meets require enough
# history to be non-NaN, so after dropna every surviving row already has meet_number >= 5 -
# true beginners (1-4 meets) are unobservable to this feature set. Splitting at the
# post-filter median (8) instead of an absolute "early career" cutoff.
beginner_advanced = pd.Series(
    ['beginner' if n <= 8 else 'advanced' for n in test_df['meet_number']],
    index=test_df.index,
)

experience_results = subgroup_mae(y_test, predictions, beginner_advanced.to_numpy())
print("Beginner vs. advanced MAE:")
print(experience_results)

# --- age bands ---
age_bins = [0, 20, 30, 40, 200]
age_labels = ['<20', '20-29', '30-39', '40+']
age_bands = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels, right=False)

age_results = subgroup_mae(y_test, predictions, age_bands.to_numpy())
print("\nAge band MAE:")
print(age_results)

# --- residual histograms per subgroup ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for group_value in experience_results['group']:
    mask = (beginner_advanced == group_value).to_numpy()
    axes[0].hist(residuals[mask], bins=50, alpha=0.5, label=group_value, density=True)
axes[0].set_title("Residuals: beginner vs. advanced")
axes[0].set_xlabel("actual - predicted")
axes[0].legend()

for group_value in age_labels:
    mask = (age_bands == group_value).to_numpy()
    axes[1].hist(residuals[mask], bins=50, alpha=0.5, label=group_value, density=True)
axes[1].set_title("Residuals: age band")
axes[1].set_xlabel("actual - predicted")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/subgroup_residuals.png")

# --- bootstrap confidence interval on overall test MAE ---
low, high = bootstrap_mae_ci(y_test, predictions, n_boot=1000)
print(f"\nBootstrap 90% CI for MAE: [{low:.4f}, {high:.4f}]")
