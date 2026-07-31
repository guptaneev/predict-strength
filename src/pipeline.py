import pandas as pd

from src.data import load_clean_data
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features
from src.metrics import mae
from src.evaluate import subgroup_mae, bootstrap_mae_ci
from src.models.boosting import GradientBoostingScratch

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

# best combo found in scripts/07_cv_tuning.py's grid search
MODEL_PARAMS = dict(n_estimators=100, learning_rate=0.2, max_depth=3)


def run():
    df = load_clean_data()
    pairs = build_meet_pairs(df)
    pairs = build_features(pairs)
    pairs = pairs.dropna(subset=FEATURE_COLS + ['next_TotalKg'])

    train_df, test_df = train_test_split_by_lifter(pairs)

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df['next_TotalKg'].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df['next_TotalKg'].to_numpy()

    model = GradientBoostingScratch(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Overall MAE:", mae(y_test, predictions))

    beginner_advanced = pd.Series(
        ['beginner' if n <= 8 else 'advanced' for n in test_df['meet_number']],
        index=test_df.index,
    )
    print("\nBeginner vs. advanced MAE:")
    print(subgroup_mae(y_test, predictions, beginner_advanced.to_numpy()))

    age_bins = [0, 20, 30, 40, 200]
    age_labels = ['<20', '20-29', '30-39', '40+']
    age_bands = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels, right=False)
    print("\nAge band MAE:")
    print(subgroup_mae(y_test, predictions, age_bands.to_numpy()))

    low, high = bootstrap_mae_ci(y_test, predictions, n_boot=1000)
    print(f"\nBootstrap 90% CI for MAE: [{low:.4f}, {high:.4f}]")


if __name__ == '__main__':
    run()
