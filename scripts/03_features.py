import pandas as pd

from src.pairs import build_meet_pairs
from src.features import build_features

# use the already-cleaned CSV directly instead of re-running load_clean_data()
df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])

pairs = build_meet_pairs(df)
features = build_features(pairs)

# pick the lifter with the most meet-t -> meet-t+1 pairs (i.e. the longest history)
# to get a good long sequence to eyeball
pair_counts = features.groupby('Name').size()
lifter_name = pair_counts.idxmax()
print(f"Testing lifter: {lifter_name} ({pair_counts.max()} pairs, {pair_counts.max() + 1} meets)")

cols = [
    'Name', 'Date', 'TotalKg', 'BodyweightKg', 'Age',
    'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change',
    'rolling_avg_change_last_2_meets',
    'rolling_avg_change_last_3_meets',
    'rolling_avg_change_last_4_meets',
    'delta', 'avg_past_delta', 'next_TotalKg',
]
lifter_rows = features.loc[features['Name'] == lifter_name, cols]

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
print(lifter_rows.to_string(index=False))
