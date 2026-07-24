from src.data import load_clean_data
from src.pairs import build_meet_pairs, train_test_split_by_lifter


cleaned = load_clean_data()
cleaned_test = cleaned[cleaned['Name'] == 'Neev Gupta']
print(cleaned_test[['Name', 'TotalKg']])
paired = build_meet_pairs(cleaned)
paired_test = paired[paired['Name'] == 'Neev Gupta']
print(paired_test[['Name', 'next_TotalKg']])

train_df, test_df = train_test_split_by_lifter(paired)

# check if any overlapping data in training and testing 
assert set(train_df['Name']) & set(test_df['Name']) == set()

