from src.data import load_clean_data
from sklearn.model_selection import GroupShuffleSplit


def build_meet_pairs(df):
    df = df.copy()
    df = df.sort_values(['Name', 'Date'])
    grouped = df.groupby('Name')
    df['next_TotalKg'] = grouped['TotalKg'].shift(-1)
    df = df.dropna(subset=['next_TotalKg'])
    
    # copied from BaselineModel (baseline.py)
    # adds running average past delta field
    df['delta'] = df['next_TotalKg'] - df['TotalKg']
    df['avg_past_delta'] = (
        df.groupby('Name')['delta']
        .transform(lambda s: s.shift(1).expanding().mean())
    )
            
    return df

def train_test_split_by_lifter(pairs_df, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(pairs_df, groups=pairs_df['Name']))
    train_df = pairs_df.iloc[train_idx]
    test_df = pairs_df.iloc[test_idx]
    return (train_df, test_df)
    