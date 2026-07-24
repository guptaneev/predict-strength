import numpy as np
import pandas as pd

class BaselineModel:
    # copied into build_meet_pairs (pairs.py)
    def _add_delta_features(self, df) -> pd.DataFrame:
        df = df.copy()
        df['delta'] = df['next_TotalKg'] - df['TotalKg']
        df['avg_past_delta'] = (
            df.groupby('Name')['delta']
            .transform(lambda s: s.shift(1).expanding().mean())
        )
        return df

    def fit(self, train_df):
        self.population_avg_delta = train_df['delta'].mean()
    
    def predict(self, df):
        predicted_delta = df['avg_past_delta'].fillna(self.population_avg_delta)
        return df['TotalKg'] + predicted_delta
        