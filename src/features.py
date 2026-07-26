import pandas as pd

from src.models.linear_regression import LinearRegressionScratch



def build_features(pairs_df) -> pd.DataFrame:
    pairs_df = pairs_df.copy()
    grouped = pairs_df.groupby('Name')
    
    # TotalKg, BodyweightKg, Age - no computation needed
    
    # 1-indexed count of meets up to and including meet t
    pairs_df['meet_number'] = grouped.cumcount() + 1
    
    # days since last meet (returns a Timedelta object, use .dt.days to return an integer number)
    pairs_df['days_since_last_meet'] = (pairs_df['Date'] - grouped['Date'].shift(1)).dt.days
    
    # delta between meet t and meet t - 1
    pairs_df['prev_total_change'] = pairs_df['TotalKg'] - grouped['TotalKg'].shift(1)
    
    # delta between bodyweight of lifter between meet t and meet t - 1
    pairs_df['bodyweight_change'] = pairs_df['BodyweightKg'] - grouped['BodyweightKg'].shift(1)
    
    # average of last two prev_total_change
    pairs_df['rolling_avg_change_last_2_meets'] = (
            grouped['prev_total_change']
            .transform(lambda s: s.shift(1).rolling(2).mean())
        )

    # average of last three prev_total_change
    pairs_df['rolling_avg_change_last_3_meets'] = (
            grouped['prev_total_change']
            .transform(lambda s: s.shift(1).rolling(3).mean())
        )

    # average of last four prev_total_change
    pairs_df['rolling_avg_change_last_4_meets'] = (
            grouped['prev_total_change']
            .transform(lambda s: s.shift(1).rolling(4).mean())
        )

    #
    pairs_df['linear_trend_slope_last_N_meets'] = grouped.apply(linear_trend_kgs).droplevel(0)

    return pairs_df

def linear_trend_kgs(sliced_df, n=5) -> pd.Series:
    model = LinearRegressionScratch(method='normal_equation')
    trend = []

    # convert once per lifter (numpy arrays), instead of re-deriving these
    # from the DataFrame on every single row via .iloc / .dt.days
    totals = sliced_df['TotalKg'].to_numpy()
    day_numbers = sliced_df['Date'].to_numpy().astype('datetime64[D]').astype(int)

    for i in range(len(sliced_df)):
        start = max(0, i - n)
        y = totals[start:i]
        days = day_numbers[start:i]

        # not enough (unique) points to fit a line
        if len(days) < 2 or len(set(days)) < 2:
            trend.append(float("nan"))
            continue

        # days since first meet in this window
        X = (days - days[0]).reshape(-1, 1).astype(float)
        model.fit(X, y)
        trend.append(model.weights[0])

    return pd.Series(trend, index=sliced_df.index)

        
        
        
        
    