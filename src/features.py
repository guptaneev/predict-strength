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

    # --- attempt-level features (from the current meet's own attempts - not leakage,
    # this is all known at the time of the meet being used to predict the NEXT one) ---

    attempt_cols = [
        'Squat1Kg', 'Squat2Kg', 'Squat3Kg',
        'Bench1Kg', 'Bench2Kg', 'Bench3Kg',
        'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg',
    ]
    # negative attempt value = missed; NaN = attempt not taken, doesn't count as missed
    pairs_df['missed_attempts_last_meet'] = (pairs_df[attempt_cols] < 0).sum(axis=1)

    # opener -> best jump, per lift, then averaged across squat/bench/deadlift
    lift_pct_gains = []
    for opener_col, best_col in [
        ('Squat1Kg', 'Best3SquatKg'),
        ('Bench1Kg', 'Best3BenchKg'),
        ('Deadlift1Kg', 'Best3DeadliftKg'),
    ]:
        opener = pairs_df[opener_col].abs()
        pct_gain = (pairs_df[best_col] - opener) / opener
        lift_pct_gains.append(pct_gain)

    pairs_df['opener_to_best_pct_gain'] = pd.concat(lift_pct_gains, axis=1).mean(axis=1)

    # attempt data is missing all-or-nothing per meet (some meets/federations only
    # report best-lift summaries, not attempt-by-attempt data) - not random, so instead
    # of dropping ~32% of rows, flag it and fill with a neutral value. Lets tree-based
    # models learn to route around the filled value using the indicator.
    pairs_df['has_attempt_data'] = pairs_df[attempt_cols].notna().any(axis=1).astype(int)
    pairs_df['opener_to_best_pct_gain'] = pairs_df['opener_to_best_pct_gain'].fillna(0)

    # 'Tested' only ever contains 'Yes' or NaN - NaN means not tested, by omission
    pairs_df['is_tested'] = (pairs_df['Tested'] == 'Yes').astype(int)

    # equipment category (one-hot) - Raw dropped as the baseline/reference category
    equipment_dummies = pd.get_dummies(pairs_df['Equipment'], prefix='equipment', drop_first=True)
    pairs_df = pd.concat([pairs_df, equipment_dummies.astype(int)], axis=1)

    # did equipment change since the lifter's last meet - a switch (e.g. Raw -> Single-ply)
    # would produce a large total jump that has nothing to do with "training progress"
    pairs_df['equipment_changed'] = (
        pairs_df['Equipment'] != grouped['Equipment'].shift(1)
    ).astype(int)
    # first meet has nothing to compare against - not a real "change", treat as 0
    pairs_df.loc[grouped.cumcount() == 0, 'equipment_changed'] = 0

    # --- placement / relative standing (from the current meet) ---

    pairs_df['place_last_meet'] = pd.to_numeric(pairs_df['Place'], errors='coerce')

    # percentile standing within the same competition, using Dots (bodyweight-normalized
    # strength score) instead of raw Place - comparable across weight classes/divisions
    # without needing to precisely reconstruct official placement categories
    meet_id = pairs_df[['Date', 'MeetName', 'MeetTown', 'MeetCountry']].astype('string').fillna('').agg('|'.join, axis=1)
    pairs_df['dots_percentile_in_meet'] = pairs_df.groupby(meet_id)['Dots'].rank(pct=True)

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

        
        
        
        
    