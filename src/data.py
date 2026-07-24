from pathlib import Path

import pandas as pd

def load_clean_data():
    project_root = Path(__file__).resolve().parents[1]
    raw_data = project_root / 'data' / 'openpowerlifting.csv'
    processed_data = project_root / 'data' / 'openpowerlifting_processed.csv'

    df = pd.read_csv(raw_data, low_memory=False)

    print(f"Starting with {len(df)} rows")

    # Normalize the fields we need for filtering.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Name'] = df['Name'].astype('string').str.strip()
    df['TotalKg'] = pd.to_numeric(df['TotalKg'], errors='coerce')
    df['BodyweightKg'] = pd.to_numeric(df['BodyweightKg'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['PlaceRank'] = pd.to_numeric(df['Place'], errors='coerce')
    df['Event'] = df['Event'].astype('string').str.strip()
    df['Sanctioned'] = df['Sanctioned'].astype('string').str.strip()

    valid_rows = (
        df['Date'].notna()
        & df['Name'].notna()
        & (df['Name'] != '')
        & df['TotalKg'].notna()
        & (df['TotalKg'] > 0)
        & df['BodyweightKg'].notna()
        & df['Age'].notna()
        & df['PlaceRank'].notna()
        & (df['PlaceRank'] >= 1)
        & df['Event'].eq('SBD')
        & df['Sanctioned'].eq('Yes')
        & df['BodyweightKg'].between(20, 400)
        & df['Age'].between(10, 100)
    )

    df = df.loc[valid_rows].copy()

    print(f"After dropping missing/invalid dates, identity, totals, bodyweight, age, and non-results, {len(df)} rows")

    meet_columns = ['Date', 'MeetName', 'MeetTown', 'MeetCountry']
    df['_meet_id'] = df[meet_columns].astype('string').fillna('').agg(' | '.join, axis=1)

    # Keep one row per lifter-meet, preferring the best total when duplicates exist.
    df = df.sort_values(['Name', '_meet_id', 'TotalKg', 'PlaceRank'], ascending=[True, True, False, True])
    df = df.drop_duplicates(subset=['Name', '_meet_id'], keep='first')

    print(f"After removing duplicate entries for the same lifter and meet, {len(df)} rows")

    # Keep only lifters with at least two distinct meets because the target is meet t -> meet t+1.
    df = df[df.groupby('Name')['_meet_id'].transform('nunique') > 1]

    print(f"After keeping lifters who competed in more than one meet, {len(df)} rows")

    df = df.drop(columns=['_meet_id', 'PlaceRank'])

    df.to_csv(processed_data, index=False)

    print(f"Saved processed data to {processed_data}")
    return df