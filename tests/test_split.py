import pytest
import pandas as pd
from freight_rate.split import chronological_split

def test_chronological_split_no_leakage():
    dates = pd.date_range('2025-01-01', '2025-10-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'posted_rate': [2000.0] * len(dates)
    })
    
    train_df, holdout_df = chronological_split(df, holdout_frac=0.2)
    
    # Train max date must be strictly earlier than holdout min date
    assert pd.to_datetime(train_df['date']).max() < pd.to_datetime(holdout_df['date']).min()
    
    # Check that rows are non-empty and non-overlapping
    assert len(train_df) > 0
    assert len(holdout_df) > 0
    assert len(train_df) + len(holdout_df) == len(df)
