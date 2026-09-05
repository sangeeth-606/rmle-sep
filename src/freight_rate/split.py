"""
Freight Rate Train/Validation Split Module

This module contains the chronological split function to separate the
development dataset into training and local holdout splits without
temporal data leakage.
"""

import pandas as pd


def chronological_split(
    df: pd.DataFrame, date_col: str = "date", holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame chronologically.
    
    Sorts by the date column and places the most recent `holdout_frac`
    duration of data into the holdout split, ensuring that all records
    in the training split occur strictly before the records in the holdout split.
    
    Parameters:
        df: Input DataFrame.
        date_col: Name of the column containing the date.
        holdout_frac: Fraction of the total date range duration to assign to the holdout.
        
    Returns:
        (train_df, holdout_df): A tuple of training and holdout DataFrames.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort chronologically to preserve order
    df = df.sort_values(date_col).reset_index(drop=True)
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Calculate duration of the date range in days
    total_days = (max_date - min_date).days
    
    # Find the split date boundary
    split_days = int(total_days * (1.0 - holdout_frac))
    split_date = min_date + pd.to_timedelta(split_days, unit="D")
    
    # Normalize split date to midnight to make division clean and simple
    split_date = pd.to_datetime(split_date.date())
    
    # Split the dataset
    train_df = df[df[date_col] < split_date].copy()
    holdout_df = df[df[date_col] >= split_date].copy()
    
    return train_df, holdout_df
