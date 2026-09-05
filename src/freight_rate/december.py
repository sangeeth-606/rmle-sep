"""
December Prediction Reconstruction Module

This module reconstructs missing geographic coordinates (lat/lon)
and looks up market signals (market_index, quote_signal) for the
December 2025 fixed-route chart predictions.

Lookups are read directly from the serialized pipeline metadata,
decoupling runtime inference from raw training CSVs.
"""

import os
import joblib
import pandas as pd
import numpy as np
from freight_rate.features import preprocess

# Default in-distribution coordinates for Lexington -> Fort Wayne
DEFAULT_COORDINATES = {
    "Lexington": {"lat": 36.99152, "lon": -84.99876},
    "Fort Wayne": {"lat": 41.31561, "lon": -85.36206}
}


def get_daily_market_signals(
    pipeline_artifact=None,
    train_path: str = "data/train_test.csv",
    val_path: str = "data/validation.csv"
) -> pd.DataFrame:
    """
    Retrieves daily market signals from the serialized pipeline metadata if available,
    falling back to historical CSVs only if metadata is not provided.
    """
    # 1. Try to load from pipeline metadata
    if pipeline_artifact is not None and hasattr(pipeline_artifact, "metadata_"):
        meta = pipeline_artifact.metadata_
        if "daily_market_signals" in meta:
            df_signals = pd.DataFrame(meta["daily_market_signals"])
            df_signals["date"] = pd.to_datetime(df_signals["date"])
            return df_signals

    # 2. Try loading default model_pipeline.joblib if it exists on disk
    default_pipeline_path = "models/model_pipeline.joblib"
    if os.path.exists(default_pipeline_path):
        try:
            loaded = joblib.load(default_pipeline_path)
            if hasattr(loaded, "metadata_") and "daily_market_signals" in loaded.metadata_:
                df_signals = pd.DataFrame(loaded.metadata_["daily_market_signals"])
                df_signals["date"] = pd.to_datetime(df_signals["date"])
                return df_signals
        except Exception:
            pass

    # 3. Fallback to computing from raw CSVs if pipeline metadata is not present
    if os.path.exists(train_path):
        train = pd.read_csv(train_path)
        train["date"] = pd.to_datetime(train["date"])
        frames = [train]
        if os.path.exists(val_path):
            val = pd.read_csv(val_path)
            val["date"] = pd.to_datetime(val["date"])
            frames.append(val)
        combined = pd.concat(frames, ignore_index=True)
        daily = combined.groupby("date")[["market_index", "quote_signal"]].mean().reset_index()
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["market_index"] = daily["market_index"].interpolate(method="linear").bfill().ffill()
        daily["quote_signal"] = daily["quote_signal"].interpolate(method="linear").bfill().ffill()
        return daily

    # Fallback to constant defaults if no data exists
    dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    return pd.DataFrame({
        "date": dates,
        "market_index": [0.93] * len(dates),
        "quote_signal": [2.05] * len(dates)
    })


def build_december_features(
    dec_raw: pd.DataFrame,
    pipeline_artifact=None
) -> pd.DataFrame:
    """
    Reconstructs the full feature set for the 31 December fixed-route rows:
    - Populates coordinates for Lexington and Fort Wayne.
    - Merges daily market signals (from serialized metadata).
    - Returns dataframe ready for preprocessing and model inference.
    """
    df = dec_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Coordinates from metadata or defaults
    coords = DEFAULT_COORDINATES
    if pipeline_artifact is not None and hasattr(pipeline_artifact, "metadata_"):
        coords = pipeline_artifact.metadata_.get("city_coordinates", DEFAULT_COORDINATES)
        
    df["pickup_lat"] = coords["Lexington"]["lat"]
    df["pickup_lon"] = coords["Lexington"]["lon"]
    df["delivery_lat"] = coords["Fort Wayne"]["lat"]
    df["delivery_lon"] = coords["Fort Wayne"]["lon"]
    
    # Daily market signals from metadata
    daily_signals = get_daily_market_signals(pipeline_artifact)
    df = df.merge(daily_signals, on="date", how="left")
    
    # Fill any missing signal values
    df["market_index"] = df["market_index"].ffill().bfill().fillna(0.93)
    df["quote_signal"] = df["quote_signal"].ffill().bfill().fillna(2.05)
    
    # Add a mock load_id so standard preprocessing runs cleanly
    df["load_id"] = [f"DEC-{i:02d}" for i in range(1, len(df) + 1)]
    
    return df
