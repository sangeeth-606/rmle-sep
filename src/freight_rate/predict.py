"""
Freight Rate Prediction Inference Module

This module contains inference logic to load the trained model pipeline
and generate predictions on unseen data, including safety checks such
as a positivity floor.
"""

import os
import joblib
import numpy as np
import pandas as pd
from freight_rate.features import preprocess


def predict(df: pd.DataFrame, pipeline_path: str = "models/model_pipeline.joblib") -> np.ndarray:
    """
    Loads a serialized scikit-learn Pipeline and generates predictions.
    Applies preprocessing and enforces a positivity floor.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at: {pipeline_path}")
        
    # Load serialized pipeline
    pipeline = joblib.load(pipeline_path)
    
    # Run unfitted preprocessing steps
    df_preprocessed = preprocess(df)
    
    # Generate predictions (the pipeline regressor automatically inverts log-transform)
    predictions = pipeline.predict(df_preprocessed)
    
    # Enforce positivity floor as a safety net (rates must be > 0)
    predictions = np.maximum(predictions, 1.0)
    
    return predictions
