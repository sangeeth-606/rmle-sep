import pytest
import numpy as np
import pandas as pd
from freight_rate.predict import predict

def test_predict_positivity_floor(tmp_path):
    df_sample = pd.DataFrame({
        'load_id': ['TE-000001', 'TE-000002'],
        'pickup': ['Lexington', 'Fort Wayne'],
        'delivery': ['Fort Wayne', 'Lexington'],
        'pickup_lat': [36.99152, 41.31561],
        'pickup_lon': [-84.99876, -85.36206],
        'delivery_lat': [41.31561, 36.99152],
        'delivery_lon': [-85.36206, -84.99876],
        'distance': [360.0, 360.0],
        'equipment': ['Dry Van', 'Dry Van'],
        'weight': [32000.0, 32000.0],
        'date': ['2025-11-01', '2025-11-02'],
        'market_index': [0.95, 0.95],
        'quote_signal': [2.05, 2.05]
    })
    
    # Predict using saved model
    preds = predict(df_sample, pipeline_path="models/model_pipeline.joblib")
    
    assert len(preds) == 2
    assert (preds > 0).all()
    assert (preds >= 1.0).all()
    assert isinstance(preds, np.ndarray)
