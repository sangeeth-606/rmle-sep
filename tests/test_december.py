import pytest
import pandas as pd
from freight_rate.december import build_december_features, DEFAULT_COORDINATES

def test_build_december_features():
    raw_dec = pd.DataFrame({
        'pickup': ['Lexington'] * 5,
        'delivery': ['Fort Wayne'] * 5,
        'distance': [360.0] * 5,
        'equipment': ['Dry Van'] * 5,
        'weight': [32000.0] * 5,
        'date': pd.date_range('2025-12-01', periods=5, freq='D'),
        'predicted_rate': [0.0] * 5
    })
    
    features_df = build_december_features(raw_dec)
    
    # Assert coordinates were attached correctly
    assert features_df['pickup_lat'].iloc[0] == DEFAULT_COORDINATES['Lexington']['lat']
    assert features_df['pickup_lon'].iloc[0] == DEFAULT_COORDINATES['Lexington']['lon']
    assert features_df['delivery_lat'].iloc[0] == DEFAULT_COORDINATES['Fort Wayne']['lat']
    assert features_df['delivery_lon'].iloc[0] == DEFAULT_COORDINATES['Fort Wayne']['lon']
    
    # Assert market signals are populated with no NaNs
    assert not features_df['market_index'].isna().any()
    assert not features_df['quote_signal'].isna().any()
