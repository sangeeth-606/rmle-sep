import pytest
import numpy as np
import pandas as pd
from freight_rate.features import (
    haversine_km,
    haversine_miles,
    clean_data,
    add_date_features,
    add_route_features,
    preprocess,
    EquipmentConditionalImputer,
    FrequencyEncoder,
    build_feature_pipeline
)

def test_haversine_distance():
    # Lexington KY to Fort Wayne IN
    lex_lat, lex_lon = 36.99152, -84.99876
    fw_lat, fw_lon = 41.31561, -85.36206
    
    km = haversine_km(lex_lat, lex_lon, fw_lat, fw_lon)
    miles = haversine_miles(lex_lat, lex_lon, fw_lat, fw_lon)
    
    assert km > 450 and km < 500, f"Unexpected km: {km}"
    assert miles > 280 and miles < 320, f"Unexpected miles: {miles}"
    
    # Same point distance is 0
    assert np.isclose(haversine_km(lex_lat, lex_lon, lex_lat, lex_lon), 0.0)

def test_clean_data():
    df = pd.DataFrame({
        'load_id': ['L1', 'L1', 'L2', 'L3'],
        'equipment': ['  dry van ', 'dry van', 'reefer', 'FLATBED'],
        'distance': [300.0, 300.0, -50.0, 0.0],
        'weight': [30000.0, 30000.0, 0.0, 45000.0],
        'posted_rate': [1200.0, 1200.0, -100.0, 1500.0]
    })
    
    # For training
    cleaned_train = clean_data(df, is_train=True)
    assert len(cleaned_train) == 2  # L1 duplicate dropped, L2 negative rate dropped
    assert cleaned_train['equipment'].tolist() == ['Dry Van', 'Flatbed']
    assert np.isnan(cleaned_train.loc[cleaned_train['load_id'] == 'L3', 'distance'].values[0])

def test_equipment_conditional_imputer():
    df = pd.DataFrame({
        'equipment': ['Flatbed', 'Flatbed', 'Flatbed', 'Reefer', 'Reefer', 'Dry Van'],
        'weight': [45000.0, 47000.0, np.nan, 32000.0, np.nan, np.nan]
    })
    
    imputer = EquipmentConditionalImputer(weight_col='weight', group_col='equipment')
    imputer.fit(df)
    
    transformed = imputer.transform(df)
    # Flatbed median is 46000.0
    assert transformed.loc[2, 'weight'] == 46000.0
    # Reefer median is 32000.0
    assert transformed.loc[4, 'weight'] == 32000.0
    
    # Test unseen equipment falls back to global median
    unseen_df = pd.DataFrame({'equipment': ['Tanker'], 'weight': [np.nan]})
    unseen_trans = imputer.transform(unseen_df)
    assert not np.isnan(unseen_trans.loc[0, 'weight'])

def test_normalized_frequency_encoder():
    df = pd.DataFrame({
        'pickup': ['Chicago', 'Chicago', 'Chicago', 'Dallas', 'Atlanta']
    })
    
    encoder = FrequencyEncoder(cols=['pickup'])
    encoder.fit(df)
    
    transformed = encoder.transform(df)
    # Chicago = 3 / 5 = 0.6
    assert np.isclose(transformed.loc[0, 'pickup'], 0.6)
    # Dallas = 1 / 5 = 0.2
    assert np.isclose(transformed.loc[3, 'pickup'], 0.2)
    
    # Test unseen city defaults to 0.0
    unseen_df = pd.DataFrame({'pickup': ['Miami']})
    unseen_trans = encoder.transform(unseen_df)
    assert unseen_trans.loc[0, 'pickup'] == 0.0

def test_add_date_features_cyclics():
    df = pd.DataFrame({'date': ['2025-01-01', '2025-06-15', '2025-12-31']})
    res = add_date_features(df)
    
    # All sin/cos must stay bounded strictly within [-1.0, 1.0]
    for col in ['sin_day_of_year', 'cos_day_of_year', 'sin_day_of_week', 'cos_day_of_week']:
        assert col in res.columns
        assert (res[col] >= -1.0).all() and (res[col] <= 1.0).all()
        assert not res[col].isna().any()

def test_full_pipeline_transform():
    df = pd.DataFrame({
        'load_id': ['L1', 'L2'],
        'pickup': ['Chicago', 'Dallas'],
        'delivery': ['Atlanta', 'Houston'],
        'pickup_lat': [41.8781, 32.7767],
        'pickup_lon': [-87.6298, -96.7970],
        'delivery_lat': [33.7490, 29.7604],
        'delivery_lon': [-84.3880, -95.3698],
        'distance': [715.0, 240.0],
        'equipment': ['Dry Van', 'Reefer'],
        'weight': [34000.0, np.nan],
        'date': ['2025-05-01', '2025-05-02'],
        'market_index': [1.2, np.nan],
        'quote_signal': [2.1, 2.05]
    })
    
    df_pre = preprocess(df)
    pipeline = build_feature_pipeline()
    pipeline.fit(df_pre)
    transformed = pipeline.transform(df_pre)
    
    # Transformed must have zero null values
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 2
