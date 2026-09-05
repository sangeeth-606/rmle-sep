"""
Freight Rate Feature Engineering Module

This module contains data cleaning, date/time feature extraction,
route/distance feature extraction, equipment-conditional imputation,
normalized frequency encoding, and the complete preprocessing pipeline
for the RouteRate-ML freight pricing pipeline.

Features included in the processed dataset:
1. Numeric Features:
   - Coordinates: pickup_lat, pickup_lon, delivery_lat, delivery_lon
   - Distance measures: distance (road miles), haversine_distance (great-circle miles),
     and distance_ratio (distance / haversine_distance)
   - Market signals: market_index, quote_signal
   - Load characteristics: weight (imputed conditionally by equipment type)
   - Calendar & Cyclical features:
     * month, day_of_week, is_weekend
     * sin_day_of_year, cos_day_of_year (annual harmonic cycle)
     * sin_day_of_week, cos_day_of_week (weekly harmonic cycle)
2. Categorical Features (Processed):
   - City frequencies: pickup, delivery (normalized proportions: count / N_train)
   - Equipment: Dry Van, Reefer, Flatbed (One-Hot Encoded)
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points in kilometers
    using the Haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    km = 6371.0 * c
    return km


def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points in miles.
    """
    return haversine_km(lat1, lon1, lat2, lon2) * 0.621371


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Normalizes equipment casing and whitespace.
    - Handles duplicate load_ids.
    - Guards against impossible distance and weight values (converts to NaN to be imputed).
    - Drops non-positive posted_rates (for training data only).
    """
    df = df.copy()
    
    # Normalize equipment casing and whitespace
    if 'equipment' in df.columns:
        df['equipment'] = df['equipment'].astype(str).str.strip().str.title()
        
    # Drop exact duplicate load_id rows
    if 'load_id' in df.columns:
        df = df.drop_duplicates(subset=['load_id'], keep='first')
        
    # Guard against impossible values for distance and weight by converting <= 0 to NaN
    if 'distance' in df.columns:
        df.loc[df['distance'] <= 0, 'distance'] = np.nan
    if 'weight' in df.columns:
        df.loc[df['weight'] <= 0, 'weight'] = np.nan
        
    # For training data only, drop non-positive posted_rate values
    if is_train and 'posted_rate' in df.columns:
        df = df[df['posted_rate'] > 0]
        
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds date-based features, including calendar components and cyclical harmonic encodings.
    Uses sin/cos transforms to enable periodic seasonality modeling without out-of-distribution
    threshold boundary clipping in tree-based regressors.
    """
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    
    df['day_of_week'] = dates.dt.dayofweek
    df['month'] = dates.dt.month
    df['day_of_year'] = dates.dt.dayofyear
    df['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    
    # Harmonic cyclical features (bounded in [-1, 1])
    day_of_year = dates.dt.dayofyear.astype(float)
    day_of_week = dates.dt.dayofweek.astype(float)
    
    df['sin_day_of_year'] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df['cos_day_of_year'] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    df['sin_day_of_week'] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df['cos_day_of_week'] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds route-based features, including haversine distance and distance ratio.
    """
    df = df.copy()
    
    if 'pickup_lat' in df.columns and 'pickup_lon' in df.columns:
        df['haversine_distance'] = haversine_miles(
            df['pickup_lat'], df['pickup_lon'],
            df['delivery_lat'], df['delivery_lon']
        )
        # Compute distance_ratio, guard against division by zero
        df['distance_ratio'] = df['distance'] / df['haversine_distance'].replace(0, np.nan)
        df['distance_ratio'] = df['distance_ratio'].fillna(1.0)
    else:
        df['haversine_distance'] = np.nan
        df['distance_ratio'] = 1.0
        
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to run non-fitted preprocessing steps (data cleaning,
    date features, route features). Safe to call on training, validation,
    and December prediction datasets.
    """
    df = clean_data(df, is_train=False)
    df = add_date_features(df)
    df = add_route_features(df)
    return df


class EquipmentConditionalImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to impute missing weight conditionally
    based on equipment type (Flatbed, Reefer, Dry Van).
    Falls back to the overall global median if a group is unseen.
    """
    def __init__(self, weight_col: str = 'weight', group_col: str = 'equipment'):
        self.weight_col = weight_col
        self.group_col = group_col
        self.group_medians_ = {}
        self.global_median_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if self.group_col in X_df.columns and self.weight_col in X_df.columns:
            self.group_medians_ = X_df.groupby(self.group_col)[self.weight_col].median().to_dict()
            valid_weights = X_df[self.weight_col].dropna()
            self.global_median_ = float(valid_weights.median()) if not valid_weights.empty else 30000.0
        else:
            self.global_median_ = 30000.0
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.weight_col in X_df.columns and self.group_col in X_df.columns:
            imputed = X_df[self.group_col].map(self.group_medians_).fillna(self.global_median_)
            X_df[self.weight_col] = X_df[self.weight_col].fillna(imputed).fillna(self.global_median_)
        elif self.weight_col in X_df.columns:
            X_df[self.weight_col] = X_df[self.weight_col].fillna(self.global_median_)
        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([self.weight_col], dtype=object)
        return np.array(input_features, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform normalized frequency encoding for city columns.
    Maps categories to their relative frequency proportion (count / N_train) in the training dataset.
    """
    def __init__(self, cols=None):
        self.cols = cols
        self.frequencies_ = {}
        self.min_freq_ = {}
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.frequencies_ = {}
        self.min_freq_ = {}
        n_samples = len(X_df)
        for col in self.cols:
            freq_series = X_df[col].value_counts(normalize=True)
            self.frequencies_[col] = freq_series.to_dict()
            self.min_freq_[col] = 1.0 / n_samples if n_samples > 0 else 0.0
        return self
        
    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_out = X_df.copy()
        for col in self.cols:
            # Map relative frequency proportion, default to 0 for unseen categories
            X_out[col] = X_out[col].map(self.frequencies_[col]).fillna(0.0).astype(float)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.cols, dtype=object)
        return np.array(input_features, dtype=object)


def build_feature_pipeline() -> Pipeline:
    """
    Assembles the complete preprocessing pipeline.
    Applies:
    - EquipmentConditionalImputer for weight based on equipment group
    - SimpleImputer for other numeric columns
    - Normalized FrequencyEncoder for city names
    - OneHotEncoder for equipment type
    """
    numeric_impute_cols = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'haversine_distance', 'distance_ratio',
        'market_index', 'quote_signal',
        'day_of_week', 'month', 'is_weekend',
        'sin_day_of_year', 'cos_day_of_year',
        'sin_day_of_week', 'cos_day_of_week'
    ]
    
    city_cols = ['pickup', 'delivery']
    eq_cols = ['equipment']
    
    # Numeric pipeline with median imputer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # City pipeline: normalized frequency encoding
    city_transformer = Pipeline(steps=[
        ('freq_enc', FrequencyEncoder(cols=city_cols))
    ])
    
    # Equipment pipeline: One-hot encode
    eq_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_impute_cols),
            ('cond_weight', 'passthrough', ['weight']),
            ('city', city_transformer, city_cols),
            ('eq', eq_transformer, eq_cols)
        ],
        remainder='drop'
    )
    
    # Preprocessor pipeline with equipment-conditional imputation step first
    full_preprocessor = Pipeline(steps=[
        ('conditional_imputer', EquipmentConditionalImputer(weight_col='weight', group_col='equipment')),
        ('column_transform', column_transformer)
    ])
    
    return Pipeline(steps=[('preprocessor', full_preprocessor)])
