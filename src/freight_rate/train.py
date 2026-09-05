"""
Freight Rate Model Training Module

This module contains functions for hyperparameter tuning using TimeSeriesSplit
and RandomizedSearchCV, evaluating against chronological holdouts, plotting
feature importances, and serializing the complete ML pipeline with metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def tune_hyperparameters(
    feature_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 16,
    cv_splits: int = 4,
    random_seed: int = 42
) -> tuple[dict, dict]:
    """
    Performs hyperparameter search using TimeSeriesSplit cross-validation
    and RandomizedSearchCV over the XGBRegressor search space.
    
    Returns:
        (best_params, cv_results_summary): tuple containing best params and CV metrics.
    """
    print(f"Starting RandomizedSearchCV with {cv_splits}-fold TimeSeriesSplit...")
    
    # Define XGBRegressor wrapped in TransformedTargetRegressor with log1p/expm1
    base_xgb = XGBRegressor(random_state=random_seed, n_jobs=-1)
    target_regressor = TransformedTargetRegressor(
        regressor=base_xgb,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # Assembled full pipeline for search
    search_pipeline = Pipeline(steps=[
        ('preprocessor', feature_pipeline.named_steps['preprocessor']),
        ('regressor', target_regressor)
    ])
    
    param_distributions = {
        'regressor__regressor__max_depth': [4, 5, 6, 7],
        'regressor__regressor__learning_rate': [0.03, 0.05, 0.08, 0.1],
        'regressor__regressor__n_estimators': [150, 200, 300, 400],
        'regressor__regressor__subsample': [0.7, 0.8, 0.9, 1.0],
        'regressor__regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'regressor__regressor__min_child_weight': [1, 3, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    search = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=random_seed,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    # Extract clean parameter dictionary for the regressor
    raw_best_params = search.best_params_
    best_params = {
        k.replace('regressor__regressor__', ''): v
        for k, v in raw_best_params.items()
    }
    
    cv_summary = {
        'best_cv_mae': float(-search.best_score_),
        'cv_splits': cv_splits,
        'n_iter': n_iter
    }
    
    print(f"Best CV MAE: ${cv_summary['best_cv_mae']:.2f}")
    print(f"Best hyperparameters found: {best_params}")
    return best_params, cv_summary


def train_final_pipeline(
    feature_pipeline: Pipeline,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    best_params: dict,
    random_seed: int = 42
) -> Pipeline:
    """
    Fits the final pipeline (feature preprocessing + XGBoost model)
    on the entire dataset.
    """
    print("Training final model on full train_test dataset...")
    
    xgb_model = XGBRegressor(
        max_depth=best_params.get('max_depth', 5),
        learning_rate=best_params.get('learning_rate', 0.05),
        n_estimators=best_params.get('n_estimators', 200),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_weight=best_params.get('min_child_weight', 1),
        random_state=random_seed,
        n_jobs=-1
    )
    
    regressor = TransformedTargetRegressor(
        regressor=xgb_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', feature_pipeline.named_steps['preprocessor']),
        ('regressor', regressor)
    ])
    
    final_pipeline.fit(X_full, y_full)
    return final_pipeline


def save_feature_importance_plot(
    fitted_pipeline: Pipeline,
    output_path: str = "reports/figures/feature_importance.png"
) -> None:
    """
    Extracts gain-based feature importances from the fitted XGBoost model
    and saves a horizontal bar chart of the importances.
    """
    print("Generating feature importance plot...")
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    regressor = fitted_pipeline.named_steps['regressor']
    
    # Extract feature names from preprocessor
    if hasattr(preprocessor, 'named_steps') and 'column_transform' in preprocessor.named_steps:
        col_trans = preprocessor.named_steps['column_transform']
        feature_names = list(col_trans.get_feature_names_out())
    elif hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"f_{i}" for i in range(regressor.regressor_.n_features_in_)]
        
    xgb_model = regressor.regressor_
    importances = xgb_model.feature_importances_
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Clean up prefixes for display
    df_importance['Feature'] = (
        df_importance['Feature']
        .str.replace('num__', '', regex=False)
        .str.replace('city__', '', regex=False)
        .str.replace('eq__', '', regex=False)
        .str.replace('cond_weight__', '', regex=False)
    )
    
    plt.figure(figsize=(10, 7))
    plt.barh(df_importance['Feature'], df_importance['Importance'], color='#064A56', edgecolor='none')
    plt.xlabel('Feature Importance (Gain)')
    plt.title('Gain-Based Feature Importance (XGBoost)')
    plt.grid(axis='x', color='#D9E2E4', linewidth=0.8, alpha=0.5)
    plt.gca().spines[['top', 'right']].set_visible(False)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance plot to {output_path}")
