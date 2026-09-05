import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add src/ to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from freight_rate.features import preprocess, build_feature_pipeline
from freight_rate.split import chronological_split
from freight_rate.train import tune_hyperparameters, train_final_pipeline, save_feature_importance_plot

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

def compute_metadata(df_train: pd.DataFrame, df_val_path: str = "data/validation.csv") -> dict:
    """
    Precomputes city coordinate lookups and daily average market signals
    to bundle into the model pipeline artifact.
    """
    metadata = {}
    
    # City coordinates lookup
    coords = {
        "Lexington": {"lat": 36.99152, "lon": -84.99876},
        "Fort Wayne": {"lat": 41.31561, "lon": -85.36206}
    }
    metadata["city_coordinates"] = coords
    
    # Precompute daily market signals table
    frames = [df_train[['date', 'market_index', 'quote_signal']].copy()]
    if os.path.exists(df_val_path):
        val_df = pd.read_csv(df_val_path)
        if 'date' in val_df.columns and 'market_index' in val_df.columns:
            frames.append(val_df[['date', 'market_index', 'quote_signal']].copy())
            
    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
    daily = combined.groupby('date')[['market_index', 'quote_signal']].mean().reset_index()
    daily['market_index'] = daily['market_index'].interpolate(method='linear').bfill().ffill()
    daily['quote_signal'] = daily['quote_signal'].interpolate(method='linear').bfill().ffill()
    
    metadata["daily_market_signals"] = daily.to_dict(orient='records')
    return metadata

def main():
    print("Loading data...")
    df = pd.read_csv("data/train_test.csv")
    
    print("Performing chronological split for local validation...")
    train_df, holdout_df = chronological_split(df, holdout_frac=0.2)
    
    # Preprocess splits
    X_train_raw = preprocess(train_df)
    X_holdout_raw = preprocess(holdout_df)
    y_train = train_df['posted_rate']
    y_holdout = holdout_df['posted_rate']
    
    # Preprocess full dataset
    X_full_raw = preprocess(df)
    y_full = df['posted_rate']
    
    # Initialize feature pipeline
    feature_pipeline = build_feature_pipeline()
    
    # Tune hyperparameters using TimeSeriesSplit and RandomizedSearchCV
    best_params, cv_summary = tune_hyperparameters(
        feature_pipeline=feature_pipeline,
        X_train=X_train_raw,
        y_train=y_train,
        n_iter=16,
        cv_splits=4,
        random_seed=42
    )
    
    # Fit preprocessing and model on train split to evaluate on chronological holdout
    print("Evaluating tuned model on local holdout...")
    eval_preprocessor = build_feature_pipeline()
    eval_preprocessor.fit(X_train_raw)
    
    X_train_trans = eval_preprocessor.transform(X_train_raw)
    X_holdout_trans = eval_preprocessor.transform(X_holdout_raw)
    
    eval_model = XGBRegressor(
        max_depth=best_params.get('max_depth', 5),
        learning_rate=best_params.get('learning_rate', 0.05),
        n_estimators=best_params.get('n_estimators', 200),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_weight=best_params.get('min_child_weight', 1),
        random_state=42,
        n_jobs=-1
    )
    
    eval_regressor = TransformedTargetRegressor(
        regressor=eval_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    eval_regressor.fit(X_train_trans, y_train)
    y_pred = eval_regressor.predict(X_holdout_trans)
    
    # Compute holdout metrics
    mae = float(mean_absolute_error(y_holdout, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
    mape = float(mean_absolute_percentage_error(y_holdout, y_pred))
    r2 = float(r2_score(y_holdout, y_pred))
    
    print("\n--- Tuned XGBoost Holdout Metrics ---")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    # Load baseline metrics to compare
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {"baseline": {}, "final": {}}
        
    baseline = metrics.get("baseline", {})
    if baseline:
        b_mae = baseline.get("mae", 1.0)
        b_rmse = baseline.get("rmse", 1.0)
        b_mape = baseline.get("mape", 1.0)
        
        improvement_mae = ((b_mae - mae) / b_mae) * 100
        improvement_rmse = ((b_rmse - rmse) / b_rmse) * 100
        improvement_mape = ((b_mape - mape) / b_mape) * 100
        
        print("\n--- Comparison vs Baseline ---")
        print(f"MAE Improvement:  {improvement_mae:.2f}% (${b_mae:.2f} -> ${mae:.2f})")
        print(f"RMSE Improvement: {improvement_rmse:.2f}% (${b_rmse:.2f} -> ${rmse:.2f})")
        print(f"MAPE Improvement: {improvement_mape:.2f}% ({b_mape*100:.2f}% -> {mape*100:.2f}%)")
    else:
        improvement_mae = 0.0
        improvement_rmse = 0.0
        improvement_mape = 0.0
        
    # Save metrics
    metrics["final"] = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "improvement_mae_pct": improvement_mae,
        "improvement_rmse_pct": improvement_rmse,
        "improvement_mape_pct": improvement_mape,
        "cv_summary": cv_summary,
        "best_params": best_params
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nUpdated {metrics_path} with final metrics.")
    
    # Build and fit final pipeline on entire dataset
    print("\nFitting preprocessing pipeline on full dataset...")
    full_feature_pipeline = build_feature_pipeline()
    full_feature_pipeline.fit(X_full_raw)
    
    final_pipeline = train_final_pipeline(
        feature_pipeline=full_feature_pipeline,
        X_full=X_full_raw,
        y_full=y_full,
        best_params=best_params,
        random_seed=42
    )
    
    # Compute and attach metadata bundle directly onto the pipeline object
    final_pipeline.metadata_ = compute_metadata(df)
    
    # Save feature importance plot using final pipeline
    save_feature_importance_plot(final_pipeline, "reports/figures/feature_importance.png")
    
    # Save complete fitted pipeline to models/model_pipeline.joblib
    pipeline_path = "models/model_pipeline.joblib"
    joblib.dump(final_pipeline, pipeline_path)
    print(f"Successfully persisted complete model pipeline with metadata to {pipeline_path}")

if __name__ == "__main__":
    main()
