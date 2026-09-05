import os
import sys
import json
import pandas as pd
import numpy as np

# Add src/ to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from freight_rate.features import preprocess, build_feature_pipeline
from freight_rate.split import chronological_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def main():
    print("Loading data...")
    # Load train_test data
    df = pd.read_csv("data/train_test.csv")
    
    print("Splitting data chronologically...")
    # Perform chronological split (holdout_frac=0.2)
    train_df, holdout_df = chronological_split(df, holdout_frac=0.2)
    
    # Keep target values in original scale
    y_train = train_df['posted_rate']
    y_holdout = holdout_df['posted_rate']
    
    print("Preprocessing splits...")
    # Preprocess splits
    X_train_raw = preprocess(train_df)
    X_holdout_raw = preprocess(holdout_df)
    
    # Fit features pipeline
    print("Fitting features pipeline...")
    pipeline = build_feature_pipeline()
    pipeline.fit(X_train_raw)
    
    X_train_trans = pipeline.transform(X_train_raw)
    X_holdout_trans = pipeline.transform(X_holdout_raw)
    
    print("Training Linear Regression baseline...")
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_trans, y_train)
    
    # Predict on holdout
    y_pred = model.predict(X_holdout_trans)
    
    # Calculate baseline metrics in original dollar scale
    mae = float(mean_absolute_error(y_holdout, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
    mape = float(mean_absolute_percentage_error(y_holdout, y_pred))
    
    print(f"\n--- Baseline Holdout Metrics (Original Scale) ---")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape * 100:.2f}%")
    
    # Save metrics to models/metrics.json
    metrics_path = "models/metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics = {
        "baseline": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        },
        "final": {}
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nSaved baseline metrics to {metrics_path}")

if __name__ == "__main__":
    main()
