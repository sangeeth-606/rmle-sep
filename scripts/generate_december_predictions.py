import os
import sys
import pandas as pd
import numpy as np

# Add src/ to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from freight_rate.december import build_december_features
from freight_rate.predict import predict

def main():
    print("Loading December chart inputs...")
    dec_raw = pd.read_csv("data/december_chart_inputs.csv")
    
    print("Reconstructing geographic coordinates and daily market signals from pipeline metadata...")
    dec_inputs = build_december_features(dec_raw)
    
    print("Generating predictions using the primary model pipeline...")
    preds = predict(dec_inputs, "models/model_pipeline.joblib")
    
    # Write predictions into the copy of the original 7-column DataFrame
    dec_predictions = dec_raw.copy()
    dec_predictions["predicted_rate"] = preds
    
    # Assert formatting & rules
    print("Performing assertions...")
    assert len(dec_predictions) == 31, f"Error: Expected 31 rows, got {len(dec_predictions)}"
    assert list(dec_predictions.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"], "Error: Columns mismatch"
    assert (dec_predictions["predicted_rate"] > 0).all(), "Error: Non-positive predicted rates found"
    
    # Write to root december_predictions.csv
    output_path = "december_predictions.csv"
    dec_predictions.to_csv(output_path, index=False)
    
    # Also fill predicted_rate directly into data/december_chart_inputs.csv for 100% prompt compliance
    dec_chart_inputs_path = "data/december_chart_inputs.csv"
    dec_predictions.to_csv(dec_chart_inputs_path, index=False)
    
    print(f"\nSuccessfully wrote December predictions to: {output_path} and {dec_chart_inputs_path}")
    print(f"Min prediction: ${preds.min():.2f}")
    print(f"Max prediction: ${preds.max():.2f}")
    print(f"Mean prediction: ${preds.mean():.2f}")

if __name__ == "__main__":
    main()
