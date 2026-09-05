import os
import sys
import pandas as pd
import numpy as np

# Add src/ to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from freight_rate.predict import predict

def main():
    print("Loading validation data...")
    val_df = pd.read_csv("data/validation.csv")
    template_df = pd.read_csv("data/validation_predictions_template.csv")
    
    print("Generating predictions...")
    preds = predict(val_df, "models/model_pipeline.joblib")
    
    # Create prediction frame
    pred_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": preds
    })
    
    print("Merging predictions with template...")
    # Merge on load_id to ensure order consistency
    merged_df = template_df[["load_id"]].merge(pred_df, on="load_id", how="left")
    
    # Safety Checks
    print("Performing safety assertions...")
    assert len(merged_df) == 12000, f"Error: Expected 12,000 rows, got {len(merged_df)}"
    assert list(merged_df.columns) == ["load_id", "predicted_rate"], "Error: Columns must be [load_id, predicted_rate]"
    assert not merged_df["load_id"].isnull().any(), "Error: Missing load_id found"
    assert not merged_df["predicted_rate"].isnull().any(), "Error: Missing predicted_rate found"
    assert not merged_df["load_id"].duplicated().any(), "Error: Duplicate load_id found"
    
    # Ensure all predictions are positive
    min_rate = merged_df["predicted_rate"].min()
    assert min_rate > 0, f"Error: Non-positive predicted rate found! Min rate: {min_rate}"
    
    # Assert ID set equivalence
    template_ids = set(template_df["load_id"])
    pred_ids = set(merged_df["load_id"])
    assert template_ids == pred_ids, "Error: Pred IDs do not match template IDs exactly"
    
    # Write file to repo root
    output_path = "validation_predictions.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully wrote validation predictions to: {output_path}")
    print(f"Min prediction: ${min_rate:.2f}")
    print(f"Max prediction: ${merged_df['predicted_rate'].max():.2f}")
    print(f"Mean prediction: ${merged_df['predicted_rate'].mean():.2f}")

if __name__ == "__main__":
    main()
