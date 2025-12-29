"""
Prediction helper for the full ML pipeline.
Load the trained best model and make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os


def predict_anomalies_pipeline(new_data_path, model_dir='.', save_path=None):
    """
    Load trained pipeline artifacts and predict anomalies on new CSV data.
    
    Parameters:
    -----------
    new_data_path : str
        Path to new CSV file with same structure as training data
    model_dir : str
        Directory containing best_model_pipeline.pkl, scaler_pipeline.pkl, label_encoders_pipeline.pkl
    save_path : str, optional
        If provided, save results to CSV at this path
    
    Returns:
    --------
    pd.DataFrame
        Input data with added columns: is_anomaly (-1=anomaly, 1=normal) and anomaly_label (Anomaly/Normal)
    """
    
    print(f"\n[PREDICTION] Loading data from {new_data_path}...")
    df = pd.read_csv(new_data_path)
    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Load model artifacts
    model_path = os.path.join(model_dir, 'best_model_pipeline.pkl')
    scaler_path = os.path.join(model_dir, 'scaler_pipeline.pkl')
    encoders_path = os.path.join(model_dir, 'label_encoders_pipeline.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    
    print(f"\n[PREDICTION] Loading artifacts...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    feature_names = joblib.load(features_path)
    print(f"  ✓ Loaded model, scaler, label encoders, and feature names")
    print(f"  Features expected: {feature_names}")
    
    # Preprocess new data
    print(f"\n[PREDICTION] Preprocessing new data...")
    
    # Convert numeric columns
    numeric_cols = ['duration', 'charge']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for required columns
    categorical_cols = [c for c in feature_names if c not in numeric_cols]
    for col in feature_names:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Label encode categorical features
    df_processed = df.copy()
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Map unseen values to first class
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else le.transform([le.classes_[0]])[0]
            )
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Fill any NaNs
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    print(f"  ✓ Preprocessed {df_processed.shape[0]} records")
    
    # Scale features
    X = scaler.transform(df_processed[feature_names])
    print(f"  ✓ Scaled features shape: {X.shape}")
    
    # Make predictions
    print(f"\n[PREDICTION] Running inference...")
    predictions = model.predict(X)
    
    # Map to labels
    df['is_anomaly'] = predictions
    df['anomaly_label'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    n_anomalies = (predictions == -1).sum()
    print(f"  ✓ Detected {n_anomalies} anomalies out of {len(predictions)} records ({100*n_anomalies/len(predictions):.2f}%)")
    
    # Optionally save results
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n[PREDICTION] Saved predictions to: {save_path}")
    
    return df


if __name__ == '__main__':
    # Example usage
    results = predict_anomalies_pipeline('new_df.csv', model_dir='.', save_path='predictions_pipeline.csv')
    print("\n[SUMMARY]")
    print(f"Results shape: {results.shape}")
    print(f"\nSample anomalies detected:")
    print(results[results['anomaly_label'] == 'Anomaly'][['duration', 'charge', 'city', 'call_direction', 'anomaly_label']].head())
