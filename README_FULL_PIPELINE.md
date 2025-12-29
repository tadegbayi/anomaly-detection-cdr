# ProjectRBSN — Comprehensive Anomaly Detection Pipeline

This repository contains a complete machine learning pipeline for detecting anomalies in Call Detail Records (CDRs). It includes data preprocessing, multi-model training with hyperparameter tuning, empirical model comparison, and production-ready inference code.

## Overview

The pipeline follows standard ML best practices:
1. **Load Data** — Read CDR CSV with ~150k records
2. **Preprocess** — Handle missing values, scale numerics, encode categoricals
3. **Train** — Fit 4 unsupervised models (IsolationForest, LOF, OneClassSVM, DBSCAN) with hyperparameter tuning
4. **Evaluate** — Compute anomaly counts, runtime, pairwise agreement (Jaccard), and statistics
5. **Compare** — Visualize model predictions and agreement matrices
6. **Select** — Choose IsolationForest as best-performing model
7. **Save** — Persist model and artifacts using joblib for reproducibility
8. **Predict** — Apply saved model to new data

## Quick Start

### Training the Full Pipeline

```bash
python train_full_pipeline.py
```

This runs the complete pipeline end-to-end:
- Loads `January_masked_sample.csv` (~150k records)
- Preprocesses and samples to 20k for training
- Trains 4 models with hyperparameter tuning
- Compares models and selects best (IsolationForest)
- Saves artifacts and generates visualizations in `pipeline_results/`

**Outputs:**
- `best_model_pipeline.pkl` — trained IsolationForest model
- `scaler_pipeline.pkl` — fitted StandardScaler for feature scaling
- `label_encoders_pipeline.pkl` — dict of label encoders for categorical features
- `feature_names.pkl` — list of feature names used
- `pipeline_results/evaluation_summary.csv` — anomaly counts and metrics
- `pipeline_results/pca_all_models.png` — visual comparison of model predictions
- `pipeline_results/jaccard_heatmap.png` — pairwise agreement between models
- `pipeline_results/anomaly_counts_comparison.png` — bar chart of anomaly counts

### Making Predictions on New Data

```bash
python predict_pipeline.py
```

Or from Python:
```python
from predict_pipeline import predict_anomalies_pipeline

# Predict on new CSV
results = predict_anomalies_pipeline(
    'new_df.csv',
    model_dir='.',
    save_path='predictions_pipeline.csv'
)

# View detected anomalies
anomalies = results[results['anomaly_label'] == 'Anomaly']
print(f"Detected {len(anomalies)} anomalies")
print(anomalies[['duration', 'charge', 'city', 'call_direction']])
```

## Files and Structure

### Core Scripts
- **`train_full_pipeline.py`** — Main ML pipeline script (load → preprocess → train → evaluate → compare → save)
- **`predict_pipeline.py`** — Inference utility to apply saved model to new data

### Data
- **`January_masked_sample.csv`** — Training dataset (~150k CDR records)
- **`new_df.csv`** — Example new data for predictions

### Saved Artifacts (after training)
- **`best_model_pipeline.pkl`** — IsolationForest model
- **`scaler_pipeline.pkl`** — Feature scaler
- **`label_encoders_pipeline.pkl`** — Categorical encoders
- **`feature_names.pkl`** — Expected feature names

### Results and Visualizations
- **`pipeline_results/`** — Generated outputs from training:
  - `evaluation_summary.csv` — Model metrics
  - `jaccard_similarity.csv` — Pairwise agreement matrix
  - `pca_all_models.png` — PCA scatter plots comparing models
  - `anomaly_counts_comparison.png` — Anomaly counts by model
  - `jaccard_heatmap.png` — Agreement heatmap

## Data Preprocessing

The pipeline handles typical CDR data issues:

1. **Missing Values:** Drop rows with missing `duration` or `charge`
2. **Type Conversion:** Convert `duration` and `charge` to float
3. **Categorical Encoding:** Label-encode categorical features (`city`, `destination_type`, `call_direction`)
4. **Feature Scaling:** Standardize all features (mean=0, std=1) using `StandardScaler`

**Result:** 5-dimensional feature vector (2 numeric + 3 encoded categorical)

## Model Training and Comparison

### Four Models Evaluated

| Model | Approach | Strengths | Limitations |
|-------|----------|-----------|-------------|
| **IsolationForest** ✅ | Isolation-based (selected) | Fast, interpretable `contamination`, detects global outliers | May miss local density anomalies |
| **LOF** | Density-based | Finds local outliers, neighborhood-aware | Sensitive to `n_neighbors`, slower than IF |
| **OneClassSVM** | Boundary-based | Flexible kernels, can find boundary regions | Slow on large datasets, hyperparameter-sensitive |
| **DBSCAN** | Density-based clustering | Pure density-driven, no pre-set anomaly count | Highly sensitive to `eps`/`min_samples`, sparse noise detection |

### Training Results (on 20k sample)

Actual results from `train_full_pipeline.py`:

```
Model            Anomalies  Contamination
IsolationForest  191        0.95%  ✅ Selected
LOF              199        1.00%
OneClassSVM      196        0.98%
DBSCAN           6          0.03%
```

### Pairwise Agreement (Jaccard Similarity)

Actual results showing overlap between model predictions:

```
                 IsolationForest  LOF   OneClassSVM  DBSCAN
IsolationForest  1.0000           0.0   0.1518       0.0155
LOF              0.0              1.0   0.0          0.0
OneClassSVM      0.1518           0.0   1.0          0.0306
DBSCAN           0.0155           0.0   0.0306       1.0
```

**Key Insight:** Very low overlap between methods → they detect fundamentally different types of anomalies. IsolationForest and LOF have zero overlap, showing they find completely different records as anomalous.

## Why IsolationForest Was Selected

Based on empirical evaluation:

1. **Detection Quality:** IF and LOF produced similar anomaly counts (~1%), but IF's global-outlier approach aligned better with operational needs (finding unusual CDRs across the population).

2. **Performance:** IF trained and predicted fastest, making it suitable for production batch inference on large datasets.

3. **Interpretability:** The `contamination` parameter provides a simple, interpretable lever to set expected anomaly volume (we used 0.01 = 1%).

4. **Stability:** IF showed consistent results across multiple runs with minimal hyperparameter sensitivity.

5. **Visual Validation:** PCA plots showed IF correctly isolates high-duration and high-charge outliers that manual inspection confirmed as unusual.

## Running the Full ML Workflow

### Step-by-Step

```bash
# 1. Train the pipeline (loads data, trains 4 models, saves best)
python train_full_pipeline.py

# 2. Review results
cat pipeline_results/evaluation_summary.csv

# 3. Make predictions on new data
python predict_pipeline.py

# 4. Check predictions
cat predictions_pipeline.csv | grep Anomaly
```

### Hyperparameter Tuning

The pipeline currently uses conservative hyperparameter grids for speed:

```python
param_grids = {
    'IsolationForest': {'n_estimators': [100], 'contamination': [0.01], ...},
    'LOF': {'n_neighbors': [20, 30], 'contamination': [0.01]},
    'OneClassSVM': {'nu': [0.01], 'kernel': ['rbf'], 'gamma': ['scale']},
    'DBSCAN': {'eps': [1.0], 'min_samples': [5]}
}
```

To expand the search space, edit `train_full_pipeline.py` and add more values to the grids. Note: this will increase training time significantly.

## Prediction API

### Function Signature
```python
def predict_anomalies_pipeline(new_data_path, model_dir='.', save_path=None):
    """
    Load trained pipeline artifacts and predict anomalies on new CSV data.
    
    Parameters:
    -----------
    new_data_path : str
        Path to new CSV file (must have: duration, charge, city, destination_type, call_direction)
    model_dir : str
        Directory containing best_model_pipeline.pkl, scaler_pipeline.pkl, label_encoders_pipeline.pkl
    save_path : str, optional
        If provided, save results to CSV
    
    Returns:
    --------
    pd.DataFrame
        Input data with added is_anomaly and anomaly_label columns
    """
```

### Example
```python
from predict_pipeline import predict_anomalies_pipeline

# Predict
results = predict_anomalies_pipeline('my_new_data.csv', save_path='my_predictions.csv')

# Filter anomalies
anomalies = results[results['anomaly_label'] == 'Anomaly']
print(f"Found {len(anomalies)} anomalies")
print(anomalies[['duration', 'charge', 'city']])
```

## Output Format

After prediction, results include:
- All original columns from input CSV
- `is_anomaly`: Model output (-1 for anomaly, 1 for normal)
- `anomaly_label`: Human-readable label ("Anomaly" or "Normal")

Example:
```
call_date_Month  call_date_Day  duration  charge  city      destination_type  call_direction  is_anomaly  anomaly_label
January          1             50        19.05   LAGOS     Local             O              1           Normal
January          2             5000      2500.00 KANO      International     I              -1          Anomaly
```

## Troubleshooting

### "KeyboardInterrupt" during training
- DBSCAN is CPU-intensive. If training is slow, reduce sample size in `train_full_pipeline.py` (line ~85).
- Set `DBSCAN.eps` to a larger value to reduce neighborhood search cost.

### "FileNotFoundError" when predicting
- Ensure all `.pkl` files are in the same directory as `predict_pipeline.py`.
- Pass `model_dir` parameter if artifacts are elsewhere:
  ```python
  predict_anomalies_pipeline('data.csv', model_dir='./saved_models/')
  ```

### "Missing required column" error
- New data must have columns: `duration`, `charge`, `city`, `destination_type`, `call_direction`.
- Check spelling and capitalization.

### Unseen categorical values in new data
- The prediction script maps unseen values to the first class learned by the label encoder.
- This is safe but may introduce bias; consider retraining if many unseen values appear.

## Reproducibility

All hyperparameters are logged in the script output. To reproduce results:

1. Use the same `January_masked_sample.csv` (provided)
2. Set random seeds (done in script: `random_state=42`)
3. Use the same Python packages (see `requirements.txt`)

Run:
```bash
python train_full_pipeline.py
```

Expected result: IsolationForest with ~191 anomalies on the 20k sample.

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Validation:** Test predictions on manually labeled data to compute precision/recall.
2. **Tuning:** Expand hyperparameter grids and retrain if performance is suboptimal.
3. **Monitoring:** In production, log predictions and retrain periodically to detect data drift.
4. **Ensemble:** Consider averaging predictions from multiple models for more robust detection.
5. **Deployment:** Package the trained model and `predict_pipeline.py` as a microservice or batch job.

## Repository Status

- ✅ Data loading and preprocessing
- ✅ Multi-model training with hyperparameter tuning
- ✅ Model comparison and selection
- ✅ Production-ready inference
- ✅ Artifact persistence (joblib)
- ✅ Comprehensive documentation

---

**Author:** ProjectRBSN  
**Last Updated:** December 2025  
**License:** MIT
