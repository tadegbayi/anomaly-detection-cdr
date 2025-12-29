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

## Training Results

### Actual Results (20k sample)

```
Model            Anomalies  Contamination
IsolationForest  191        0.95%  ✅ Selected
LOF              199        1.00%
OneClassSVM      196        0.98%
DBSCAN           6          0.03%
```

### Pairwise Agreement (Jaccard Similarity)

```
                 IsolationForest  LOF   OneClassSVM  DBSCAN
IsolationForest  1.0000           0.0   0.1518       0.0155
LOF              0.0              1.0   0.0          0.0
OneClassSVM      0.1518           0.0   1.0          0.0306
DBSCAN           0.0155           0.0   0.0306       1.0
```

**Key Insight:** Very low overlap between methods → they detect fundamentally different types of anomalies. IsolationForest and LOF have zero overlap, showing they find completely different records as anomalous.

## Why IsolationForest Was Selected

1. **Detection Quality:** IF flagged 191 anomalies (0.95% contamination) with minimal overlap with other methods. The zero overlap with LOF shows IF captures fundamentally different (global) outliers.

2. **Performance:** IF trained fast and produced consistent, interpretable results. OneClassSVM was much slower; DBSCAN flagged only 6 anomalies, indicating oversensitivity to parameters.

3. **Operational Control:** The `contamination` parameter is intuitive and controllable, allowing easy adjustment of expected anomaly volume.

4. **Runtime:** Fast fit/predict makes it suitable for production batch inference.

## Alternative Models for Specific Use Cases

- **LOF:** Use for complementary local-density checks (100% different anomalies than IF)
- **DBSCAN:** Reconfigure `eps`/`min_samples` for density-based clustering on targeted subsets
- **OneClassSVM:** Useful if kernel-based boundaries are required; requires careful hyperparameter tuning

## Files and Structure

### Core Scripts
- `train_full_pipeline.py` — Main ML pipeline script
- `predict_pipeline.py` — Inference utility

### Data
- `January_masked_sample.csv` — Training dataset (~150k CDR records)
- `new_df.csv` — Example new data for predictions

### Saved Artifacts (after training)
- `best_model_pipeline.pkl` — IsolationForest model
- `scaler_pipeline.pkl` — Feature scaler
- `label_encoders_pipeline.pkl` — Categorical encoders
- `feature_names.pkl` — Expected feature names

### Results and Visualizations
- `pipeline_results/` — Generated outputs from training:
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

## Troubleshooting

### "KeyboardInterrupt" during training
- DBSCAN is CPU-intensive. Reduce sample size in `train_full_pipeline.py` (line ~85).
- Set `DBSCAN.eps` to a larger value to reduce neighborhood search cost.

### "FileNotFoundError" when predicting
- Ensure all `.pkl` files are in the same directory as `predict_pipeline.py`.
- Pass `model_dir` parameter if artifacts are elsewhere.

### Unseen categorical values in new data
- The prediction script maps unseen values to the first class learned by the label encoder.
- This is safe but may introduce bias; consider retraining if many unseen values appear.

## Reproducibility

To reproduce results:
1. Use the same `January_masked_sample.csv`
2. Random seeds are set (random_state=42)
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

## Project Status

- ✅ Data loading and preprocessing (139,607 → 20,000 sample)
- ✅ Multi-model training with hyperparameter tuning (4 algorithms)
- ✅ Model comparison and selection (empirical evaluation)
- ✅ Production-ready inference
- ✅ Artifact persistence (joblib)
- ✅ Comprehensive documentation and visualizations

## Next Steps

1. Validation: Test predictions on manually labeled data
2. Tuning: Expand hyperparameter grids and retrain
3. Monitoring: Log predictions and retrain periodically
4. Ensemble: Average predictions from multiple models
5. Deployment: Package as microservice or batch job

---

**Last Updated:** December 2025  
**Repository:** https://github.com/tadegbayi/anomaly-detection-cdr  
**License:** MIT
