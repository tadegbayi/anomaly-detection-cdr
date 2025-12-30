# ProjectRBSN — Anomaly Detection for CDRs

This repository contains a complete machine learning pipeline for detecting anomalies in Call Detail Records (CDRs), including data preprocessing, multi-model training with hyperparameter tuning, empirical comparison, and production-ready inference.

## Quick Start

### Training the Full Pipeline

```bash
python train_full_pipeline.py
```

This runs the complete pipeline end-to-end:
- Loads `January_masked_sample.csv` (~150k records)
- Preprocesses and samples to 20k for training
- Trains 4 models (IsolationForest, LOF, OneClassSVM, DBSCAN) with hyperparameter tuning
- Compares models and selects best (IsolationForest)
- Saves artifacts and generates visualizations in `pipeline_results/`

**Outputs:**
- `best_model_pipeline.pkl` — trained IsolationForest model
- `scaler_pipeline.pkl` — fitted StandardScaler
- `label_encoders_pipeline.pkl` — categorical encoders
- `feature_names.pkl` — feature names used
- `pipeline_results/evaluation_summary.csv` — anomaly counts and metrics
- `pipeline_results/pca_all_models.png` — visual comparison of models
- `pipeline_results/jaccard_heatmap.png` — pairwise agreement matrix
- `pipeline_results/anomaly_counts_comparison.png` — anomaly counts by model

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

## ML Workflow Overview

The `train_full_pipeline.py` script implements a complete 8-step ML workflow:

1. **Load Data** — Read January_masked_sample.csv (~150k records)
2. **Preprocess** — Handle missing values, convert types, scale numerics, encode categoricals
3. **Sample** — Downsample to 20k for training speed
4. **Train** — Fit 4 unsupervised models with hyperparameter tuning
5. **Evaluate** — Compute anomaly counts, runtime, pairwise agreement (Jaccard)
6. **Compare** — Visualize predictions and agreement matrices
7. **Select** — Choose IsolationForest as best-performing model
8. **Predict** — Apply saved model to new data

## Files and Structure

## Why Unsupervised Learning

This project uses unsupervised anomaly detection because the available CDR dataset does not include a label or target column that identifies anomalies. Supervised methods require labeled examples of "normal" vs "anomalous" calls to learn a decision boundary — labels that are costly or impractical to obtain for large telecom datasets. Unsupervised approaches detect unusual or rare patterns directly from feature distributions (or density/structure) without ground-truth anomaly labels. This makes them well suited for exploratory detection, early-warning systems, and cases where anomalies are rare, evolving, or unknown ahead of time.

Key practical reasons:
- No labeled anomalies in `January_masked_sample.csv` (unsupervised needed).
- Anomaly labeling at scale requires manual review or domain-driven heuristics.
- Unsupervised algorithms provide fast, low-friction ways to surface candidate anomalies for later human validation or semi-supervised workflows.

## Why These Four Algorithms

We selected four representative algorithms to cover complementary anomaly-detection paradigms and practical trade-offs:

- **IsolationForest (global isolation):** fast, scalable, and interpretable via `contamination`. Good for finding global outliers that are isolated in feature space; suitable for production batch scoring.
- **Local Outlier Factor (LOF, local density):** detects local density anomalies (points that are unusual relative to neighbors). Useful when anomalies are context-dependent (neighborhood-based).
- **OneClassSVM (boundary-based):** models a decision boundary around the majority class using kernel methods. Offers a different, boundary-focused perspective useful for complex, nonlinear separations.
- **DBSCAN (density-based clustering + noise):** finds dense clusters and labels low-density points as noise. Useful when anomalies are isolated noise relative to well-formed clusters.

Why these four and not every method?
- They represent core methodological families (isolation, local density, boundary, clustering/density), giving complementary views of "anomaly." Comparing them highlights how different definitions of anomaly affect results.
- All four are available in scikit-learn (no heavy external dependencies) and are easy to serialize/operate in production.
- They span a useful performance spectrum: `IsolationForest` (fast/production-ready) → `LOF`/`OneClassSVM` (neighborhood/boundary-focused) → `DBSCAN` (cluster/noise detection).
- More advanced options (deep autoencoders, LSTM-based sequence models, supervised ensembles) were intentionally deferred because they require labeled data, more compute, or significant engineering for this initial baseline.


### Core Scripts
- **`train_full_pipeline.py`** — Main ML pipeline (load → preprocess → train → evaluate → save)
- **`predict_pipeline.py`** — Inference utility for new data

### Data
- **`January_masked_sample.csv`** — Training dataset (~150k CDR records, 13 columns)
- **`new_df.csv`** — Example new data for predictions

### Saved Artifacts (after training)
- **`best_model_pipeline.pkl`** — IsolationForest model
- **`scaler_pipeline.pkl`** — StandardScaler for feature scaling
- **`label_encoders_pipeline.pkl`** — Categorical encoders dict
- **`feature_names.pkl`** — Feature list

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
3. **Categorical Encoding:** Label-encode 3 categorical features (`city`, `destination_type`, `call_direction`)
4. **Feature Scaling:** Standardize all features (mean=0, std=1) using StandardScaler
5. **Sampling:** Downsample 150k records to 20k for manageable training

**Result:** 5-dimensional feature vector (2 numeric + 3 encoded categorical)

## Model Training and Comparison

### Four Models Evaluated

| Model | Approach | Anomalies* | Strengths | Limitations |
|-------|----------|------------|-----------|-------------|
| **IsolationForest** ✅ | Isolation-based | 191 | Fast, interpretable, detects global outliers | May miss local density anomalies |
| **LOF** | Density-based | 199 | Finds local outliers, neighborhood-aware | Sensitive to `n_neighbors`, slower than IF |
| **OneClassSVM** | Boundary-based | 196 | Flexible kernels, boundary detection | Slow on large datasets, hyperparameter-sensitive |
| **DBSCAN** | Density clustering | 6 | Pure density-driven, pure noise detection | Highly sensitive to `eps`/`min_samples` |

*Results on 20k sample at ~1% contamination

### Training Results (20k sample)

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

Based on empirical evaluation across multiple dimensions:

1. **Detection Quality:** IF and LOF produced similar anomaly counts (~1%), but IF's global-outlier approach aligned better with operational needs (finding unusual CDRs across the population).

2. **Performance & Scalability:** IF trained fastest and predicted consistently. OneClassSVM was much slower; DBSCAN flagged only 6 anomalies, indicating parameter oversensitivity.

3. **Operational Control:** The `contamination` parameter is intuitive and easily adjustable for setting expected anomaly volume (we used `contamination=0.01` = 1%).

4. **Interpretability:** IF correctly isolates high-duration and high-charge outliers that manual inspection confirmed as unusual.

5. **Production Suitability:** Fast fit/predict makes it ideal for batch inference on large datasets.

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
        Directory containing model artifacts (default: current directory)
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

### Output Format

After prediction, results include all original columns plus:
- `is_anomaly`: Model output (-1 for anomaly, 1 for normal)
- `anomaly_label`: Human-readable label ("Anomaly" or "Normal")

Example:
```
call_date_Month  call_date_Day  duration  charge  city      destination_type  is_anomaly  anomaly_label
January          1             50        19.05   LAGOS     Local              1           Normal
January          2             5000      2500.00 KANO      International     -1          Anomaly
```

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

The pipeline uses conservative hyperparameter grids for speed. To expand the search space, edit `train_full_pipeline.py` and add more values to the grids:

```python
param_grids = {
    'IsolationForest': {'n_estimators': [100, 200], 'contamination': [0.01, 0.02], ...},
    'LOF': {'n_neighbors': [20, 30, 40], 'contamination': [0.01, 0.02]},
    'OneClassSVM': {'nu': [0.01, 0.05], 'kernel': ['rbf', 'sigmoid'], 'gamma': ['scale', 'auto']},
    'DBSCAN': {'eps': [0.5, 1.0, 2.0], 'min_samples': [5, 10]}
}
```

Note: This will increase training time significantly.

## Troubleshooting

### "KeyboardInterrupt" during training
- DBSCAN is CPU-intensive. Reduce sample size in `train_full_pipeline.py` (around line 85).
- Increase DBSCAN `eps` parameter to reduce neighborhood search cost.

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
- This is safe but may introduce bias; consider retraining if many unseen values appear frequently.

### Unicode/escape errors on Windows
- Use forward slashes: `C:/path/to/file.csv`
- Or use raw strings: `r"C:\path\to\file.csv"`

## Reproducibility

All hyperparameters are logged in script output. To reproduce results:

1. Use the same `January_masked_sample.csv` (provided)
2. Random seeds are set (random_state=42 throughout)
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

## Exploratory Data Analysis (Optional)

Before running the full pipeline, you can explore the data with these snippets:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('January_masked_sample.csv')

# Histogram for duration distribution
sns.histplot(df['duration'].astype(float), bins=80, kde=True)
plt.savefig('eda_duration_hist.png')

# Boxplot of charge by city
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='city', y='charge')
plt.xticks(rotation=45)
plt.savefig('eda_charge_by_city.png')

# Correlation heatmap
num_cols = ['duration', 'charge']
sns.heatmap(df[num_cols].corr(), annot=True)
plt.savefig('eda_corr.png')
```

## Next Steps

1. **Validation:** Test predictions on manually labeled data to compute precision/recall
2. **Tuning:** Expand hyperparameter grids and retrain if needed
3. **Monitoring:** Log predictions and retrain periodically to detect data drift
4. **Ensemble:** Consider averaging predictions from multiple models
5. **Deployment:** Package as microservice or batch job

## Alternative Models for Specific Use Cases

- **LOF:** Use for complementary local-density checks (100% different anomalies than IF)
- **DBSCAN:** Reconfigure `eps`/`min_samples` for density-based clustering on targeted subsets
- **OneClassSVM:** Useful if kernel-based boundaries are required; requires careful tuning

## Project Status

- ✅ Data loading and preprocessing
- ✅ Multi-model training with hyperparameter tuning
- ✅ Model comparison and empirical selection
- ✅ Production-ready inference
- ✅ Artifact persistence (joblib)
- ✅ Comprehensive documentation

---

**Last Updated:** December 2025
