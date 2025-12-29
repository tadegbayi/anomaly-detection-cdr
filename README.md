# ProjectRBSN — Anomaly Detection for CDRs

This repository contains a complete machine learning pipeline for detecting anomalies in Call Detail Records (CDRs), including data preprocessing, multi-model training, empirical comparison, and production inference.

## Quick Start

### Full Pipeline (Recommended)

For a comprehensive ML workflow covering data load → preprocess → train 4 models → evaluate → select best → save:

```bash
python train_full_pipeline.py
```

This runs the complete pipeline and saves all artifacts. See [README_FULL_PIPELINE.md](README_FULL_PIPELINE.md) for full documentation.

Then predict on new data:
```bash
python predict_pipeline.py
```

### Summary of Files

**Core Pipeline:**
- `train_full_pipeline.py` — Complete ML workflow (load, preprocess, train 4 models, evaluate, save)
- `predict_pipeline.py` — Inference utility for new data

**Data:**
- `January_masked_sample.csv` — Training dataset (~150k CDR records, 13 columns)
- `new_df.csv` — Example new data for predictions

**Saved Artifacts (after training):**
- `best_model_pipeline.pkl` — IsolationForest model
- `scaler_pipeline.pkl` — StandardScaler
- `label_encoders_pipeline.pkl` — Categorical encoders
- `feature_names.pkl` — Feature names

**Results:**
- `pipeline_results/` — Evaluation metrics, visualizations, and comparison matrices

## Full ML Workflow (Step by Step)

The `train_full_pipeline.py` script implements:

1. **Load Data** — Read January_masked_sample.csv
2. **Preprocess** — Handle missing values, scale numerics, encode categoricals
3. **Train** — Fit IsolationForest, LOF, OneClassSVM, DBSCAN with hyperparameter tuning
4. **Evaluate** — Compute anomaly counts, runtime, pairwise agreement (Jaccard)
5. **Compare** — Generate PCA, heatmap, and count visualizations
6. **Select Best** — IsolationForest chosen based on empirical metrics
7. **Save** — Persist model and preprocessing artifacts using joblib
8. **Predict** — Apply to new data

## Model Selection: Why IsolationForest?

| Model | Anomalies | Runtime | Performance | Reason |
|-------|-----------|---------|-------------|--------|
| **IsolationForest** ✅ | 191 | Fast | Global outliers | **Selected:** Best balance of speed, interpretability, and detection quality |
| LOF | 199 | Medium | Local outliers | Good alternative for density-based checks |
| OneClassSVM | 196 | Slow | Boundary-based | Useful with kernel tuning but slower |
| DBSCAN | 6 | Slow | Density clustering | Sensitive to parameters, sparse on this data |

**Key Insight:** Pairwise Jaccard similarity is near 0, meaning each model detects fundamentally different anomalies. IsolationForest's global-outlier approach aligns best with finding unusual CDRs.

## Training Results (20k Sample)

```
Model          Anomalies  Contamination
IsolationForest   191        0.95%
LOF               199        1.00%
OneClassSVM       196        0.98%
DBSCAN              6        0.03%
```

Pairwise Agreement (Jaccard Similarity):
```
Models          Agreement
IF vs LOF       0.00 (no overlap)
IF vs OCSVM     0.152 (15% overlap)
IF vs DBSCAN    0.016 (2% overlap)
```

## Making Predictions

```python
from predict_pipeline import predict_anomalies_pipeline

# Predict on new CSV
results = predict_anomalies_pipeline('new_df.csv', save_path='predictions.csv')

# View anomalies
print(results[results['anomaly_label'] == 'Anomaly'])
```

Output includes original data plus:
- `is_anomaly`: -1 (anomaly) or 1 (normal)
- `anomaly_label`: "Anomaly" or "Normal"

## Data Preprocessing

The pipeline:
- Loads CSV and converts numeric columns
- Drops rows with missing `duration` or `charge`
- Label-encodes 3 categorical features (`city`, `destination_type`, `call_direction`)
- Standardizes all features using StandardScaler
- Samples to 20k rows for faster training

Result: 5D feature vector (2 numeric + 3 categorical)

## Project Structure

```
ProjectRBSN/
├── train_full_pipeline.py         # Main ML pipeline (load → train → evaluate → save)
├── predict_pipeline.py             # Inference on new data
├── January_masked_sample.csv       # Training data (~150k records)
├── new_df.csv                      # Example new data
├── best_model_pipeline.pkl         # Trained IsolationForest
├── scaler_pipeline.pkl             # Feature scaler
├── label_encoders_pipeline.pkl     # Categorical encoders
├── feature_names.pkl               # Feature list
├── pipeline_results/               # Evaluation, plots, CSVs
│   ├── evaluation_summary.csv
│   ├── jaccard_similarity.csv
│   ├── pca_all_models.png
│   ├── anomaly_counts_comparison.png
│   └── jaccard_heatmap.png
├── README.md                       # This file
└── README_FULL_PIPELINE.md         # Detailed documentation
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Install:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**"KeyboardInterrupt" during DBSCAN training?**
- DBSCAN is CPU-intensive. Edit `train_full_pipeline.py` to reduce sample size or increase DBSCAN `eps`.

**"FileNotFoundError" during prediction?**
- Ensure `.pkl` files are in the same directory. Pass `model_dir` if they're elsewhere.

**Unseen categorical values in new data?**
- They're safely mapped to the first learned class. Consider retraining if this happens frequently.

## Full Documentation

**Notes & recommendations**
- File paths: use forward slashes (`/`) or raw strings on Windows to avoid escape issues.
- Encoding: `predict_new_data` maps unseen categorical values to the first class learned by the `LabelEncoder`. Consider updating this behavior if you prefer a different fallback (e.g., `Unknown` or retraining encoders).
- Missing columns in new data will raise an error — ensure `duration`, `charge`, `city`, `destination_type`, `call_direction` exist.
- If you re-run training with a different preprocessing pipeline, remember to update the saved artifacts and `model_dir` used by `predict_new_data`.

**Exploratory Data Analysis (EDA)**
The following EDA steps and visualizations are useful to understand the data distribution and spot obvious anomalies before modeling.

- Univariate distributions (histograms / KDE) for numeric features such as `duration` and `charge` to inspect skew and heavy tails.
- Boxplots per `city` or `destination_type` to find groups with outlying behavior.
- Countplots for categorical features (`city`, `destination_type`, `call_direction`) to check class imbalance.
- Correlation matrix / heatmap for numeric features to check multicollinearity before scaling.
- PCA scatterplot (2 components) colored by model labels — useful for visual comparison (one example output: `anomaly_comparison_local.png`).

Example snippets (run in a Python REPL or notebook):
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('January_masked_sample.csv')
# histogram for duration
sns.histplot(df['duration'].astype(float), bins=80, kde=True)
plt.savefig('eda_duration_hist.png')

# boxplot of charge by city
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='city', y='charge')
plt.xticks(rotation=45)
plt.savefig('eda_charge_by_city.png')

# correlation heatmap
num_cols = ['duration','charge']
sns.heatmap(df[num_cols].corr(), annot=True)
plt.savefig('eda_corr.png')
```

Collect and save these figures in a folder (e.g., `eda/`) and include them in reports shared with teammates.

Preview of generated EDA figures (saved in `eda/`):

![Duration distribution](eda/eda_duration_hist.png)

![Charge distribution](eda/eda_charge_hist.png)

![Charge by top cities](eda/eda_charge_by_city.png)

![Call direction counts](eda/eda_call_direction_counts.png)

![Numeric correlation heatmap](eda/eda_corr.png)

![PCA of features colored by duration](eda/eda_pca_duration.png)


**Model Selection — Why IsolationForest was chosen**
We evaluated three approaches in `anomaly_detection.py`: `IsolationForest`, `LocalOutlierFactor` (LOF), and `OneClassSVM` (OCSVM). The final decision to use `IsolationForest` for production predictions was based on the following considerations:

- **Detection goal (global vs local outliers):** IsolationForest is designed for global outliers which match our operational goal of finding unusual call records across the dataset. LOF identifies local density anomalies which can be useful for fine-grained, neighborhood-level detection but is more sensitive to parameter `n_neighbors` and duplicate values.
- **Scalability and performance:** IsolationForest scales well to larger datasets and has faster fit/predict times than OCSVM. OCSVM can be computationally expensive and sensitive to kernel/hyperparameters (`gamma`, `nu`) on large, noisy CDR datasets.
- **Robustness to feature scaling:** After consistent scaling, IsolationForest performs reliably across numeric and encoded categorical features; OCSVM requires careful kernel tuning and LOF can be affected by duplicate rows (the implementation warns about duplicates).
- **Interpretability & control:** IsolationForest supports a `contamination` parameter (estimated fraction of outliers) which provided an easy, interpretable lever during experiments. This matched our operational requirement to set an expected anomaly budget (we used `contamination=0.01` in experiments).
- **Empirical evaluation:** We compared models by:
	- Visual inspection of PCA scatterplots colored by predicted labels (see `anomaly_comparison_local.png`).
	- Checking the number of flagged records and reviewing samples to validate that flagged records correspond to unusual durations/charges or unlikely category combinations.
	- Observing runtime/memory characteristics on the sample subset.

In short, IsolationForest offered the best trade-off between detection quality for global outliers, runtime performance, and operational control. LOF remains available in the repo for local-density checks and further experimentation; OCSVM can be revisited with smaller feature sets or different kernels if boundary-based detection is later required.

**Empirical Model Comparison**
We ran a side-by-side experiment across contamination rates (0.1%, 0.5%, 1%, 2%, 5%) on a 20,000-record sample. The results show:
- **Anomaly counts:** IsolationForest and LOF produce nearly identical counts across all contamination levels, while OneClassSVM consistently flags more records (e.g., 296 vs. 199 for IF at 1% contamination).
- **Agreement (Jaccard similarity):** Very low overlap between IF and LOF predictions (near 0), indicating they find fundamentally different sets of outliers. OneClassSVM overlaps more with IF as contamination increases (up to ~37% at 5%), but finds a distinct boundary-based set.
- **Visual patterns (PCA):** IsolationForest highlights isolated global outliers; LOF surfaces local density anomalies; OneClassSVM identifies boundary regions. The choice of IF aligns with our goal of detecting unusual call records across the population.

Embedded model-comparison figures (saved in `model_comparison/`):

![Anomalies detected vs contamination](model_comparison/anomaly_counts_vs_contamination.png)

![Pairwise Jaccard similarity vs contamination](model_comparison/jaccard_vs_contamination.png)

![PCA comparison (contamination=0.01)](model_comparison/pca_models_cont_0.01.png)

**Four-Model Evaluation (DBSCAN, IsolationForest, LOF, OneClassSVM)**
We performed a focused comparison of four unsupervised methods on the same sampled data to evaluate runtime, anomaly counts, pairwise agreement, and simple anomaly statistics (mean `duration` / `charge` among flagged records).

- **Procedure:** fit all four models on the same sample (downsampled for heavy methods), record wall-clock fit/predict time, count flagged anomalies, compute pairwise Jaccard similarity of anomaly sets, and inspect PCA visualizations.
- **Findings:**
	- IsolationForest (IF) and LOF often produce similar overall counts for a given contamination setting, but they frequently flag different individual records (low Jaccard overlap) — LOF finds local-density outliers while IF finds globally isolated points.
	- OneClassSVM tended to flag a larger and less stable set of anomalies without careful kernel/nu tuning.
	- DBSCAN can find noise points but is sensitive to `eps`/`min_samples` and is computationally expensive on higher-dimensional, larger samples; it is more useful for small, density-driven investigations.
	- Runtime: IF was the fastest and most predictable on medium samples; OCSVM was the slowest; LOF/DBSCAN costs grow with neighborhood computations.

- **Artifacts:** see `model_comparison/` for contamination sweeps and PCA comparisons; when generated, `unsupervised_comparison/` contains per-model counts, timings and pairwise Jaccard matrices for the DBSCAN+IF+LOF+OCSVM runs.

**Final recommendation**
For production use on these CDRs we selected **IsolationForest**. Rationale:

- Balanced detection: isolates global outliers that aligned best with manual checks in our samples.
- Performance and scalability: faster and more predictable than OneClassSVM, and less sensitive to neighborhood parameters than LOF/DBSCAN.
- Operational control: `contamination` provides a simple, interpretable way to set expected anomaly volume.

LOF and DBSCAN remain useful as complementary checks (local/density anomalies). OCSVM can be revisited with tighter feature sets or more tuning.


**Troubleshooting**
- Unicode/escape errors when specifying Windows paths: use `C:/path/to/file.csv` or prefix with `r"C:\path\to\file.csv"`.
- `FileNotFoundError` when loading model artifacts: ensure the `.pkl` files are present in `model_dir` or pass the correct `model_dir` to `predict_anomalies()`.
- If LOF warns about duplicate values, increase `n_neighbors` in `anomaly_detection.py`.


