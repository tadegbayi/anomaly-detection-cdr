import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def load_sample(path='January_masked_sample.csv', n_samples=3000, random_state=42):
    df = pd.read_csv(path)
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=random_state).copy()
    # clean numeric columns
    if 'duration' in df.columns:
        df['duration'] = df['duration'].astype(str).str.replace(',', '', regex=False)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    if 'charge' in df.columns:
        df['charge'] = pd.to_numeric(df['charge'], errors='coerce')
    return df


def prepare_features(df, features=None):
    if features is None:
        features = ['duration', 'charge', 'city', 'destination_type', 'call_direction']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[features].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, X, scaler


def run_all(Xs):
    results = {}

    import time

    # IsolationForest
    t0 = time.time()
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso_labels = iso.fit_predict(Xs)
    results['IsolationForest'] = iso_labels
    results['IsolationForest_time_s'] = time.time() - t0


    # LOF
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    lof_labels = lof.fit_predict(Xs)
    results['LOF'] = lof_labels
    results['LOF_time_s'] = time.time() - t0


    # OneClassSVM
    t0 = time.time()
    oc = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
    oc_labels = oc.fit_predict(Xs)
    results['OneClassSVM'] = oc_labels
    results['OneClassSVM_time_s'] = time.time() - t0

    # DBSCAN (density-based). DBSCAN labels: -1 = noise -> treat as anomaly
    # Use eps chosen for scaled data; default eps=0.5, min_samples=5
    # DBSCAN (choose eps conservatively for scaled data)
    t0 = time.time()
    db = DBSCAN(eps=0.6, min_samples=5)
    db_labels = db.fit_predict(Xs)
    # convert DBSCAN cluster labels to anomaly labels: -1 -> -1, else 1
    db_anom = np.where(db_labels == -1, -1, 1)
    results['DBSCAN'] = db_anom
    results['DBSCAN_time_s'] = time.time() - t0

    return results


def summarize_and_plot(df, Xs, results, out='unsupervised_comparison'):
    os.makedirs(out, exist_ok=True)


    # Counts and timing
    rows = []
    methods = [k for k in results.keys() if not k.endswith('_time_s')]
    for name in methods:
        labels = results[name]
        time_s = results.get(f"{name}_time_s", None)
        rows.append({'method': name, 'n_anomalies': int((labels == -1).sum()), 'n_total': len(labels), 'time_s': time_s})
    df_counts = pd.DataFrame(rows)
    df_counts.to_csv(os.path.join(out, 'anomaly_counts.csv'), index=False)

    # Compute anomaly stats (duration/charge) for each method
    stats = []
    for name in methods:
        labels = results[name]
        mask = (labels == -1)
        if 'duration' in df.columns:
            mean_dur = float(df.loc[mask, 'duration'].mean()) if mask.sum() > 0 else None
            mean_charge = float(df.loc[mask, 'charge'].mean()) if mask.sum() > 0 else None
        else:
            mean_dur = None
            mean_charge = None
        stats.append({'method': name, 'n_anomalies': int(mask.sum()), 'mean_duration': mean_dur, 'mean_charge': mean_charge})
    pd.DataFrame(stats).to_csv(os.path.join(out, 'anomaly_stats.csv'), index=False)

    # Pairwise Jaccard
    pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            a = results[methods[i]]
            b = results[methods[j]]
            pairs.append({'pair': f"{methods[i]}__{methods[j]}", 'jaccard': jaccard(a, b)})
    pd.DataFrame(pairs).to_csv(os.path.join(out, 'pairwise_jaccard.csv'), index=False)

    # PCA for visual comparison (downsample if large)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    n_plot = min(3000, Xp.shape[0])
    idx = np.random.RandomState(1).choice(Xp.shape[0], n_plot, replace=False)

    methods = list(results.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        anom = (results[m] == -1).astype(int)
        sc = ax.scatter(Xp[idx, 0], Xp[idx, 1], c=anom[idx], cmap='coolwarm', s=8, alpha=0.7)
        ax.set_title(m)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'pca_unsupervised_compare.png'), bbox_inches='tight')
    plt.close()

    print('Saved counts and PCA plot to', out)


def main():
    df = load_sample()
    Xs, Xraw, scaler = prepare_features(df)
    results = run_all(Xs)
    summarize_and_plot(df, Xs, results)


if __name__ == '__main__':
    main()
