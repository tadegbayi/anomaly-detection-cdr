import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from run_unsupervised_compare import load_sample, prepare_features, run_all
import numpy as np

out = 'unsupervised_comparison'
os.makedirs(out, exist_ok=True)

df = load_sample()
Xs, Xraw, scaler = prepare_features(df)
results = run_all(Xs)

pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)

n_plot = min(3000, Xp.shape[0])
idx = np.random.RandomState(1).choice(Xp.shape[0], n_plot, replace=False)

methods = [k for k in results.keys() if not k.endswith('_time_s')]
fig, axes = plt.subplots(1, len(methods), figsize=(5 * max(len(methods),1), 4))
if len(methods) == 1:
    axes = [axes]
for ax, m in zip(axes, methods):
    anom = (results[m] == -1).astype(int)
    ax.scatter(Xp[idx, 0], Xp[idx, 1], c=anom[idx], cmap='coolwarm', s=8, alpha=0.7)
    ax.set_title(m)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_path = os.path.join(out, 'pca_unsupervised_compare.png')
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print('Saved PCA to', out_path)
