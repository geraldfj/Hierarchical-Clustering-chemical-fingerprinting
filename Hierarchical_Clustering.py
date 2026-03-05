# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 23:15:41 2026

@author: gzj0002
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# -----------------------------
# Settings
# -----------------------------
file_path = r"C:\Users\gzj0002\Box\Research\USS_Arizona\Biomarker_ratios.xlsx"
sheet_name = "Sheet1"
col_id = "Sample_id"

# Hierarchical clustering settings
LINKAGE_METHOD = "ward"        # "ward" (recommended with Euclidean), or "average", "complete"
DIST_METRIC = "euclidean"      # "euclidean" for ward; for others you can use "cityblock", "cosine", etc.
N_CLUSTERS = None                 # set to None if you want to cut by distance instead
CUT_DISTANCE = None            # e.g., 8.0 (used only if N_CLUSTERS is None)

# -----------------------------
# Load + clean data
# -----------------------------
df = pd.read_excel(file_path, sheet_name=sheet_name)
labels = df[col_id].astype(str).values

X = df.drop(columns=[col_id], errors="ignore").select_dtypes(include=[np.number]).copy()
X = X.dropna(axis=1, how="all")
mask_good = ~X.isna().any(axis=1)
X = X.loc[mask_good].reset_index(drop=True)
labels = labels[mask_good.values]

# Optional: group (Oil vs Sheen) for later coloring/inspection
def group_from_label(s: str) -> str:
    s = s.strip().upper()
    if re.match(r"^O\d+", s):
        return "Oil"
    if re.match(r"^S\d+", s):
        return "Sheen"
    return "Other"

groups = np.array([group_from_label(s) for s in labels])

# -----------------------------
# Standardize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Hierarchical clustering (linkage matrix)
# -----------------------------
Z = linkage(X_scaled, method=LINKAGE_METHOD, metric=DIST_METRIC)

# -----------------------------
# Dendrogram
# -----------------------------
plt.figure(figsize=(6, 6))
dendrogram(
    Z,
    labels=labels,
    orientation="right",
    leaf_rotation=360,
    leaf_font_size=9,
    color_threshold=None
)
plt.title(f"Hierarchical Clustering Dendrogram ({LINKAGE_METHOD}, {DIST_METRIC})")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# -----------------------------
# Assign cluster labels (either by number of clusters or by cut distance)
# -----------------------------
if N_CLUSTERS is not None:
    cluster_ids = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
elif CUT_DISTANCE is not None:
    cluster_ids = fcluster(Z, t=CUT_DISTANCE, criterion="distance")
else:
    raise ValueError("Set either N_CLUSTERS or CUT_DISTANCE.")

# -----------------------------
# Optional: visualize clusters on PCA (PC1 vs PC2)
# -----------------------------
pca = PCA(n_components=2, random_state=0)
PC = pca.fit_transform(X_scaled)
expl_var = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(9, 7))
for cid in np.unique(cluster_ids):
    idx = cluster_ids == cid
    plt.scatter(PC[idx, 0], PC[idx, 1], label=f"Cluster {cid}", alpha=0.85)

for x, y, lab in zip(PC[:, 0], PC[:, 1], labels):
    plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel(f"PC1 ({expl_var[0]:.1f}% var)")
plt.ylabel(f"PC2 ({expl_var[1]:.1f}% var)")
plt.title("PCA (PC1 vs PC2) colored by Hierarchical Clusters")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Output table (sample -> group -> cluster)
# -----------------------------
out = pd.DataFrame({"Sample_id": labels, "Group": groups, "Cluster": cluster_ids})
print(out.sort_values(["Cluster", "Sample_id"]).to_string(index=False))
