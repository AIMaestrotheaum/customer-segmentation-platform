"""
Customer Segmentation - Cluster Evaluation

Tests K-Means for K = 2 through K = 10.

Metrics:
    - Inertia
    - Silhouette Score
    - Smallest Cluster
    - Largest Cluster
    - Smallest Cluster %
    - Largest Cluster %

A sample is used for silhouette calculation because
the dataset contains 1,000,000 customers.
"""

import os

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "../outputs/clustering_features.csv"
OUTPUT_PATH = "../outputs/k_selection_results.csv"

K_VALUES = range(2, 11)

SILHOUETTE_SAMPLE_SIZE = 10_000

RANDOM_STATE = 42

N_INIT = 10


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("CUSTOMER SEGMENTATION — CLUSTER EVALUATION")
print("=" * 70)

print("\nLoading clustering features...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_PATH}\n\n"
        "Run preprocessing.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# Expected features
# ============================================================

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary_Value",
    "Average_Order_Value",
    "Engagement_Score",
    "Discount_Dependency",
    "Return_Rate",
    "Online_Purchase_Ratio",
    "InStore_Purchase_Ratio",
    "Avg_Items_Per_Transaction",
    "Support_Interaction_Rate"
]


missing_features = [
    column
    for column in FEATURES
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing clustering features: "
        f"{missing_features}"
    )


# ============================================================
# Prepare X
# ============================================================

X = df[FEATURES].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)

print("Features:", len(FEATURES))


# ============================================================
# Standardization
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling completed.")


# ============================================================
# Silhouette sample
# ============================================================

sample_size = min(
    SILHOUETTE_SAMPLE_SIZE,
    len(X_scaled)
)

rng = np.random.RandomState(
    RANDOM_STATE
)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

print(
    f"Silhouette sample size: {sample_size:,}"
)


# ============================================================
# Evaluate K values
# ============================================================

results = []

print("\nTesting K values...")

for k in K_VALUES:

    print(f"\nK = {k}")

    model = KMeans(
        n_clusters=k,
        n_init=N_INIT,
        random_state=RANDOM_STATE
    )

    labels = model.fit_predict(
        X_scaled
    )

    # --------------------------------------------------------
    # Inertia
    # --------------------------------------------------------

    inertia = model.inertia_

    # --------------------------------------------------------
    # Silhouette
    # --------------------------------------------------------

    sample_labels = labels[
        sample_indices
    ]

    silhouette = silhouette_score(
        X_sample,
        sample_labels
    )

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    cluster_counts = pd.Series(
        labels
    ).value_counts()

    smallest_cluster = int(
        cluster_counts.min()
    )

    largest_cluster = int(
        cluster_counts.max()
    )

    smallest_percentage = (
        smallest_cluster
        / len(labels)
        * 100
    )

    largest_percentage = (
        largest_cluster
        / len(labels)
        * 100
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "K": k,

        "Inertia": inertia,

        "Silhouette_Score": silhouette,

        "Smallest_Cluster":
            smallest_cluster,

        "Largest_Cluster":
            largest_cluster,

        "Smallest_Cluster_%":
            smallest_percentage,

        "Largest_Cluster_%":
            largest_percentage

    })

    print(
        f"Silhouette: {silhouette:.4f}"
    )

    print(
        f"Inertia: {inertia:.2f}"
    )

    print(
        f"Smallest cluster: "
        f"{smallest_cluster:,} "
        f"({smallest_percentage:.2f}%)"
    )

    print(
        f"Largest cluster: "
        f"{largest_cluster:,} "
        f"({largest_percentage:.2f}%)"
    )


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.round({
    "Inertia": 4,
    "Silhouette_Score": 4,
    "Smallest_Cluster_%": 4,
    "Largest_Cluster_%": 4
})


# ============================================================
# Select best K
# ============================================================

best_row = results_df.loc[
    results_df["Silhouette_Score"].idxmax()
]

best_k = int(
    best_row["K"]
)

print("\n" + "=" * 70)
print("K SELECTION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nBest K based on silhouette score:")
print(best_k)

print(
    f"Best silhouette score: "
    f"{best_row['Silhouette_Score']:.4f}"
)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER EVALUATION COMPLETED")
print("=" * 70)

print(
    f"Recommended K: {best_k}"
)

print(
    f"Silhouette Score: "
    f"{best_row['Silhouette_Score']:.4f}"
)

print(
    f"Smallest Cluster: "
    f"{int(best_row['Smallest_Cluster']):,} "
    f"({best_row['Smallest_Cluster_%']:.2f}%)"
)

print(
    f"Largest Cluster: "
    f"{int(best_row['Largest_Cluster']):,} "
    f"({best_row['Largest_Cluster_%']:.2f}%)"
)

print("=" * 70)