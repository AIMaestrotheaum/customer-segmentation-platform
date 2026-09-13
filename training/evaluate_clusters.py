# ============================================================
# CUSTOMER SEGMENTATION — CLUSTER EVALUATION
# ============================================================

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "clustering_features.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "kmeans_model.joblib"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "clustering_evaluation.csv"
)


# ============================================================
# CLUSTERING FEATURES
# ============================================================

CLUSTERING_FEATURES = [
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER SEGMENTATION — MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Clustering feature file not found:\n{DATA_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"K-Means model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading clustering data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        col for col in CLUSTERING_FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"\nMissing clustering features:\n{missing_features}"
        )

    # --------------------------------------------------------
    # Load model and scaler
    # --------------------------------------------------------

    print("\nLoading K-Means model...")
    model = joblib.load(MODEL_PATH)

    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    print(f"\nModel: {model}")
    print(f"Scaler: {scaler}")

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = df[CLUSTERING_FEATURES].copy()

    # Make sure everything is numeric
    for column in CLUSTERING_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(0)

    print(
        f"\nNumber of clustering features: {X.shape[1]}"
    )

    print("\nFeatures used:")

    for i, feature in enumerate(CLUSTERING_FEATURES, 1):
        print(f"{i:2}. {feature}")

    # --------------------------------------------------------
    # Scale using SAVED scaler
    # --------------------------------------------------------

    print("\nScaling features using saved scaler...")

    X_scaled = scaler.transform(X)

    # --------------------------------------------------------
    # Get cluster labels from saved model
    # --------------------------------------------------------

    print("Generating cluster predictions...")

    cluster_labels = model.predict(X_scaled)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    unique_clusters = np.unique(cluster_labels)

    print(
        f"\nClusters found: {unique_clusters.tolist()}"
    )

    if len(unique_clusters) < 2:
        raise ValueError(
            "Evaluation requires at least 2 clusters."
        )

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    cluster_counts = pd.Series(
        cluster_labels
    ).value_counts().sort_index()

    total_customers = len(cluster_labels)

    smallest_cluster = cluster_counts.min()
    largest_cluster = cluster_counts.max()

    smallest_percentage = (
        smallest_cluster / total_customers * 100
    )

    largest_percentage = (
        largest_cluster / total_customers * 100
    )

    print("\nCluster sizes:")

    cluster_size_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Customers": cluster_counts.values,
        "Percentage": (
            cluster_counts.values /
            total_customers * 100
        ).round(2)
    })

    print(
        cluster_size_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # Silhouette Score
    # --------------------------------------------------------

    print("\nCalculating Silhouette Score...")

    # Use sample for large dataset
    SAMPLE_SIZE = min(
        10000,
        len(X_scaled)
    )

    rng = np.random.RandomState(42)

    if len(X_scaled) > SAMPLE_SIZE:

        sample_indices = rng.choice(
            len(X_scaled),
            size=SAMPLE_SIZE,
            replace=False
        )

        X_sample = X_scaled[sample_indices]
        labels_sample = cluster_labels[sample_indices]

        silhouette = silhouette_score(
            X_sample,
            labels_sample
        )

        silhouette_sample_size = SAMPLE_SIZE

    else:

        silhouette = silhouette_score(
            X_scaled,
            cluster_labels
        )

        silhouette_sample_size = len(X_scaled)

    # --------------------------------------------------------
    # Davies-Bouldin Score
    # --------------------------------------------------------

    print("Calculating Davies-Bouldin Score...")

    db_score = davies_bouldin_score(
        X_scaled,
        cluster_labels
    )

    # --------------------------------------------------------
    # Calinski-Harabasz Score
    # --------------------------------------------------------

    print("Calculating Calinski-Harabasz Score...")

    ch_score = calinski_harabasz_score(
        X_scaled,
        cluster_labels
    )

    # --------------------------------------------------------
    # Inertia
    # --------------------------------------------------------

    inertia = model.inertia_

    # --------------------------------------------------------
    # Evaluation interpretation
    # --------------------------------------------------------

    if silhouette >= 0.50:
        silhouette_interpretation = "Strong separation"

    elif silhouette >= 0.25:
        silhouette_interpretation = "Moderate separation"

    else:
        silhouette_interpretation = "Weak separation"

    # --------------------------------------------------------
    # Create evaluation result
    # --------------------------------------------------------

    evaluation = pd.DataFrame({
        "Customers": [total_customers],
        "Features": [len(CLUSTERING_FEATURES)],
        "K": [model.n_clusters],
        "Silhouette_Score": [silhouette],
        "Silhouette_Sample_Size": [
            silhouette_sample_size
        ],
        "Davies_Bouldin_Score": [db_score],
        "Calinski_Harabasz_Score": [ch_score],
        "Inertia": [inertia],
        "Smallest_Cluster": [smallest_cluster],
        "Largest_Cluster": [largest_cluster],
        "Smallest_Cluster_%": [
            smallest_percentage
        ],
        "Largest_Cluster_%": [
            largest_percentage
        ],
        "Silhouette_Interpretation": [
            silhouette_interpretation
        ]
    })

    # --------------------------------------------------------
    # Save evaluation
    # --------------------------------------------------------

    evaluation.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CUSTOMER SEGMENTATION — EVALUATION REPORT")
    print("=" * 70)

    print(
        f"Customers:              {total_customers:,}"
    )

    print(
        f"Features:               {len(CLUSTERING_FEATURES)}"
    )

    print(
        f"Selected K:             {model.n_clusters}"
    )

    print(
        f"Silhouette Score:       {silhouette:.4f}"
    )

    print(
        f"Silhouette Sample Size: {silhouette_sample_size:,}"
    )

    print(
        f"Davies-Bouldin Score:   {db_score:.4f}"
    )

    print(
        f"Calinski-Harabasz:      {ch_score:.2f}"
    )

    print(
        f"Inertia:                {inertia:.2f}"
    )

    print(
        f"Smallest Cluster:       "
        f"{smallest_percentage:.2f}%"
    )

    print(
        f"Largest Cluster:        "
        f"{largest_percentage:.2f}%"
    )

    print(
        f"Interpretation:         "
        f"{silhouette_interpretation}"
    )

    print("\nEvaluation saved to:")

    print(OUTPUT_PATH)

    print("=" * 70)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY.")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()