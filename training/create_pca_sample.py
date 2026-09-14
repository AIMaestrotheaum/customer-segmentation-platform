import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_FILE = "outputs/customer_clusters.csv"
OUTPUT_FILE = "outputs/pca_sample.csv"

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
    "Support_Interaction_Rate",
]

print("Loading customer data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total customers: {len(df):,}")

# Take 10,000 customers for visualization
sample_size = min(10000, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
).copy()

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    sample[FEATURES]
)

# PCA → 2 dimensions
pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

# Create small visualization dataset
pca_df = pd.DataFrame({
    "customer_id": sample["customer_id"].values,
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Cluster": sample["Cluster"].values,
})

pca_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"PCA sample created: {OUTPUT_FILE}")
print(f"Rows: {len(pca_df):,}")
print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum():.2%}"
)