# ============================================================
# CUSTOMER SEGMENTATION PLATFORM
# STEP 8 — CLUSTER PROFILING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "customer_clusters.csv"

PROFILE_FILE = BASE_DIR / "outputs" / "cluster_profile.csv"
BUSINESS_PROFILE_FILE = (
    BASE_DIR / "outputs" / "cluster_business_profiles.csv"
)
RECOMMENDATIONS_FILE = (
    BASE_DIR / "outputs" / "cluster_recommendations.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER CLUSTER PROFILING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

if "Cluster" not in df.columns:
    raise ValueError(
        "Cluster column not found in customer_clusters.csv"
    )


# ============================================================
# 3. FEATURES
# ============================================================

clustering_features = [
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

available_features = [
    feature
    for feature in clustering_features
    if feature in df.columns
]

missing_features = [
    feature
    for feature in clustering_features
    if feature not in df.columns
]

if missing_features:
    print("\nWARNING - Missing features:")
    print(missing_features)


# ============================================================
# 4. CLUSTER SIZE
# ============================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    cluster_counts / len(df) * 100
)

cluster_size_df = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Customers": cluster_counts.values,
    "Percentage": cluster_percentages.round(2).values
})

print("\nCluster sizes:")
print(cluster_size_df.to_string(index=False))


# ============================================================
# 5. MEAN PROFILE
# ============================================================

cluster_profile = (
    df
    .groupby("Cluster")[available_features]
    .mean()
    .round(3)
)

cluster_profile.insert(
    0,
    "Customers",
    cluster_counts
)

cluster_profile.insert(
    1,
    "Percentage",
    cluster_percentages.round(2)
)

print("\nCluster mean profile:")
print(cluster_profile.to_string())


cluster_profile.to_csv(
    PROFILE_FILE
)


# ============================================================
# 6. OVERALL MEAN AND STANDARD DEVIATION
# ============================================================

overall_mean = df[available_features].mean()

overall_std = (
    df[available_features]
    .std()
    .replace(0, np.nan)
)


# ============================================================
# 7. STANDARDIZED CLUSTER DIFFERENCES
# ============================================================

standardized_profile = (
    df
    .groupby("Cluster")[available_features]
    .mean()
    .sub(overall_mean)
    .div(overall_std)
)

standardized_profile = standardized_profile.round(2)

print("\nStandardized cluster differences:")
print(
    standardized_profile.to_string()
)


# ============================================================
# 8. RELATIVE CLUSTER VALUES
# ============================================================

relative_profile = (
    df
    .groupby("Cluster")[available_features]
    .mean()
    .div(overall_mean.replace(0, np.nan))
)

relative_profile = relative_profile.round(3)


# ============================================================
# 9. IDENTIFY IMPORTANT FEATURES
# ============================================================

feature_rows = []

for cluster in standardized_profile.index:

    row = standardized_profile.loc[cluster]

    positive = (
        row[row > 0]
        .sort_values(ascending=False)
    )

    negative = (
        row[row < 0]
        .sort_values()
    )

    top_positive = positive.head(3)
    top_negative = negative.head(3)

    feature_rows.append({

        "Cluster": cluster,

        "Positive_Feature_1":
            top_positive.index[0]
            if len(top_positive) > 0
            else "",

        "Positive_Feature_1_Score":
            top_positive.iloc[0]
            if len(top_positive) > 0
            else np.nan,

        "Positive_Feature_2":
            top_positive.index[1]
            if len(top_positive) > 1
            else "",

        "Positive_Feature_2_Score":
            top_positive.iloc[1]
            if len(top_positive) > 1
            else np.nan,

        "Positive_Feature_3":
            top_positive.index[2]
            if len(top_positive) > 2
            else "",

        "Positive_Feature_3_Score":
            top_positive.iloc[2]
            if len(top_positive) > 2
            else np.nan,

        "Negative_Feature_1":
            top_negative.index[0]
            if len(top_negative) > 0
            else "",

        "Negative_Feature_1_Score":
            top_negative.iloc[0]
            if len(top_negative) > 0
            else np.nan,

        "Negative_Feature_2":
            top_negative.index[1]
            if len(top_negative) > 1
            else "",

        "Negative_Feature_2_Score":
            top_negative.iloc[1]
            if len(top_negative) > 1
            else np.nan

    })


feature_summary = pd.DataFrame(
    feature_rows
)


# ============================================================
# 10. DATA-DRIVEN SEGMENT NAMES
# ============================================================

def get_segment_name(cluster):

    row = standardized_profile.loc[cluster]

    aov = row.get(
        "Average_Order_Value",
        np.nan
    )

    monetary = row.get(
        "Monetary_Value",
        np.nan
    )

    frequency = row.get(
        "Frequency",
        np.nan
    )

    engagement = row.get(
        "Engagement_Score",
        np.nan
    )

    discount = row.get(
        "Discount_Dependency",
        np.nan
    )

    support = row.get(
        "Support_Interaction_Rate",
        np.nan
    )

    returns = row.get(
        "Return_Rate",
        np.nan
    )

    online = row.get(
        "Online_Purchase_Ratio",
        np.nan
    )

    instore = row.get(
        "InStore_Purchase_Ratio",
        np.nan
    )

    # --------------------------------------------------------
    # High order value
    # --------------------------------------------------------

    if (
        pd.notna(aov)
        and aov >= 2
        and (
            pd.isna(monetary)
            or monetary >= 1
        )
    ):
        return "High-Order-Value Customers"

    # --------------------------------------------------------
    # High engagement
    # --------------------------------------------------------

    if (
        pd.notna(engagement)
        and engagement >= 2
    ):
        return "Highly Engaged Customers"

    # --------------------------------------------------------
    # High discount dependency
    # --------------------------------------------------------

    if (
        pd.notna(discount)
        and discount >= 1.5
    ):
        return "Discount-Dependent Customers"

    # --------------------------------------------------------
    # High return behavior
    # --------------------------------------------------------

    if (
        pd.notna(returns)
        and returns >= 1.5
    ):
        return "High-Return Customers"

    # --------------------------------------------------------
    # High support interaction
    # --------------------------------------------------------

    if (
        pd.notna(support)
        and support >= 2
    ):
        return "Support-Intensive Customers"

    # --------------------------------------------------------
    # High frequency
    # --------------------------------------------------------

    if (
        pd.notna(frequency)
        and frequency >= 1.5
    ):
        return "Frequent Customers"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "Mainstream Customers"


# ============================================================
# 11. BUILD BUSINESS PROFILES
# ============================================================

business_profiles = []

for cluster in sorted(df["Cluster"].unique()):

    segment_name = get_segment_name(cluster)

    row = standardized_profile.loc[cluster]

    strongest_features = (
        row.abs()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )

    business_profiles.append({

        "Cluster": cluster,

        "Segment_Name": segment_name,

        "Customers": int(
            cluster_counts.loc[cluster]
        ),

        "Percentage": round(
            cluster_percentages.loc[cluster],
            2
        ),

        "Primary_Driver":
            strongest_features[0]
            if len(strongest_features) > 0
            else "",

        "Secondary_Driver":
            strongest_features[1]
            if len(strongest_features) > 1
            else "",

        "Third_Driver":
            strongest_features[2]
            if len(strongest_features) > 2
            else "",

        "Fourth_Driver":
            strongest_features[3]
            if len(strongest_features) > 3
            else "",

        "Fifth_Driver":
            strongest_features[4]
            if len(strongest_features) > 4
            else ""

    })


business_profile_df = pd.DataFrame(
    business_profiles
)


# ============================================================
# 12. RECOMMENDATIONS
# ============================================================

recommendations = []


for cluster in sorted(df["Cluster"].unique()):

    row = standardized_profile.loc[cluster]

    segment_name = business_profile_df[
        business_profile_df["Cluster"] == cluster
    ]["Segment_Name"].iloc[0]

    # ----------------------------------------
    # High order value
    # ----------------------------------------

    if (
        row.get("Average_Order_Value", -999)
        >= 2
    ):

        recommendation = (
            "Focus on retention, cross-selling, "
            "premium products, and personalized offers. "
            "Avoid unnecessary blanket discounts."
        )

    # ----------------------------------------
    # High engagement
    # ----------------------------------------

    elif (
        row.get("Engagement_Score", -999)
        >= 2
    ):

        recommendation = (
            "Use personalized campaigns, "
            "cross-selling, product recommendations, "
            "and loyalty initiatives."
        )

    # ----------------------------------------
    # Discount dependent
    # ----------------------------------------

    elif (
        row.get("Discount_Dependency", -999)
        >= 1.5
    ):

        recommendation = (
            "Use targeted promotions instead of "
            "blanket discounts and test bundles "
            "or loyalty incentives."
        )

    # ----------------------------------------
    # High returns
    # ----------------------------------------

    elif (
        row.get("Return_Rate", -999)
        >= 1.5
    ):

        recommendation = (
            "Investigate return drivers and improve "
            "product information, fulfillment, "
            "and post-purchase experience."
        )

    # ----------------------------------------
    # High support
    # ----------------------------------------

    elif (
        row.get("Support_Interaction_Rate", -999)
        >= 2
    ):

        recommendation = (
            "Investigate support-contact drivers. "
            "Improve product information and "
            "post-purchase support while retaining "
            "these customers."
        )

    # ----------------------------------------
    # High frequency
    # ----------------------------------------

    elif (
        row.get("Frequency", -999)
        >= 1.5
    ):

        recommendation = (
            "Use loyalty programs, replenishment "
            "campaigns, and cross-selling to "
            "increase customer lifetime value."
        )

    # ----------------------------------------
    # Default
    # ----------------------------------------

    else:

        recommendation = (
            "Treat this as the baseline customer "
            "population. Use targeted engagement, "
            "retention, and personalized recommendations."
        )

    recommendations.append({

        "Cluster": cluster,

        "Segment_Name": segment_name,

        "Recommendation": recommendation

    })


recommendations_df = pd.DataFrame(
    recommendations
)


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

business_profile_df.to_csv(
    BUSINESS_PROFILE_FILE,
    index=False
)

recommendations_df.to_csv(
    RECOMMENDATIONS_FILE,
    index=False
)


# ============================================================
# 14. DISPLAY FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLUSTER PROFILING — FINAL REPORT")
print("=" * 70)

print(
    f"Total customers: {len(df):,}"
)

print(
    f"Number of clusters: {df['Cluster'].nunique()}"
)

print("\nBusiness profiles:")

print(
    business_profile_df.to_string(
        index=False
    )
)

print("\nRecommendations:")

print(
    recommendations_df.to_string(
        index=False
    )
)


print("\n")
print("Output files:")

print(
    f"- {PROFILE_FILE}"
)

print(
    f"- {BUSINESS_PROFILE_FILE}"
)

print(
    f"- {RECOMMENDATIONS_FILE}"
)

print("=" * 70)

print(
    "\nCluster profiling completed successfully."
)