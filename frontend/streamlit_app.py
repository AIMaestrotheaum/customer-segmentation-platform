import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import requests

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Platform",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

PROFILE_FILE = OUTPUT_DIR / "cluster_profile.csv"

BUSINESS_PROFILE_FILE = (
    OUTPUT_DIR / "cluster_business_profiles.csv"
)

RECOMMENDATIONS_FILE = (
    OUTPUT_DIR / "cluster_recommendations.csv"
)

CUSTOMER_CLUSTERS_FILE = (
    OUTPUT_DIR / "customer_clusters.csv"
)


# ============================================================
# FASTAPI
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://backend:8000/predict"
)
# ============================================================
# MODEL FEATURES
# ============================================================

CLUSTER_FEATURES = [
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
# LOAD CLUSTER PROFILE
# ============================================================

@st.cache_data
def load_cluster_profile():

    if not PROFILE_FILE.exists():
        return None

    try:

        df = pd.read_csv(
            PROFILE_FILE,
            index_col=0
        )

        return df

    except Exception:

        return None


# ============================================================
# LOAD BUSINESS PROFILES
# ============================================================

@st.cache_data
def load_business_profiles():

    if not BUSINESS_PROFILE_FILE.exists():
        return None

    try:

        return pd.read_csv(
            BUSINESS_PROFILE_FILE
        )

    except Exception:

        return None


# ============================================================
# LOAD RECOMMENDATIONS
# ============================================================

@st.cache_data
def load_recommendations():

    if not RECOMMENDATIONS_FILE.exists():
        return None

    try:

        return pd.read_csv(
            RECOMMENDATIONS_FILE
        )

    except Exception:

        return None


# ============================================================
# LOAD CUSTOMER CLUSTERS
# ============================================================

@st.cache_data
def load_customer_clusters():

    if not CUSTOMER_CLUSTERS_FILE.exists():
        return None

    try:

        df = pd.read_csv(
            CUSTOMER_CLUSTERS_FILE
        )

        return df

    except Exception:

        return None


# ============================================================
# LOAD DATA
# ============================================================

cluster_profile = load_cluster_profile()

business_profiles = load_business_profiles()

recommendations = load_recommendations()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "📊 Customer Segmentation Platform"
)

st.caption(
    "K-Means Customer Segmentation & Behavioral Analytics"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Customer Prediction",
        "📊 Segmentation Analytics"
    ]
)


# ============================================================
# ============================================================
# CUSTOMER PREDICTION
# ============================================================
# ============================================================

if page == "🔮 Customer Prediction":

    st.header(
        "Customer Information"
    )

    st.write(
        "Enter customer behavioral information "
        "to predict the customer's segment."
    )


    # ========================================================
    # INPUT COLUMNS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=180.0,
            step=1.0
        )

        frequency = st.number_input(
            "Frequency",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        monetary_value = st.number_input(
            "Monetary Value",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        average_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=500.0,
            step=50.0
        )


    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        engagement_score = st.number_input(
            "Engagement Score",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        discount_dependency = st.number_input(
            "Discount Dependency",
            min_value=0.0,
            value=0.25,
            step=0.05
        )

        return_rate = st.number_input(
            "Return Rate",
            min_value=0.0,
            value=0.06,
            step=0.01
        )

        support_interaction_rate = st.number_input(
            "Support Interaction Rate",
            min_value=0.0,
            value=1.0,
            step=0.1
        )


    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:

        online_purchase_ratio = st.number_input(
            "Online Purchase Ratio",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        instore_purchase_ratio = st.number_input(
            "In-Store Purchase Ratio",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        avg_items_per_transaction = st.number_input(
            "Average Items per Transaction",
            min_value=0.0,
            value=5.5,
            step=0.1
        )


    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Customer Segment",
        type="primary",
        use_container_width=True
    ):

        customer_data = {

            "Recency": recency,

            "Frequency": frequency,

            "Monetary_Value": monetary_value,

            "Average_Order_Value":
                average_order_value,

            "Engagement_Score":
                engagement_score,

            "Discount_Dependency":
                discount_dependency,

            "Return_Rate":
                return_rate,

            "Online_Purchase_Ratio":
                online_purchase_ratio,

            "InStore_Purchase_Ratio":
                instore_purchase_ratio,

            "Avg_Items_Per_Transaction":
                avg_items_per_transaction,

            "Support_Interaction_Rate":
                support_interaction_rate
        }


        try:

            response = requests.post(
                API_URL,
                json=customer_data,
                timeout=10
            )


            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Customer segmentation completed."
                )

                st.divider()

                result_col1, result_col2 = (
                    st.columns(2)
                )


                with result_col1:

                    st.metric(
                        "Cluster",
                        result["cluster"]
                    )


                with result_col2:

                    st.metric(
                        "Customer Segment",
                        result["segment"]
                    )


                st.subheader(
                    "Business Recommendation"
                )

                st.info(
                    result["recommendation"]
                )


            elif response.status_code == 422:

                st.error(
                    "Invalid customer input."
                )

                st.json(
                    response.json()
                )


            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.json(
                    response.json()
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI."
            )

            st.code(
                "uvicorn backend.app:app --reload"
            )


        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out."
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# ============================================================
# ============================================================
# SEGMENTATION ANALYTICS
# ============================================================
# ============================================================

else:

    st.header(
        "📊 Segmentation Analytics"
    )

    st.write(
        "Explore customer behavior across the K-Means clusters."
    )


    # ========================================================
    # DATA CHECK
    # ========================================================

    if cluster_profile is None:

        st.error(
            "cluster_profile.csv could not be loaded."
        )

        st.stop()


    # ========================================================
    # OVERVIEW METRICS
    # ========================================================

    total_customers = int(
        cluster_profile["Customers"].sum()
    )

    total_clusters = len(
        cluster_profile
    )

    largest_cluster = int(
        cluster_profile["Customers"].max()
    )

    smallest_cluster = int(
        cluster_profile["Customers"].min()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "Number of Clusters",
            total_clusters
        )


    with col3:

        st.metric(
            "Largest Cluster",
            f"{largest_cluster:,}"
        )


    with col4:

        st.metric(
            "Smallest Cluster",
            f"{smallest_cluster:,}"
        )


    st.divider()


    # ========================================================
    # CLUSTER DISTRIBUTION
    # ========================================================

    st.subheader(
        "1️⃣ Customer Distribution by Cluster"
    )


    distribution = cluster_profile[
        ["Customers", "Percentage"]
    ].copy()


    distribution.index = [
        f"Cluster {x}"
        for x in distribution.index
    ]


    st.bar_chart(
        distribution["Customers"]
    )


    st.dataframe(
        distribution,
        use_container_width=True
    )


    # ========================================================
    # BEHAVIORAL COMPARISON
    # ========================================================

    st.subheader(
        "2️⃣ Behavioral Comparison"
    )


    comparison_features = [
        "Recency",
        "Frequency",
        "Monetary_Value",
        "Average_Order_Value",
        "Engagement_Score",
        "Discount_Dependency",
        "Return_Rate",
        "Support_Interaction_Rate"
    ]


    available_features = [
        x for x in comparison_features
        if x in cluster_profile.columns
    ]


    selected_feature = st.selectbox(
        "Select feature",
        available_features
    )


    feature_chart = cluster_profile[
        [selected_feature]
    ].copy()


    feature_chart.index = [
        f"Cluster {x}"
        for x in feature_chart.index
    ]


    st.bar_chart(
        feature_chart
    )


    # ========================================================
    # KEY BUSINESS METRICS
    # ========================================================

    st.subheader(
        "3️⃣ Key Business Metrics"
    )


    metric_features = [
        "Monetary_Value",
        "Average_Order_Value",
        "Frequency",
        "Engagement_Score"
    ]


    metric_features = [
        x for x in metric_features
        if x in cluster_profile.columns
    ]


    metric_col1, metric_col2 = st.columns(2)


    with metric_col1:

        st.write(
            "**Monetary Value by Cluster**"
        )

        st.bar_chart(
            cluster_profile[
                ["Monetary_Value"]
            ]
        )


        st.write(
            "**Frequency by Cluster**"
        )

        st.bar_chart(
            cluster_profile[
                ["Frequency"]
            ]
        )


    with metric_col2:

        st.write(
            "**Average Order Value by Cluster**"
        )

        st.bar_chart(
            cluster_profile[
                ["Average_Order_Value"]
            ]
        )


        st.write(
            "**Engagement Score by Cluster**"
        )

        st.bar_chart(
            cluster_profile[
                ["Engagement_Score"]
            ]
        )


    # ========================================================
    # SUPPORT + RETURN BEHAVIOR
    # ========================================================

    st.subheader(
        "4️⃣ Customer Service & Retention Indicators"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Support Interaction Rate**"
        )

        st.bar_chart(
            cluster_profile[
                ["Support_Interaction_Rate"]
            ]
        )


    with col2:

        st.write(
            "**Return Rate**"
        )

        st.bar_chart(
            cluster_profile[
                ["Return_Rate"]
            ]
        )


    # ========================================================
    # FULL PROFILE
    # ========================================================

    st.subheader(
        "5️⃣ Complete Cluster Profile"
    )


    st.dataframe(
        cluster_profile,
        use_container_width=True
    )


    # ========================================================
    # BUSINESS PROFILES
    # ========================================================

    if business_profiles is not None:

        st.subheader(
            "6️⃣ Business Segment Profiles"
        )

        st.dataframe(
            business_profiles,
            use_container_width=True
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    if recommendations is not None:

        st.subheader(
            "7️⃣ Business Recommendations"
        )


        for _, row in recommendations.iterrows():

            cluster = row["Cluster"]

            segment = row["Segment_Name"]

            recommendation = row[
                "Recommendation"
            ]


            with st.expander(
                f"Cluster {cluster} — {segment}"
            ):

                st.write(
                    recommendation
                )


    # ========================================================
    # PCA VISUALIZATION
    # ========================================================

    st.subheader(
        "8️⃣ PCA Cluster Visualization"
    )

    st.write(
        "A 2-dimensional representation of the "
        "11 clustering features. A 10,000-customer "
        "sample is used for visualization."
    )


    # Load customer-level cluster data only when needed
    customer_data = load_customer_clusters()


    if customer_data is None:

        st.warning(
            "customer_clusters.csv could not be loaded."
        )


    else:

        missing_features = [
            feature
            for feature in CLUSTER_FEATURES
            if feature not in customer_data.columns
        ]


        if missing_features:

            st.error(
                "Missing clustering features:"
            )

            st.write(
                missing_features
            )


        elif "Cluster" not in customer_data.columns:

            st.error(
                "Cluster column is missing."
            )


        else:

            # ================================================
            # SAMPLE DATA
            # ================================================

            SAMPLE_SIZE = 10000

            if len(customer_data) > SAMPLE_SIZE:

                pca_data = customer_data.sample(
                    n=SAMPLE_SIZE,
                    random_state=42
                )

            else:

                pca_data = customer_data.copy()


            # ================================================
            # NUMERIC CONVERSION
            # ================================================

            X_pca = pca_data[
                CLUSTER_FEATURES
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )


            # ================================================
            # HANDLE MISSING VALUES
            # ================================================

            X_pca = X_pca.replace(
                [np.inf, -np.inf],
                np.nan
            )


            X_pca = X_pca.fillna(
                X_pca.median()
            )


            # ================================================
            # STANDARDIZE
            # ================================================

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(
                X_pca
            )


            # ================================================
            # PCA
            # ================================================

            pca = PCA(
                n_components=2,
                random_state=42
            )


            X_pca_2d = pca.fit_transform(
                X_scaled
            )


            # ================================================
            # PCA DATAFRAME
            # ================================================

            pca_df = pd.DataFrame({

                "PCA_1": X_pca_2d[:, 0],

                "PCA_2": X_pca_2d[:, 1],

                "Cluster":
                    pca_data[
                        "Cluster"
                    ].astype(str).values

            })


            # ================================================
            # SCATTER CHART
            # ================================================

            st.scatter_chart(
                pca_df,
                x="PCA_1",
                y="PCA_2",
                color="Cluster"
            )


            # ================================================
            # EXPLAINED VARIANCE
            # ================================================

            variance_1 = (
                pca.explained_variance_ratio_[0]
                * 100
            )

            variance_2 = (
                pca.explained_variance_ratio_[1]
                * 100
            )


            st.write(
                f"**PCA 1 explained variance:** "
                f"{variance_1:.2f}%"
            )

            st.write(
                f"**PCA 2 explained variance:** "
                f"{variance_2:.2f}%"
            )

            st.write(
                f"**Combined explained variance:** "
                f"{variance_1 + variance_2:.2f}%"
            )


    # ========================================================
    # DOWNLOAD PROFILE
    # ========================================================

    st.subheader(
        "9️⃣ Download Cluster Analysis"
    )


    profile_csv = cluster_profile.to_csv()


    st.download_button(
        label="⬇️ Download Cluster Profile",
        data=profile_csv,
        file_name="cluster_profile.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Segmentation Platform | "
    "K-Means Clustering | "
    "Behavioral Analytics"
)