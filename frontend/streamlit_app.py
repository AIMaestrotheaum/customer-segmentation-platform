import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


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


PROFILE_FILE = (
    OUTPUT_DIR / "cluster_profile.csv"
)

BUSINESS_PROFILE_FILE = (
    OUTPUT_DIR / "cluster_business_profiles.csv"
)

RECOMMENDATIONS_FILE = (
    OUTPUT_DIR / "cluster_recommendations.csv"
)

PCA_SAMPLE_FILE = (
    OUTPUT_DIR / "pca_sample.csv"
)


# ============================================================
# FASTAPI
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://backend:8000/predict"
).strip().rstrip("/")


# If Render environment variable contains only the backend
# base URL, automatically add /predict.
if not API_URL.endswith("/predict"):
    API_URL = f"{API_URL}/predict"


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

        df = pd.read_csv(
            BUSINESS_PROFILE_FILE
        )

        return df

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

        df = pd.read_csv(
            RECOMMENDATIONS_FILE
        )

        return df

    except Exception:

        return None


# ============================================================
# LOAD PCA SAMPLE
# ============================================================

@st.cache_data
def load_pca_sample():

    if not PCA_SAMPLE_FILE.exists():
        return None

    try:

        df = pd.read_csv(
            PCA_SAMPLE_FILE
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
# CUSTOMER PREDICTION
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


            # =================================================
            # SUCCESS
            # =================================================

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
                        result.get(
                            "cluster",
                            "N/A"
                        )
                    )


                with result_col2:

                    st.metric(
                        "Customer Segment",
                        result.get(
                            "segment",
                            "N/A"
                        )
                    )


                st.subheader(
                    "Business Recommendation"
                )

                st.info(
                    result.get(
                        "recommendation",
                        "No recommendation available."
                    )
                )


            # =================================================
            # VALIDATION ERROR
            # =================================================

            elif response.status_code == 422:

                st.error(
                    "Invalid customer input."
                )

                try:

                    st.json(
                        response.json()
                    )

                except Exception:

                    st.write(
                        response.text
                    )


            # =================================================
            # OTHER API ERROR
            # =================================================

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                try:

                    st.json(
                        response.json()
                    )

                except Exception:

                    st.write(
                        response.text
                    )


        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI."
            )

            st.write(
                "FastAPI endpoint:"
            )

            st.code(
                API_URL
            )


        # =====================================================
        # TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out."
            )

            st.code(
                API_URL
            )


        # =====================================================
        # GENERAL ERROR
        # =====================================================

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# ============================================================
# SEGMENTATION ANALYTICS
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
        feature
        for feature in comparison_features
        if feature in cluster_profile.columns
    ]


    if available_features:

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

    else:

        st.warning(
            "No behavioral comparison features are available."
        )


    # ========================================================
    # KEY BUSINESS METRICS
    # ========================================================

    st.subheader(
        "3️⃣ Key Business Metrics"
    )


    metric_col1, metric_col2 = st.columns(2)


    with metric_col1:

        st.write(
            "**Monetary Value by Cluster**"
        )

        if "Monetary_Value" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Monetary_Value"]
                ]
            )

        else:

            st.warning(
                "Monetary_Value is unavailable."
            )


        st.write(
            "**Frequency by Cluster**"
        )

        if "Frequency" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Frequency"]
                ]
            )

        else:

            st.warning(
                "Frequency is unavailable."
            )


    with metric_col2:

        st.write(
            "**Average Order Value by Cluster**"
        )

        if "Average_Order_Value" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Average_Order_Value"]
                ]
            )

        else:

            st.warning(
                "Average_Order_Value is unavailable."
            )


        st.write(
            "**Engagement Score by Cluster**"
        )

        if "Engagement_Score" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Engagement_Score"]
                ]
            )

        else:

            st.warning(
                "Engagement_Score is unavailable."
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

        if "Support_Interaction_Rate" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Support_Interaction_Rate"]
                ]
            )

        else:

            st.warning(
                "Support_Interaction_Rate is unavailable."
            )


    with col2:

        st.write(
            "**Return Rate**"
        )

        if "Return_Rate" in cluster_profile.columns:

            st.bar_chart(
                cluster_profile[
                    ["Return_Rate"]
                ]
            )

        else:

            st.warning(
                "Return_Rate is unavailable."
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

    else:

        st.warning(
            "cluster_business_profiles.csv could not be loaded."
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    if recommendations is not None:

        st.subheader(
            "7️⃣ Business Recommendations"
        )


        required_recommendation_columns = [
            "Cluster",
            "Segment_Name",
            "Recommendation"
        ]


        missing_recommendation_columns = [
            column
            for column in required_recommendation_columns
            if column not in recommendations.columns
        ]


        if missing_recommendation_columns:

            st.error(
                "Recommendation file is missing columns: "
                + ", ".join(
                    missing_recommendation_columns
                )
            )

        else:

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

    else:

        st.warning(
            "cluster_recommendations.csv could not be loaded."
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


    # ========================================================
    # LOAD PRECOMPUTED PCA SAMPLE
    # ========================================================

    pca_data = load_pca_sample()


    if pca_data is None:

        st.warning(
            "pca_sample.csv could not be loaded."
        )

    else:

        required_pca_columns = [
            "customer_id",
            "PCA1",
            "PCA2",
            "Cluster"
        ]


        missing_pca_columns = [
            column
            for column in required_pca_columns
            if column not in pca_data.columns
        ]


        if missing_pca_columns:

            st.error(
                "PCA sample is missing required columns: "
                + ", ".join(
                    missing_pca_columns
                )
            )


        else:

            # ================================================
            # NUMERIC CONVERSION
            # ================================================

            pca_data["PCA1"] = pd.to_numeric(
                pca_data["PCA1"],
                errors="coerce"
            )

            pca_data["PCA2"] = pd.to_numeric(
                pca_data["PCA2"],
                errors="coerce"
            )


            # ================================================
            # REMOVE INVALID VALUES
            # ================================================

            pca_data = pca_data.replace(
                [np.inf, -np.inf],
                np.nan
            )


            pca_data = pca_data.dropna(
                subset=[
                    "PCA1",
                    "PCA2",
                    "Cluster"
                ]
            )


            # ================================================
            # CLUSTER AS CATEGORY
            # ================================================

            pca_data["Cluster"] = (
                pca_data["Cluster"]
                .astype(str)
            )


            # ================================================
            # PCA SUMMARY
            # ================================================

            pca_col1, pca_col2, pca_col3 = (
                st.columns(3)
            )


            with pca_col1:

                st.metric(
                    "PCA Sample Size",
                    f"{len(pca_data):,}"
                )


            with pca_col2:

                st.metric(
                    "PCA Dimensions",
                    "2"
                )


            with pca_col3:

                st.metric(
                    "Clusters",
                    pca_data["Cluster"].nunique()
                )


            st.write(
                "Each point represents one customer "
                "in the PCA visualization sample."
            )


            # ================================================
            # SCATTER CHART
            # ================================================

            st.scatter_chart(
                pca_data,
                x="PCA1",
                y="PCA2",
                color="Cluster"
            )


            # ================================================
            # PCA DATA TABLE
            # ================================================

            with st.expander(
                "View PCA Sample Data"
            ):

                st.dataframe(
                    pca_data,
                    use_container_width=True
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