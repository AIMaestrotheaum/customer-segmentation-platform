import pandas as pd


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


def prepare_features(customer_data: dict):

    df = pd.DataFrame(
        [customer_data]
    )

    df = df[FEATURES]

    return df