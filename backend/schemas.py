from pydantic import BaseModel, Field


class CustomerInput(BaseModel):

    Recency: float = Field(..., ge=0)

    Frequency: float = Field(..., ge=0)

    Monetary_Value: float = Field(..., ge=0)

    Average_Order_Value: float = Field(..., ge=0)

    Engagement_Score: float = Field(..., ge=0)

    Discount_Dependency: float = Field(..., ge=0)

    Return_Rate: float = Field(..., ge=0)

    Online_Purchase_Ratio: float = Field(..., ge=0)

    InStore_Purchase_Ratio: float = Field(..., ge=0)

    Avg_Items_Per_Transaction: float = Field(..., ge=0)

    Support_Interaction_Rate: float = Field(..., ge=0)


class PredictionResponse(BaseModel):

    cluster: int

    segment: str

    recommendation: str