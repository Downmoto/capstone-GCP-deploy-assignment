from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    sex: str
    chestpain_type: str = Field(alias="chest pain type")
    restingbp: float = Field(alias="resting bp s")
    cholesterol: float
    fastingbs: int = Field(alias="fasting bs")
    restingecg: str = Field(alias="resting ecg")
    maxhr: float = Field(alias="max hr")
    exerciseangina: str = Field(alias="exercise angina")
    oldpeak: float
    st_slope: str = Field(alias="st slope")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[HeartFeatures]


class PredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
