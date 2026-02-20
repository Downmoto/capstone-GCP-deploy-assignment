from contextlib import asynccontextmanager
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from typing import Any, AsyncIterator

from src.model_loader import load_artifacts
from src.schemas import PredictionRequest, PredictionResponse
from src.security import require_api_key

model: Any | None = None
preprocessor: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global model, preprocessor
    model, preprocessor = load_artifacts()
    yield


app = FastAPI(title="heart failure prediction api", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifacts are not loaded",
        )

    rows = [item.model_dump(by_alias=True) for item in payload.data]
    frame = pd.DataFrame(rows)

    transformed = preprocessor.transform(frame)
    predictions = model.predict(transformed)

    probabilities: list[float]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[:, 1].tolist()
    else:
        probabilities = [float("nan")] * len(predictions)

    return PredictionResponse(
        predictions=[int(v) for v in predictions.tolist()],
        probabilities=[float(v) for v in probabilities],
    )
