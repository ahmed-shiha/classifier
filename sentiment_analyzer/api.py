# from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    # probabilities: Dict[str, float]
    sentiment_1: str
    sentiment_2: str
    # confidence: float


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    # sentiment, confidence, probabilities = model.predict(request.text)
    sentiment_1, sentiment_2 = model.predict(request.text)

    return SentimentResponse(
        sentiment_1=sentiment_1, sentiment_2=sentiment_2
        # , confidence=confidence, probabilities=probabilities
    )
