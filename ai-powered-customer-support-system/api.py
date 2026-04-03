from fastapi import FastAPI
from src.inferencial import predict
from src.priority import detect_priority

app = FastAPI(
    title="Customer Text Intelligence API",
    description="DistilBERT-based multi-label NLP system (4GB RAM optimized)",
    version="1.2"
)

@app.post("/predict")
def classify(text: str):
    """
    Input: customer message text
    Output: detected issues, priority, confidence
    """

    issues, confidence = predict(text)
    priority = detect_priority(text)

    return {
        "issues": issues,
        "priority": priority,
        "confidence": round(confidence, 2)
    }
