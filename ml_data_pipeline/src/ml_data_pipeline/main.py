from fastapi import FastAPI
from pydantic import BaseModel
from ml_data_pipeline.models import ModelFactory

app = FastAPI(title="ML Data Pipeline API", version="1.0")

# Define a request schema
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

# Define a prediction route
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Load your model (if not already loaded globally)
    model = ModelFactory.get_model("tree")
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict(features)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)