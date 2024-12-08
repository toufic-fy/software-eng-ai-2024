from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from ml_data_pipeline.config import load_config
from ml_data_pipeline.models import ModelFactory
from ml_data_pipeline.train import train

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
async def predict(request: PredictionRequest) -> dict[str, str]:
    # Load your model (if not already loaded globally)
    model = ModelFactory.get_model("tree")
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict(features)
    return {"prediction": prediction}

@app.get("/train")
async def train_model(config_path: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        background_tasks.train_model_background(config_path)
        return {"status": "success"}
    except Exception:
        return {"status": "error while training!"}

def train_model_background(config_path: str) -> None:
        config = load_config(config_path)
        train(config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # config = load_config("config/config_dev.yaml")
    # print(config.mlflow)
