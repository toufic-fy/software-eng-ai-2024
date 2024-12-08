from fastapi import APIRouter, BackgroundTasks

from ml_data_pipeline.config import load_config
from ml_data_pipeline.train import train

router = APIRouter()


@router.get("/train")
async def train_model(config_path: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        background_tasks.add_task(train_model_background, config_path)
        return {"status": "success"}
    except Exception:
        return {"status": "error while training!"}

def train_model_background(config_path: str) -> None:
        config = load_config(config_path)
        train(config)
