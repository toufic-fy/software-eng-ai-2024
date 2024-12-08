# src/ml_data_pipeline/endpoints/pipeline.py
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from prometheus_client import Counter, Summary
from pydantic import BaseModel

from ml_data_pipeline.config import ModelConfig, TransformationConfig
from ml_data_pipeline.core import load_pipeline

# Prometheus Metrics
REQUEST_COUNT = Counter(
"predict_requests_total", "Total number of requests to the predict endpoint"
)
REQUEST_LATENCY = Summary(
"predict_request_latency_seconds", "Latency of predict requests in seconds"
)
REQUEST_ERRORS = Counter(
"predict_request_errors_total", "Total number of errors in predictrequests"
)


# Input and Output schemas
class PredictInput(BaseModel):
    data: List[Dict[str, Any]]  # List of dictionaries, each representing a row


class PredictOutput(BaseModel):
    predictions: List[Any]  # List of dictionaries with predictions


# Create a router instance
router = APIRouter()

# Instantiate the Pipeline With Default Configration
TRANSFORMATION_CONFIG = TransformationConfig(scaling_method="standard", normalize=True)
MODEL_CONFIG = ModelConfig(type="linear")

pipeline_endpoint = load_pipeline(TRANSFORMATION_CONFIG, MODEL_CONFIG)


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput) -> PredictOutput:
    """
    Converts input JSON to DataFrame, runs the pipeline, and converts output DataFrame to JSON.
    """
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        try:
            # Convert input JSON to pandas DataFrame
            input_df = pd.DataFrame(input_data.data)
            logger.info("Input data converted to DataFrame.")

            # Run pipeline predict method
            predictions_df = pipeline_endpoint.run(input_df)

            return PredictOutput(predictions=predictions_df)
        except Exception as e:
            REQUEST_ERRORS.inc()
            logger.error(f"Error in predict endpoint: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed.")
