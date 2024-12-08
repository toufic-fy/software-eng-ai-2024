from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from ml_data_pipeline.endpoints.health import router as health_router
from ml_data_pipeline.endpoints.pipeline import router as pipeline_router
from ml_data_pipeline.endpoints.train import router as train_router

app = FastAPI(title="ML Data Pipeline API", version="1.0")

Instrumentator().instrument(app).expose(app)
# Define a request schema

# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])
app.include_router(train_router, prefix="/api", tags=["Train"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

