# src/ml_data_pipeline/config.py
from pydantic import BaseModel, field_validator
from omegaconf import OmegaConf

class DataLoaderConfig(BaseModel):
    file_path: str
    file_type: str

    @field_validator("file_type")
    def validate_file_type(cls, value):
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    normalize: bool
    scaling_method: str

    @field_validator("scaling_method")
    def validate_scaling_method(cls, value):
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class ModelConfig(BaseModel):
    type: str

    @field_validator("type")
    def validate_scaling_method(cls, value):
        if value not in {"tree", "linear"}:
            raise ValueError("model type must be 'linear' or 'tree'")
        return value


class Config(BaseModel):
    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig


def load_config(config_path: str) -> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config(**config_dict)
