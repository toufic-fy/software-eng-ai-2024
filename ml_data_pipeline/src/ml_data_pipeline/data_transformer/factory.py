# src/ml_data_pipeline/data_transform/factory.py
from .standard_scaler_transformer import StandardScalerTransformer
from .minmax_scaler_transformer import MinMaxScalerTransformer
from .base_transformer import DataTransformer

class TransformerFactory:
    @staticmethod
    def get_transformer(scaling_method: str) -> DataTransformer:
        if scaling_method == "standard":
            return StandardScalerTransformer()
        elif scaling_method == "minmax":
            return MinMaxScalerTransformer()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
