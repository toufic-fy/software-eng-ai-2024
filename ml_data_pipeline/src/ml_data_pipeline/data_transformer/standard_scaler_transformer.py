# src/ml_data_pipeline/data_transform/standard_scaler_transformer.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .base_transformer import DataTransformer

class StandardScalerTransformer(DataTransformer):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
