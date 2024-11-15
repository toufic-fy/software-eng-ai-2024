# src/ml_data_pipeline/data_transform/minmax_scaler_transformer.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .base_transformer import DataTransformer

class MinMaxScalerTransformer(DataTransformer):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
