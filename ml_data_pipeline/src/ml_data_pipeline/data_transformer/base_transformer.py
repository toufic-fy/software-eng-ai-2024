# src/ml_data_pipeline/data_transform/base_transformer.py
from abc import ABC, abstractmethod
import pandas as pd

class DataTransformer(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
