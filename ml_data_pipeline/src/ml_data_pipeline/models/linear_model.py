# src/ml_data_pipeline/model/linear_model.py
import pandas as pd
from .base_model import Model

class LinearModel(Model):
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        print("Training Linear Model on data")
        # Dummy training logic
        self.coefficients = [0.5] * X.shape[1]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        print("Predicting with Linear Model")
        return pd.DataFrame([sum(row * coef for row, coef in zip(X.iloc[i], self.coefficients))] for i in range(X.shape[0]))
