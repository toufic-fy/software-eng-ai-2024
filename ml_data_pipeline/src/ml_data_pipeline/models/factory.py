# src/ml_data_pipeline/model/factory.py
from .linear_model import LinearModel
from .tree_model import DecisionTreeModel
from .base_model import Model

class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> Model:
        if model_type == "linear":
            return LinearModel()
        elif model_type == "tree":
            return DecisionTreeModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
