import pandas as pd
from .base_model import Model
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel(Model):
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:        
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
