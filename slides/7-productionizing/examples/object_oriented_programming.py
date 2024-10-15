from abc import ABC, abstractmethod
from typing import Tuple, Type

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Model(ABC):
    """Abstract base class for models."""
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

class RandomForestModel(Model):
    """Random Forest Classifier model."""
    def __init__(self, n_estimators: int = 20, max_depth: int = 5) -> None:
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

class KerasBinaryClassifier(Model):
    """Simple binary classification model using Keras."""
    def __init__(self, input_dim: int, epochs: int = 100, batch_size: int = 32) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return (predictions > 0.5).flatten()

class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def get_model(model_name: str, **kwargs) -> Model:
        # Assume all model classes are defined in the global scope.
        model_class = globals()[model_name]
        return model_class(**kwargs)

class Workflow:
    """Main workflow class for model training and evaluation."""
    def run_workflow(self, model_name: str, model_kwargs: dict, filepath: str, fill_na_value: float, target_name: str, test_size: float, random_state: int) -> None:
        X, y = self.load_and_preprocess_data(filepath, fill_na_value, target_name)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        model = ModelFactory.get_model(model_name, **model_kwargs)
        model.train(X_train, y_train)
        accuracy = self.evaluate_model(model, X_test, y_test)
        print(f"Model Accuracy: {accuracy}")

    def load_and_preprocess_data(self, filepath: str, fill_na_value: float, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data."""
        data = pd.read_csv(filepath)
        data = data.fillna(fill_na_value)
        X = data.drop(target_name, axis=1)
        y = data[target_name]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into a train and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def evaluate_model(self, model: Model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Evaluate the model with a single metric."""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

# Example usage
workflow = Workflow()
workflow.run_workflow(
    filepath='dataset.csv',
    fill_na_value=0.0,
    target_name='target',
    test_size=0.2,
    random_state="42",
    model_name='RandomForestModel',  # Or 'KerasBinaryClassifier'
    model_kwargs={'n_estimators': 30},
)

