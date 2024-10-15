from typing import Callable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(filepath: str, fill_na_value: float, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess data."""
    data = pd.read_csv(filepath)
    data = data.fillna(fill_na_value)
    X = data.drop(target_name, axis=1)
    y = data[target_name]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into a train and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_func: Callable[[], BaseEstimator], **kwargs) -> BaseEstimator:
    """Train the model with inputs and target data."""
    model = model_func(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate the model with a single metric."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def get_model(model_name: str) -> Callable[[], BaseEstimator]:
    """High-order function to select the model to train."""
    if model_name == "logistic_regression":
        return LogisticRegression
    elif model_name == "random_forest":
        return RandomForestClassifier
    else:
        raise ValueError(f"Model {model_name} is not supported.")

def run_workflow(model_name: str, model_kwargs: dict, filepath: str, fill_na_value: float, target_name: str, test_size: float, random_state: int) -> None:
    """Orchestrate the training workflow."""
    X, y = load_and_preprocess_data(filepath, fill_na_value, target_name)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    model_func = get_model(model_name)
    model = train_model(X_train, y_train, model_func, **model_kwargs)
    evaluate_model(model, X_test, y_test)

# Example usage
run_workflow(
    filepath='dataset.csv',
    fill_na_value=0.0,
    target_name='target',
    test_size=0.2,
    random_state=42,
    model_name='random_forest',  # Or 'logistic_regression'
    model_kwargs={'n_estimators': 30},
)

