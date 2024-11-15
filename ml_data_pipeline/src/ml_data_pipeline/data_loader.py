# src/ml_data_pipeline/data_loader.py
import pandas as pd

def load_data():
    # Example: Create a dummy DataFrame
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0]
    })
    return data
