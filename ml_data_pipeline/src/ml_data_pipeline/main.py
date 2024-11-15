# src/ml_data_pipeline/main.py
from ml_data_pipeline.data_loader import load_data

def main():
    data = load_data()
    print("Loaded Data:")
    print(data)

if __name__ == "__main__":
    main()
