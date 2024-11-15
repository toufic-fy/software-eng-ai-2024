# src/ml_data_pipeline/main.py
import argparse
from ml_data_pipeline.config import load_config
from ml_data_pipeline.data_loader import DataLoaderFactory
from ml_data_pipeline.data_transformer import TransformerFactory


parser = argparse.ArgumentParser(description="Run the ML data pipeline with specified configuration.")
parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )

def main():
    args = parser.parse_args()
    config = load_config(args.config)
    print("Loaded Configuration:")
    print(config)

    # Use DataLoaderFactory to load data
    data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
    data = data_loader.load_data(config.data_loader.file_path)
    print("Loaded Data:")
    print(data)

    # Use TransformerFactory to transform data
    transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)
    transformed_data = transformer.transform(data)
    print("Transformed Data:")
    print(transformed_data)

if __name__ == "__main__":
    main()
