import mlflow
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig, Tags
from fashion.model import FashionClassifier

# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# Initialize model with the config path
custom_model = FashionClassifier(
    config=config, tags=tags, spark=spark, code_paths=["../dist/fashion-0.0.1-py3-none-any.whl"]
)
custom_model.load_data()
custom_model.prepare_features()
# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()

run_id = mlflow.search_runs(experiment_names=["/Shared/fashion-image-custom"]).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-fashion-image-model")

# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()

# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# Register model
custom_model.register_model()

# Predict on the test set

# test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# X_test = test_set.drop(config.target).toPandas()

# predictions_df = custom_model.load_latest_model_and_predict(X_test)
