import argparse

import mlflow
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig, Tags
from fashion.model import FashionClassifier

# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()

config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model with the config path
custom_model = FashionClassifier(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[f"{args.root_path}/artifacts/.internal/fashion_classifier-0.0.1-py3-none-any.whl"],
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
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_images").toPandas()

test_image = test_set["image"][0]

predictions_df = custom_model.load_latest_model_and_predict(test_image)
