import argparse
import json
import os
import time

import mlflow
import numpy as np
import requests
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from fastai.vision.all import PILImage
from loguru import logger
from pyspark.dbutils import DBUtils

from fashion.config import ProjectConfig
from fashion.monitoring import create_or_refresh_monitoring

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

args = parser.parse_args()
config_path = f"{args.root_path}/files/project_config.yml"

# Load configuration
config = ProjectConfig.from_yaml(config_path=config_path)

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()
dbutils = DBUtils(spark)

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_images").toPandas()

model = mlflow.fastai.load_model("models:/gso_dev_gsomlops.vpion.fashion_image_model_custom@latest-model")


def call_endpoint(record, model):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/fashion-image-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"inputs": record},
    )

    data = json.loads(response.text)
    predictions = data["predictions"][0]["predictions"]
    predicted_index = np.argmax(predictions)
    predicted_label = model.dls.vocab[predicted_index]

    return predicted_label


def evaluate_model(i):
    image_path = (
        f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/images_compressed/"
        + test_set["image"][i]
    )
    image = PILImage.create(image_path)
    image_array = np.array(image.resize((224, 224))).reshape(1, 224, 224, 3).tolist()
    call_endpoint(image_array, model)


logger.info("Populating the inference table with test data...")
for i in range(len(test_set)):
    evaluate_model(i)
    time.sleep(0.2)


create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace, model=model)
