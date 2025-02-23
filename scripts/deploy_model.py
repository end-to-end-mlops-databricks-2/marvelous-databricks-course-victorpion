import json
import os

import mlflow
import numpy as np
import requests
from fastai.vision.all import PILImage
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig
from fashion.model_serving import ModelServing

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.fashion_image_model_custom", endpoint_name="fashion-image-model-serving"
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# Sample 100 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_images").toPandas()

# Load model to retrieve vocabulary
model = mlflow.fastai.load_model("models:/gso_dev_gsomlops.vpion.fashion_image_model_custom@latest-model")


def call_endpoint(record, vocab):
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
    predicted_label = vocab[predicted_index]

    return predicted_label


def evaluate_model(i):
    image_path = "/Volumes/gso_dev_gsomlops/vpion/fashion/images_compressed/" + test_set["image"][i]
    image = PILImage.create(image_path)
    image_array = np.array(image.resize((224, 224))).reshape(1, 224, 224, 3).tolist()
    print(image.show())
    print(f"True label : {test_set['label'][i]}")
    print(f"Predicted label : {call_endpoint(image_array, model.dls.vocab)}")


evaluate_model(0)
