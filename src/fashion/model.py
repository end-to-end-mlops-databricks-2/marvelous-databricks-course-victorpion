from functools import partial
from typing import List

import mlflow
import numpy as np
import pandas as pd
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    PILImage,
    RandomResizedCrop,
    accuracy,
    resnet18,
    vision_learner,
)
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig, Tags


def get_x(r, catalog_name, schema_name, volume_name):
    return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/images_compressed/{r['image']}"


def get_y(r):
    return r["label"]


class FashionClassifier:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.volume_name = self.config.volume_name
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_images")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_images").toPandas()
        self.data_version = (
            self.spark.sql(f"DESCRIBE HISTORY {self.catalog_name}.{self.schema_name}.train_images")
            .select("version")
            .limit(1)
            .collect()[0][0]
        )
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self):
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            LightGBM regression model
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        # Create DataBlock
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=partial(get_x, self.catalog_name, self.schema_name, self.volume_name),
            get_y=get_y,
            item_tfms=RandomResizedCrop(128, min_scale=0.35),
        )  # ensure every item is of the same size
        self.dls = dblock.dataloaders(self.train_set)  # collates items from dataset into minibatches
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self):
        """
        Train the model.
        """
        logger.info("🚀 Starting training...")
        self.learn = vision_learner(self.dls, resnet18, metrics=accuracy)
        self.learn.fine_tune(5, base_lr=3e-3)

    def log_model(self):
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"] + [f"code/{package.split('/')[-1]}" for package in self.code_paths]
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            _, accuracy = self.learn.validate()
            logger.info(f"📊 Accuracy: {accuracy}")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("data_version", self.data_version)
            image_path = f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}/images_compressed/598090c2-f12f-4e60-9b23-d556a38117ad.jpg"
            image = PILImage.create(image_path)  # noqa: F405
            image = image.resize((224, 224))
            image_array = np.array(image).reshape(1, 224, 224, 3)
            # Log the model
            signature = infer_signature(model_input=image_array, model_output="category")
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_images",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.fastai.log_model(
                fastai_learner=self.learn,
                artifact_path="pyfunc-fashion-image-model",
                code_paths=self.code_paths,
                conda_env=_mlflow_conda_env(additional_pip_deps=additional_pip_deps),
                signature=signature,
            )

    def register_model(self):
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-fashion-image-model",
            name=f"{self.catalog_name}.{self.schema_name}.fashion_image_model_custom",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")
        MlflowClient().set_registered_model_alias(
            f"{self.catalog_name}.{self.schema_name}.fashion_image_model_custom",
            "latest-model",
            registered_model.version,
        )

    def retrieve_current_run_dataset(self):
        return mlflow.data.get_source(mlflow.get_run(self.run_id).inputs.dataset_inputs[0].dataset).load()

    def retrieve_current_run_metadata(self):
        run_data = mlflow.get_run(self.run_id).data.to_dictionary()
        return run_data["metrics"], run_data["params"]

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        model = mlflow.pyfunc.load_model(
            f"models:/{self.catalog_name}.{self.schema_name}.fashion_image_model_custom@latest-model"
        )
        return model.predict(input_data)
