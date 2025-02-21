from typing import List

import mlflow
import mlflow.fastai
import pandas as pd
from fastai.vision.all import *  # noqa: F403
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig, Tags

# class FashionImageModelWrapper(mlflow.pyfunc.PythonModel):
#     def __init__(self, model):
#         self.model = model

#     def predict(self, context, model_input: str):

#         if isinstance(model_input, pd.DataFrame):
#             model_input = model_input.to_dict(orient="records")[0]

#         if isinstance(model_input, pd.Series):
#             model_input = model_input.to_dict()

#         if not isinstance(model_input, list | dict):
#             msg = f"Unexpected input format: {type(model_input)}. Expected a dictionary or pandas DataFrame. Model input: {model_input}"
#             raise TypeError(msg)

#         print(f"Type of model input : {type(model_input)}")
#         print(f"Model input : {model_input}")
#         predictions = self.model.predict(model_input["image"])
#         # looks like {"Prediction": "Category"}
#         return {"Prediction": predictions[0]}


class CustomModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
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
        self.data_version = "0"  # describe history -> retrieve

        # self.X_train = self.train_set[self.num_features + self.cat_features]
        # self.y_train = self.train_set[self.target]
        # self.X_test = self.test_set[self.num_features + self.cat_features]
        # self.y_test = self.test_set[self.target]
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

        get_x = lambda r: f"/Volumes/{self.catalog_name}/{self.schema_name}/fashion/images_compressed/" + r["image"]  # noqa: E731
        get_y = lambda r: r["label"]  # noqa: E731
        # Create DataBlock
        dblock = DataBlock(  # noqa: F405
            blocks=(ImageBlock, CategoryBlock),  # noqa: F405
            get_x=get_x,
            get_y=get_y,
            item_tfms=RandomResizedCrop(128, min_scale=0.35),  # noqa: F405
        )  # ensure every item is of the same size
        self.dls = dblock.dataloaders(self.train_set)  # collates items from dataset into minibatches
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self):
        """
        Train the model.
        """
        logger.info("🚀 Starting training...")
        self.learn = vision_learner(self.dls, resnet18, metrics=accuracy)  # noqa: F405
        self.learn.fine_tune(1, base_lr=3e-3)

    def predict(self, context, model_input: str):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]

        if isinstance(model_input, pd.Series):
            model_input = model_input.to_dict()

        if not isinstance(model_input, list | dict):
            msg = f"Unexpected input format: {type(model_input)}. Expected a dictionary or pandas DataFrame. Model input: {model_input}"
            raise TypeError(msg)

        print(f"Type of model input : {type(model_input)}")
        print(f"Model input : {model_input}")
        predictions = self.learn.predict(model_input["image"])
        # looks like {"Prediction": "Category"}
        return {"Prediction": predictions[0]}

    def log_model(self):
        """
        Log the model.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            # y_pred = self.pipeline.predict(self.X_test)

            # Retrieve validation accuracy
            _, accuracy = self.learn.validate()

            logger.info(f"📊 Validation Accuracy: {accuracy}")

            # Log parameters and metrics
            mlflow.log_metric("accuracy", accuracy)

            # example_image_input_path = f"/Volumes/{self.catalog_name}/{self.schema_name}/fashion/images_compressed/" + self.test_set["image"][0]
            # example_image_input = Image.open(example_image_input_path)
            # example_array_input = np.array(example_image_input)

            # Log the model
            signature = infer_signature(model_input={"image": "string"}, model_output="category")
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_images",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            self.spark = None
            self.train_set_spark = None

            mlflow.fastai.log_model(
                fastai_learner=self,
                artifact_path="pyfunc-fashion-image-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
            )

    def register_model(self):
        """
        Register model in UC
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-fashion-image-model",
            name=f"{self.catalog_name}.{self.schema_name}.fashion_image_model_custom",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.fashion_image_model_custom",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self):
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.fashion_image_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions: None is context
        predictions = model.predict(input_data)

        # This also works
        # model.unwrap_python_model().predict(None, input_data)
        # check out this article:
        # https://medium.com/towards-data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535

        # Return predictions as a DataFrame
        return predictions
