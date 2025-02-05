import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from fashion.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession, images_path):
        self.df = pandas_df
        self.config = config
        self.spark = spark
        self.images_path = images_path

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""

        self.df.loc[self.df["label"] == "Not sure", "label"] = "Not_sure"  # format Not sure category
        self.df["image"] = self.df["image"] + ".jpg"  # add .jpg to all image ids
        self.df["label_cat"] = self.df["label"] + "_" + self.df["kids"].astype(str)  # merge kids boolean with category
        self.df = self.df[["image", "label_cat"]]  # keep only image id and category
        self.df["label_cat"] = self.df["label_cat"].astype("category")  # change data type to category
        images_ids = set(os.listdir(self.images_path))  # list the images ids present in the images folder
        self.df = self.df[self.df["image"].isin(images_ids)]  # remove rows with missing images from the csv

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_images"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_images"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_images "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_images "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
