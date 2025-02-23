import pandas as pd
from loguru import logger
from PIL import Image
from pyspark.dbutils import DBUtils
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
        dbutils = DBUtils(self.spark)
        self.df = self.df.copy()
        self.df = self.df[~self.df["label"].isin(["Not sure", "Skip", "Other"])]  # remove non-clothing categories
        self.df["image"] = self.df["image"] + ".jpg"  # add .jpg to all image ids
        self.df = self.df[["image", "label"]]  # keep only image id and label
        self.df["label"] = self.df["label"].astype("category")  # change data type to category
        images = dbutils.fs.ls(self.images_path)
        images_ids = set([img.name for img in images])  # list the images ids present in the images folder
        self.df = self.df[self.df["image"].isin(images_ids)]  # remove rows with missing images from the csv
        self.df = self.remove_corrupt_images(self.df)
        # value_counts = self.df["label"].value_counts()
        # categories_to_keep = value_counts[value_counts >= 10].index
        # self.df = self.df[self.df["label"].isin(categories_to_keep)]
        logger.info(f"{len(self.df)} valid images in the dataset.")

    def is_corrupt(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return False
        except Exception:
            return True

    def remove_corrupt_images(self, df):
        logger.info("Removing corrupted images...")
        valid_rows = []
        for _, row in df.iterrows():
            image_id = row["image"]
            image_path = f"{self.images_path}{image_id}"
            if not self.is_corrupt(image_path):
                valid_rows.append(row)

        df_cleaned = pd.DataFrame(valid_rows)

        return df_cleaned

    def split_data(self, test_size=0.01, random_state=42):
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

        train_set_path = f"{self.config.catalog_name}.{self.config.schema_name}.train_images"

        train_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(train_set_path)

        logger.info(f"Saved {len(train_set)} images for training at : {train_set_path}")

        test_set_path = f"{self.config.catalog_name}.{self.config.schema_name}.test_images"

        test_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(test_set_path)

        logger.info(f"Saved {len(test_set)} images for testing at : {test_set_path}")

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_images "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_images "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
