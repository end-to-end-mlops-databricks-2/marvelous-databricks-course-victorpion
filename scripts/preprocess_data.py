import argparse
import logging

import yaml
from pyspark.sql import SparkSession

from fashion.config import ProjectConfig
from fashion.data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the fashion prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/images_data.csv",
    header=True,
    inferSchema=True,
).toPandas()


# Initialize DataProcessor
data_processor = DataProcessor(
    pandas_df=df,
    config=config,
    spark=spark,
    images_path=f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/images_compressed/",
)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
