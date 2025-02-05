# Databricks notebook source
pip install tensorflow

# COMMAND ----------

import sys
sys.path.append("/Workspace/Users/vpion@eu.delhaize.com/.bundle/marvelous-databricks-course-victorpion/dev/files/src/fashion_mnist")

# COMMAND ----------

import tensorflow as tf
from data_processor import DataProcessor

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
processor = DataProcessor(normalize=True, add_channel_dim=True)
x_train_processed = processor.process(x_train)
x_test_processed = processor.process(x_test)
print(x_train_processed.shape, x_test_processed.shape)

# COMMAND ----------

