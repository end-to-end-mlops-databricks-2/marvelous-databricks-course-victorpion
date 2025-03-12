import os

import numpy as np
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from fastai.vision.all import PILImage
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


def create_or_refresh_monitoring(config, spark, workspace, model):
    inf_table = spark.sql(
        f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`fashion-image-model-serving_payload`"
    )

    def get_prediction_label(prediction):
        return model.dls.vocab[np.argmax(prediction)]

    get_prediction_label_udf = F.udf(get_prediction_label, StringType())

    request_schema = StructType([StructField("inputs", ArrayType(ArrayType(ArrayType(ArrayType(IntegerType())))))])
    response_schema = StructType(
        [StructField("predictions", ArrayType(StructType([StructField("predictions", ArrayType(DoubleType()))])))]
    )
    inf_table_parsed = (
        inf_table.withColumn("extracted_request", F.from_json(F.col("request"), request_schema)["inputs"])
        .withColumn(
            "extracted_response", F.from_json(F.col("response"), response_schema)["predictions"][0]["predictions"]
        )
        .withColumn("predicted_label", get_prediction_label_udf("extracted_response"))
    )

    base_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/images_compressed/"

    test_images_df = spark.table(f"{config.catalog_name}.{config.schema_name}.test_images")
    test_image_names = test_images_df.select("image").rdd.flatMap(lambda x: x).collect()

    image_data = []
    for image_file in os.listdir(base_path):
        if image_file in test_image_names:
            processed_data = preprocess_image(os.path.join(base_path, image_file))
            if processed_data:
                image_data.append((image_file, processed_data))

    image_schema = ["image", "extracted_request"]
    image_df = spark.createDataFrame(image_data, image_schema)
    image_df = F.broadcast(image_df)
    df_final = inf_table_parsed.join(image_df, on="extracted_request", how="inner").join(
        test_images_df.drop("update_timestamp_utc"), on="image", how="inner"
    )

    df_final = df_final.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        "predicted_label",
        "label",
        F.lit("fashion-classifier").alias("model_name"),
    )

    df_final.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def preprocess_image(image_path):
    img = PILImage.create(image_path)
    return np.array(img.resize((224, 224))).reshape(1, 224, 224, 3).tolist()


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="predicted_label",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="label",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")


# def create_alert():

#     w = WorkspaceClient()
#     srcs = w.data_sources.list()
#     alert_query = """
#     SELECT
#     (COUNT(CASE WHEN mean_absolute_error > 70000 THEN 1 END) * 100.0 / COUNT(CASE WHEN mean_absolute_error IS NOT NULL AND NOT isnan(mean_absolute_error) THEN 1 END)) AS percentage_higher_than_70000
#     FROM mlops_prod.house_prices.model_monitoring_profile_metrics"""


#     query = w.queries.create(query=sql.CreateQueryRequestQuery(display_name=f'fashion-classifier-alert-query-{time.time_ns()}',
#                                                             warehouse_id=srcs[0].warehouse_id,
#                                                             description="Alert on house price model accuracy",
#                                                             query_text=alert_query))

#     alert = w.alerts.create(
#         alert=sql.CreateAlertRequestAlert(condition=sql.AlertCondition(operand=sql.AlertConditionOperand(
#             column=sql.AlertOperandColumn(name="percentage_higher_than_70000")),
#                 op=sql.AlertOperator.GREATER_THAN,
#                 threshold=sql.AlertConditionThreshold(
#                     value=sql.AlertOperandValue(
#                         double_value=45))),
#                 display_name=f'house-price-mae-alert-{time.time_ns()}',
#                 query_id=query.id
#             )
#         )


#     # cleanup
#     w.queries.delete(id=query.id)
#     w.alerts.delete(id=alert.id)
