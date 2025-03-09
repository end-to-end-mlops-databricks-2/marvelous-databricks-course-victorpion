import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from fashion.config import ProjectConfig
from fashion.monitoring import create_or_refresh_monitoring

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

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
