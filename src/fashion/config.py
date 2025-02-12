import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    catalog_name: str
    schema_name: str
    experiment_name_custom: str

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
