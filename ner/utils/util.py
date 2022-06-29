import os
import yaml
from typing import Dict
from from_root import from_root


def read_config(file_name: str) -> Dict:
    """
    This Function reads the config.yaml from root directory and
    return configuration in dictionary.

    Returns: Dict of config
    """
    config_path = os.path.join(from_root(), file_name)
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content
