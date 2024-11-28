"""
This module contains utility function(s) to load a config file.
- load_config: Loads a config file.
"""

import yaml

def load_config(path="configs/test.yaml"):
    """
    Loads a config file.

    Args:
        path (str, optional)
            Path to config file. Defaults to "configs/test.yaml".

    Returns:
        config: dict
            The loaded config file.
    """

    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
        print(f"[INFO] Config {path} loaded.")

    return config
