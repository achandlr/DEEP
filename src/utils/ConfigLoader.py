import json
import os

def load_config(config_path):
    """
    Load a JSON configuration file from the given path.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_config_path(config_type, config_name):
    """
    Get the path to a specific configuration file.

    Args:
        config_type (str): The type of configuration.
        config_name (str): The name of the configuration.

    Returns:
        str: The path to the configuration file.
    """
    return os.path.join('configs', config_type, f'{config_name}.json')