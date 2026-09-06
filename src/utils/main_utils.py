import os
import sys
import yaml
import pickle
import numpy as np
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        logger.info(f"Reading YAML file from path: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary/object to a YAML file.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logger.info(f"YAML file written successfully to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to disk using pickle serialization.
    """
    try:
        logger.info(f"Saving object to path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object successfully saved at path: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads a pickle-serialized object from disk.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
        logger.info(f"Loading object from path: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to disk in binary format (.npy).
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Saved numpy array to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a binary NumPy array (.npy) from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)