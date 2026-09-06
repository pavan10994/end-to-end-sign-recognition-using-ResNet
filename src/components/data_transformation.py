import os
import sys
import pickle
from typing import Tuple
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger
from src.utils.main_utils import save_object


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Defines image transformation pipelines for training and validation/testing.
        """
        try:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            return train_transform, eval_transform
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Applies image dataset splits (60% train, 30% val, 10% test) and persists transform artifacts.
        """
        try:
            logger.info("=== Starting Data Transformation Pipeline Component ===")
            train_transform, eval_transform = self.get_data_transforms()

            dataset_dir = self.ingestion_artifact.extracted_data_path
            logger.info(f"Loading ImageFolder dataset from: {dataset_dir}")

            full_dataset = ImageFolder(root=dataset_dir, transform=train_transform)
            total_count = len(full_dataset)

            train_size = int(0.60 * total_count)
            val_size = int(0.30 * total_count)
            test_size = total_count - train_size - val_size

            logger.info(f"Dataset split counts - Total: {total_count}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

            # Save dataset transform object
            save_object(
                file_path=self.config.transform_object_file_path,
                obj=train_transform
            )

            # Create output directories for splits
            os.makedirs(self.config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.config.transformed_val_dir, exist_ok=True)
            os.makedirs(self.config.transformed_test_dir, exist_ok=True)

            artifact = DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_dir,
                transformed_val_path=self.config.transformed_val_dir,
                transformed_test_path=self.config.transformed_test_dir,
                transform_object_path=self.config.transform_object_file_path,
            )

            logger.info(f"Data Transformation completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)