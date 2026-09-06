import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder

from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger
from src.utils.main_utils import load_object


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.config = model_eval_config
            self.transformation_artifact = data_transformation_artifact
            self.trainer_artifact = model_trainer_artifact
            self.gcloud_syncer = GCloudSync()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model_loss(self, model_path: str, dataloader: DataLoader) -> float:
        """
        Loads state weights from model_path and evaluates loss over test set.
        """
        try:
            model = models.resnet34(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(num_ftrs, 2))
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0

            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)

            return running_loss / len(dataloader.dataset)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Executes model comparison logic against existing GCS production model.
        """
        try:
            logger.info("=== Starting Model Evaluation Pipeline Component ===")
            
            transform = load_object(self.transformation_artifact.transform_object_path)
            full_dataset = ImageFolder(
                root=self.transformation_artifact.transformed_train_path, 
                transform=transform
            )
            total_count = len(full_dataset)
            train_size = int(0.60 * total_count)
            val_size = int(0.30 * total_count)
            test_size = total_count - train_size - val_size

            _, _, test_ds = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            # Evaluate newly trained model
            trained_model_loss = self.evaluate_model_loss(
                self.trainer_artifact.trained_model_path, test_loader
            )

            gcs_model_path = os.path.join(self.config.model_evaluation_dir, self.config.model_file_name)
            
            # Download remote model if present
            self.gcloud_syncer.sync_folder_from_gcloud(
                gcp_bucket_url=f"gs://{self.config.gcs_model_bucket}",
                filename=self.config.model_file_name,
                destination_filepath=self.config.model_evaluation_dir
            )

            is_accepted = True
            gcs_model_loss = float("inf")

            if os.path.exists(gcs_model_path):
                gcs_model_loss = self.evaluate_model_loss(gcs_model_path, test_loader)
                logger.info(f"Production Loss: {gcs_model_loss:.4f} | Newly Trained Loss: {trained_model_loss:.4f}")
                
                if trained_model_loss >= gcs_model_loss:
                    is_accepted = False
                    logger.info("Newly trained model did NOT outperform production model. Rejecting pusher phase.")
                else:
                    logger.info("Newly trained model outperformed production model. Accepting pusher phase.")
            else:
                logger.info("No remote model found in GCS. Accepting newly trained model as baseline.")

            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                gcs_model_loss=gcs_model_loss,
                trained_model_loss=trained_model_loss,
            )
            return artifact
        except Exception as e:
            raise CustomException(e, sys)