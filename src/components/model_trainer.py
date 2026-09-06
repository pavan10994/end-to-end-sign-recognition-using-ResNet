import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger
from src.utils.main_utils import load_object


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.config = model_trainer_config
            self.transformation_artifact = data_transformation_artifact
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self) -> nn.Module:
        """
        Loads ResNet-34 pre-trained model and modifies FC layer for 2-class binary signature output.
        """
        try:
            logger.info("Initializing ResNet-34 transfer learning model architecture...")
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(num_ftrs, 2)
            )
            
            return model.to(self.device)
        except Exception as e:
            raise CustomException(e, sys)

    def train_epoch(self, model, dataloader, criterion, optimizer) -> float:
        """
        Executes one epoch of training over the train dataset loader.
        """
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        return running_loss / len(dataloader.dataset)

    def validate_epoch(self, model, dataloader, criterion) -> float:
        """
        Executes evaluation pass over the validation dataset loader.
        """
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                
        return running_loss / len(dataloader.dataset)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Runs model training loops over specified epochs, monitors losses, and saves model.pt.
        """
        try:
            logger.info("=== Starting Model Trainer Pipeline Component ===")
            transform = load_object(self.transformation_artifact.transform_object_path)

            # Re-instantiate image dataset loaders
            full_dataset = ImageFolder(
                root=self.transformation_artifact.transformed_train_path, 
                transform=transform
            )
            total_count = len(full_dataset)
            train_size = int(0.60 * total_count)
            val_size = int(0.30 * total_count)
            test_size = total_count - train_size - val_size

            train_ds, val_ds, _ = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )

            train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

            model = self.get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(), 
                lr=self.config.learning_rate, 
                momentum=self.config.momentum
            )

            logger.info(f"Training across {self.config.epochs} epochs on device: {self.device}")
            best_val_loss = float("inf")
            final_train_loss = 0.0

            for epoch in range(1, self.config.epochs + 1):
                train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
                val_loss = self.validate_epoch(model, val_loader, criterion)
                
                logger.info(f"Epoch [{epoch}/{self.config.epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    final_train_loss = train_loss

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            torch.save(model.state_dict(), self.config.trained_model_file_path)
            logger.info(f"Model checkpoint saved successfully at: {self.config.trained_model_file_path}")

            artifact = ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_file_path,
                train_loss=final_train_loss,
                val_loss=best_val_loss,
            )
            return artifact
        except Exception as e:
            raise CustomException(e, sys)