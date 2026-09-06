import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from src.configuration.gcloud_syncer import GCloudSync
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


class PredictionPipeline:
    def __init__(
        self,
        gcs_bucket_name: str = "signature-recognition-model-bucket",
        model_file_name: str = "model.pt",
        local_model_dir: str = "artifacts/prediction",
    ):
        try:
            self.gcs_bucket_name = gcs_bucket_name
            self.model_file_name = model_file_name
            self.local_model_dir = local_model_dir
            self.local_model_path = os.path.join(self.local_model_dir, self.model_file_name)
            self.gcloud_syncer = GCloudSync()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.class_labels = {0: "Forged", 1: "Genuine"}

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self) -> nn.Module:
        try:
            if not os.path.exists(self.local_model_path):
                logger.info("Local prediction model missing. Syncing from GCS bucket...")
                self.gcloud_syncer.sync_folder_from_gcloud(
                    gcp_bucket_url=f"gs://{self.gcs_bucket_name}",
                    filename=self.model_file_name,
                    destination_filepath=self.local_model_dir,
                )

            model = models.resnet34(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(num_ftrs, 2))
            model.load_state_dict(torch.load(self.local_model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image: Image.Image) -> dict:
        try:
            model = self.load_model()
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            prediction_label = self.class_labels.get(predicted_class_idx, "Unknown")
            logger.info(f"Prediction result: {prediction_label} with confidence {confidence:.4f}")

            return {
                "prediction": prediction_label,
                "confidence": round(confidence, 4),
            }
        except Exception as e:
            raise CustomException(e, sys)