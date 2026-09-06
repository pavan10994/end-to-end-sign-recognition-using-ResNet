import os
import sys
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.config = model_pusher_config
            self.trainer_artifact = model_trainer_artifact
            self.gcloud_syncer = GCloudSync()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Pushes verified trained model to target GCP Cloud Storage Bucket.
        """
        try:
            logger.info("=== Starting Model Pusher Pipeline Component ===")
            
            filepath = os.path.dirname(self.trainer_artifact.trained_model_path)
            filename = os.path.basename(self.trainer_artifact.trained_model_path)
            gcp_bucket_url = f"gs://{self.config.gcs_model_bucket}"

            self.gcloud_syncer.sync_folder_to_gcloud(
                gcp_bucket_url=gcp_bucket_url,
                filepath=filepath,
                filename=filename,
            )

            artifact = ModelPusherArtifact(
                pushed_model_path=self.trainer_artifact.trained_model_path,
                gcs_bucket_url=gcp_bucket_url,
            )
            logger.info(f"Model successfully pushed to GCS: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)