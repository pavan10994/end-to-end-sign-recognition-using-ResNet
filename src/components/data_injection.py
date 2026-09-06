import os
import sys
import zipfile
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
            self.gcloud_syncer = GCloudSync()
        except Exception as e:
            raise CustomException(e, sys)

    def download_data_from_gcs(self) -> str:
        """
        Downloads the raw dataset zip file from GCS.
        """
        try:
            logger.info("Starting dataset download from GCS bucket...")
            os.makedirs(self.config.feature_store_file_path, exist_ok=True)
            self.gcloud_syncer.sync_folder_from_gcloud(
                gcp_bucket_url=f"gs://{self.config.data_bucket_name}",
                filename=self.config.zip_file_name,
                destination_filepath=self.config.feature_store_file_path,
            )
            downloaded_file_path = os.path.join(
                self.config.feature_store_file_path, self.config.zip_file_name
            )
            logger.info(f"Dataset successfully downloaded to: {downloaded_file_path}")
            return downloaded_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def extract_zip_file(self, zip_file_path: str) -> str:
        """
        Unzips the dataset into the target ingestion folder and cleans up the zip archive.
        """
        try:
            logger.info("Extracting dataset zip file...")
            os.makedirs(self.config.ingested_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.config.ingested_dir)
            
            logger.info(f"Dataset extracted into: {self.config.ingested_dir}")
            
            # Clean up raw zip after extraction
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
                logger.info(f"Removed zip archive: {zip_file_path}")

            return self.config.ingested_dir
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Triggers data ingestion pipeline component.
        """
        try:
            logger.info("=== Starting Data Ingestion Pipeline Component ===")
            zip_file_path = self.download_data_from_gcs()
            extracted_path = self.extract_zip_file(zip_file_path=zip_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_path=self.config.feature_store_file_path,
                extracted_data_path=extracted_path,
            )
            logger.info(f"Data Ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)