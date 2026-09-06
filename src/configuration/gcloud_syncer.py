import os
import sys
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


class GCloudSync:
    """
    Handles synchronization between local directory paths and Google Cloud Storage buckets using gsutil CLI.
    """

    def sync_folder_to_gcloud(self, gcp_bucket_url: str, filepath: str, filename: str) -> None:
        """
        Syncs/uploads a local file or directory to a Google Cloud Storage bucket target.
        """
        try:
            command = f"gsutil cp {filepath}/{filename} {gcp_bucket_url}/{filename}"
            logger.info(f"Executing GCS Upload Command: {command}")
            os.system(command)
        except Exception as e:
            raise CustomException(e, sys)

    def sync_folder_from_gcloud(self, gcp_bucket_url: str, filename: str, destination_filepath: str) -> None:
        """
        Syncs/downloads a file from a Google Cloud Storage bucket to a target local path.
        """
        try:
            os.makedirs(destination_filepath, exist_ok=True)
            command = f"gsutil cp {gcp_bucket_url}/{filename} {destination_filepath}/{filename}"
            logger.info(f"Executing GCS Download Command: {command}")
            os.system(command)
        except Exception as e:
            raise CustomException(e, sys)