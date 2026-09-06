import sys
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger
from src.utils.main_utils import read_yaml_file

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.components.data_injection import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config_dict = read_yaml_file(config_path)
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_ingestion(self):
        try:
            cfg = self.config_dict["data_ingestion_config"]
            ingestion_config = DataIngestionConfig(
                data_ingestion_dir=cfg["data_ingestion_dir"],
                feature_store_file_path=cfg["feature_store_file_path"],
                ingested_dir=cfg["ingested_dir"],
                data_bucket_name=cfg["data_bucket_name"],
                zip_file_name=cfg["zip_file_name"],
            )
            data_ingestion = DataIngestion(data_ingestion_config=ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact):
        try:
            cfg = self.config_dict["data_transformation_config"]
            transformation_config = DataTransformationConfig(
                data_transformation_dir=cfg["data_transformation_dir"],
                transformed_train_dir=cfg["transformed_train_dir"],
                transformed_val_dir=cfg["transformed_val_dir"],
                transformed_test_dir=cfg["transformed_test_dir"],
                transform_object_file_path=cfg["transform_object_file_path"],
            )
            data_transformation = DataTransformation(
                data_transformation_config=transformation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            cfg = self.config_dict["model_trainer_config"]
            trainer_config = ModelTrainerConfig(
                model_trainer_dir=cfg["model_trainer_dir"],
                trained_model_file_path=cfg["trained_model_file_path"],
                learning_rate=cfg["learning_rate"],
                momentum=cfg["momentum"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
            )
            model_trainer = ModelTrainer(
                model_trainer_config=trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, data_transformation_artifact, model_trainer_artifact):
        try:
            cfg = self.config_dict["model_evaluation_config"]
            eval_config = ModelEvaluationConfig(
                model_evaluation_dir=cfg["model_evaluation_dir"],
                gcs_model_bucket=cfg["gcs_model_bucket"],
                model_file_name=cfg["model_file_name"],
            )
            model_eval = ModelEvaluation(
                model_eval_config=eval_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_trainer_artifact):
        try:
            cfg = self.config_dict["model_pusher_config"]
            pusher_config = ModelPusherConfig(
                model_pusher_dir=cfg["model_pusher_dir"],
                gcs_model_bucket=cfg["gcs_model_bucket"],
                model_file_name=cfg["model_file_name"],
            )
            model_pusher = ModelPusher(
                model_pusher_config=pusher_config,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logger.info("Starting Full Training Pipeline...")
            ingestion_artifact = self.start_data_ingestion()
            transformation_artifact = self.start_data_transformation(ingestion_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            eval_artifact = self.start_model_evaluation(transformation_artifact, trainer_artifact)

            if eval_artifact.is_model_accepted:
                logger.info("Model accepted by evaluation step. Pushing model to GCS...")
                pusher_artifact = self.start_model_pusher(trainer_artifact)
                logger.info(f"Pipeline finished successfully with model pushed: {pusher_artifact}")
            else:
                logger.info("Model rejected by evaluation step. Skipping pusher stage.")

        except Exception as e:
            raise CustomException(e, sys)