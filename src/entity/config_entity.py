import os
from dataclasses import dataclass
from src.constants import defaults


@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = defaults.ARTIFACT_DIR


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str
    feature_store_file_path: str
    ingested_dir: str
    data_bucket_name: str
    zip_file_name: str


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str
    transformed_train_dir: str
    transformed_val_dir: str
    transformed_test_dir: str
    transform_object_file_path: str


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str
    trained_model_file_path: str
    learning_rate: float
    momentum: float
    epochs: int
    batch_size: int


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str
    gcs_model_bucket: str
    model_file_name: str


@dataclass
class ModelPusherConfig:
    model_pusher_dir: str
    gcs_model_bucket: str
    model_file_name: str