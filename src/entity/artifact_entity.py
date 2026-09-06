from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    feature_store_path: str
    extracted_data_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_val_path: str
    transformed_test_path: str
    transform_object_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_loss: float
    val_loss: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    gcs_model_loss: float
    trained_model_loss: float


@dataclass
class ModelPusherArtifact:
    pushed_model_path: str
    gcs_bucket_url: str