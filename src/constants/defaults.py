import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACT_DIR: str = os.path.join(os.getcwd(), "artifacts", TIMESTAMP)

# Pipeline Common Constants
DATA_BUCKET_NAME: str = "signature-recognition-data-bucket"
MODEL_BUCKET_NAME: str = "signature-recognition-model-bucket"
DATA_ZIP_FILE_NAME: str = "cedar_dataset.zip"
MODEL_FILE_NAME: str = "model.pt"

# Data Ingestion Constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
TRANSFORM_OBJECT_FILE_NAME: str = "transform.pkl"

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9
EPOCHS: int = 20
BATCH_SIZE: int = 32

# Model Evaluation Constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"

# Model Pusher Constants
MODEL_PUSHER_DIR_NAME: str = "model_pusher"