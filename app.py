import io
import sys
from PIL import Image
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger

app = FastAPI(
    title="Signature Recognition MLOps API",
    description="Production-grade API for offline signature verification (Genuine vs. Forged)",
    version="1.0.0",
)

# Enable CORS for external frontend or mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "Signature Recognition Pipeline API is active"}


def run_training_in_background():
    """
    Background worker function to trigger the full training pipeline execution.
    """
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Background training pipeline failed: {str(e)}")


@app.get("/train", tags=["Pipeline"])
async def train_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers the end-to-end training pipeline asynchronously in the background.
    """
    try:
        background_tasks.add_task(run_training_in_background)
        return JSONResponse(
            status_code=202,
            content={
                "status": "success",
                "message": "Training pipeline initiated in background task.",
            },
        )
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/predict", tags=["Inference"])
async def predict_signature(file: UploadFile = File(...)):
    """
    Receives an uploaded signature image file, runs evaluation transform, 
    and classifies signature as Genuine or Forged.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload an image file (PNG/JPEG)."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predictor = PredictionPipeline()
        result = predictor.predict(image=image)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            },
        )
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)