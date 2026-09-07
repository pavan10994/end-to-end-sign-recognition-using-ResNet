# ✍️ End-to-End Offline Signature Verification Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--34-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com/)

An end-to-end MLOps pipeline designed for **Offline Signature Verification**, classifying handwritten signatures as **Genuine** or **Forged**. The project follows a modular machine learning architecture covering data ingestion, transformation, model training, evaluation, model promotion, API serving, containerization, and cloud deployment.

The pipeline uses **transfer learning with ResNet-34**, serves predictions through a **FastAPI REST API**, executes training asynchronously using **FastAPI BackgroundTasks**, and is containerized using **Docker** for deployment.

---

## 🛠️ Complete Tech Stack & Tools

| Layer                              | Tool / Technology       | Purpose                                                   |
| :---------------------------------- | :----------------------- | :---------------------------------------------------------|
| **Language & Environment**          | Python 3.10              | Core application and machine learning development         |
| **Deep Learning Framework**         | PyTorch, Torchvision     | Deep learning model development and image processing      |
| **Transfer Learning Model**         | ResNet-34                | Feature extraction and signature classification           |
| **Data Processing**                 | NumPy, Pillow            | Image processing and numerical operations                 |
| **Machine Learning Utilities**      | Scikit-learn             | Dataset splitting and supporting ML utilities              |
| **Configuration & Serialization**   | PyYAML, Dill             | Configuration management and Python object serialization  |
| **Web Framework**                   | FastAPI                  | REST API development and model serving                     |
| **API Server**                      | Uvicorn                  | ASGI server for running the FastAPI application             |
| **Data Validation**                 | Pydantic                 | Request and response validation                            |
| **Asynchronous Processing**         | FastAPI BackgroundTasks  | Non-blocking training pipeline execution                   |
| **Containerization**                | Docker                   | Reproducible application environment                       |
| **Version Control**                 | Git, GitHub              | Source code management and collaboration                   |
| **Cloud Deployment**                | Render                   | Container-based cloud application deployment                |

---

## 📐 Pipeline Architecture

```text
       ┌────────────────────────┐
       │   Dataset (CEDAR Zip)  │
       └───────────┬────────────┘
                    │
                    ▼
  ┌──────────────────────────────────┐
  │ 1. Data Ingestion Component       │
  └────────────────┬───────────────────┘
                    │
                    │ Extract and organize
                    │ raw signature images
                    ▼
  ┌──────────────────────────────────┐
  │ 2. Data Transformation Component  │
  └────────────────┬───────────────────┘
                    │
                    │ Resize to 224 × 224
                    │ Standardize & normalize
                    │ Split dataset
                    ▼
  ┌──────────────────────────────────┐
  │ 3. Model Trainer Component        │
  └────────────────┬───────────────────┘
                    │
                    │ Fine-tune ResNet-34
                    │ SGD + Cross-Entropy Loss
                    │ Save model checkpoint
                    ▼
  ┌──────────────────────────────────┐
  │ 4. Model Evaluation Component     │
  └────────────────┬───────────────────┘
                    │
                    ▼
           ┌─────────────────────┐
           │ Is New Model Better? │
           └──────────┬───────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
           YES                    NO
            │                     │
            ▼                     ▼
┌───────────────────────┐   ┌───────────────────────┐
│ 5. Model Pusher        │   │ Discard New Model /   │
│ Update Production      │   │ Keep Active Model     │
│ Model Artifact         │   └───────────────────────┘
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────────────┐
│ FastAPI Prediction Service      │
└──────────────┬───────────────────┘
               │
               ▼
┌────────────────────────────────┐
│ Docker Container                │ ──► Deploy to Render
└────────────────────────────────┘
```

---

## 🔄 MLOps Workflow

The complete workflow follows the sequence:

**CEDAR Signature Dataset**

→ **Data Ingestion**

→ **Data Transformation**

→ **ResNet-34 Model Training**

→ **Model Evaluation**

→ **Model Comparison**

→ **Model Promotion**

→ **FastAPI Prediction Service**

→ **Docker Containerization**

→ **Render Deployment**

The evaluation stage acts as a quality gate. A newly trained model is compared against the currently active model before it is promoted as the production artifact.

---

## 📊 Data Processing Pipeline

The dataset passes through the following stages:

### 1. Data Ingestion

The data ingestion component is responsible for extracting and organizing the raw signature dataset from the source artifact.

### 2. Data Transformation

The transformation pipeline prepares signature images for ResNet-34 training by performing:

* Image resizing to **224 × 224**
* Image standardization
* Image normalization
* Dataset splitting

The dataset is divided into:

* **60% Training Data**
* **30% Validation Data**
* **10% Test Data**

### 3. Model Training

The model training component uses transfer learning with **ResNet-34**.

The training configuration includes:

* Pretrained ResNet-34 architecture
* Fine-tuning for signature classification
* Stochastic Gradient Descent (SGD)
* Cross-Entropy Loss
* Model checkpoint generation

The trained model is stored as:

```text
model.pt
```

---

## 🧠 Model Architecture

The project uses **ResNet-34**, a convolutional neural network architecture available through Torchvision.

Transfer learning allows the model to use previously learned visual representations and adapt them to the offline signature classification task.

The final classification task distinguishes between two classes:

1. **Genuine**
2. **Forged**

The trained model receives a processed signature image and produces raw prediction scores for the available classes.

These scores are converted into probabilities using the Softmax function.

---

## 📂 Project Directory Structure

```text
signature_recognition/
├── .github/
│   └── workflows/
│       └── deploy.yaml              # CI/CD deployment workflow configuration
├── config/
│   └── config.yaml                  # Centralized pipeline configuration
├── src/
│   ├── components/                  # Core ML pipeline components
│   │   ├── data_ingestion.py        # Dataset ingestion logic
│   │   ├── data_transformation.py   # Image preprocessing and dataset splitting
│   │   ├── model_trainer.py         # ResNet-34 model training
│   │   ├── model_evaluation.py      # Model evaluation and comparison
│   │   └── model_pusher.py          # Production model promotion
│   │
│   ├── entity/                      # Configuration and artifact dataclasses
│   │
│   ├── pipeline/                    # Pipeline orchestration modules
│   │   ├── train_pipeline.py        # Training pipeline execution
│   │   └── prediction_pipeline.py   # Prediction pipeline execution
│   │
│   ├── constants/                   # System-wide constants
│   │
│   ├── custom_exception.py          # Custom exception handling
│   │
│   └── custom_logger.py             # Application logging configuration
│
├── app.py                           # FastAPI application entry point
├── Dockerfile                       # Docker container configuration
├── requirements.txt                 # Python project dependencies
├── setup.py                         # Python package configuration
└── README.md                        # Project documentation
```

---

## 🚀 Local Setup & Execution Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/signature-recognition-mlops.git
cd signature-recognition-mlops
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username if necessary.

---

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment based on your operating system.

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
venv\Scripts\activate
```

#### Linux / macOS / Git Bash

```bash
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

### Step 4: Run the FastAPI Application

```bash
python app.py
```

Once the application is running, open:

```text
http://localhost:8080/docs
```

The FastAPI Swagger interface can be used to explore and test the available API endpoints.

---

## 🌐 API Endpoints

| Method | Endpoint   | Description                                                                     |
| :----- | :--------- | :------------------------------------------------------------------------------- |
| `GET`  | `/`        | Health check endpoint returning API status                                       |
| `GET`  | `/train`   | Asynchronously triggers the complete training and evaluation pipeline            |
| `POST` | `/predict` | Accepts a signature image and returns the predicted class and confidence score   |

---

## 🔄 Training API

The training endpoint triggers the complete MLOps workflow.

```text
GET /train
```

The training process is executed asynchronously using FastAPI `BackgroundTasks`.

This allows the API to respond without blocking the request while the training pipeline executes in the background.

The training workflow includes:

1. Data ingestion
2. Data transformation
3. Model training
4. Model evaluation
5. Model comparison
6. Model promotion when the newly trained model satisfies the evaluation criteria

---

## 🔍 Signature Prediction API

The prediction endpoint accepts a signature image and performs inference using the active production model.

```text
POST /predict
```

The API returns:

* Uploaded filename
* Predicted signature class
* Prediction confidence

### Example Response

```json
{
  "filename": "sample_signature.png",
  "prediction": "Genuine",
  "confidence": 0.9842
}
```

The prediction value can be:

```text
Genuine
```

or:

```text
Forged
```

---

## 🐳 Docker Containerization

The application is containerized using Docker to provide a consistent runtime environment across local development and cloud deployment.

The Docker configuration uses a Python-based environment suitable for running the FastAPI application and the PyTorch inference pipeline.

### Build Docker Image

```bash
docker build -t signature-recognition-mlops .
```

### Run Docker Container Locally

```bash
docker run -p 8080:8080 signature-recognition-mlops
```

Once running, access the application at:

```text
http://localhost:8080/docs
```

---

## ☁️ Deployment on Render

The application is configured for deployment using a Docker-based Render web service.

The deployment process includes:

1. Building the Docker container.
2. Installing project dependencies.
3. Installing the local Python package.
4. Starting the FastAPI application using Uvicorn.
5. Exposing the application through the Render-assigned service URL.

### Container Base Image

```text
python:3.10-slim
```

### Dependency Installation

```bash
pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt && pip install -e .
```

### Application Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## ⚡ CPU-Optimized Deployment

The Docker environment is configured with CPU-focused dependencies to reduce unnecessary resource usage during deployment.

This approach is useful for lightweight cloud environments with limited memory resources.

The deployment environment is designed to run:

* FastAPI application services
* Signature image preprocessing
* PyTorch model inference
* REST API requests

without requiring GPU infrastructure.

---

## 📏 Model Evaluation

The model evaluation component compares a newly trained model against the currently active production model.

The evaluation stage acts as a quality gate before model promotion.

```text
Newly Trained Model
        │
        ▼
Evaluate on Validation/Test Dataset
        │
        ▼
Compare With Active Production Model
        │
        ▼
Is New Model Better?
     │          │
    YES         NO
     │          │
     ▼          ▼
Promote      Keep Current
New Model    Production Model
```

This process helps prevent automatically replacing the active model with a newly trained model that does not meet the required evaluation criteria.

---

## 🧮 Model Evaluation & Inference Math

### Cross-Entropy Loss

The classification model uses Cross-Entropy Loss.

$$
\text{Loss} = -\sum_{c=1}^{2} y_c \log(\hat{y}_c)
$$

Where:

* $y_c$ represents the true class label.
* $\hat{y}_c$ represents the predicted probability for class $c$.

---

### Softmax Confidence Calculation

The model generates raw output scores called logits.

Softmax converts these logits into class probabilities.

$$
P(\text{Class}_i) =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

The class with the highest probability is selected as the final prediction.

The corresponding probability is returned as the confidence score.

---

## 📝 Logging & Exception Handling

The project includes custom logging and exception handling utilities.

### Custom Logger

The logging module provides centralized logging for pipeline execution and application events.

Logs can help track:

* Pipeline execution stages
* Model training events
* Data processing operations
* Application runtime events
* Errors and debugging information

### Custom Exception Handling

Custom exception handling provides additional context when application errors occur.

This can include information useful for debugging, such as the location where an exception occurred.

---

## 🎯 Project Highlights

* End-to-end MLOps architecture
* Modular Python project structure
* Offline signature verification
* Genuine versus Forged classification
* Transfer learning using ResNet-34
* Automated data transformation
* Automated model training
* Model evaluation quality gate
* Production model promotion workflow
* Asynchronous training using FastAPI `BackgroundTasks`
* REST API-based prediction service
* Docker containerization
* CPU-focused cloud deployment
* Render deployment configuration
* Custom logging and exception handling

---

## 🔮 Future Improvements

Possible future enhancements include:

* Automated CI/CD testing workflows
* Automated model retraining
* Model performance monitoring
* Data and model drift detection
* Experiment tracking and model versioning
* Model registry integration
* Additional signature verification architectures
* Hyperparameter optimization
* Authentication and API security
* Batch signature prediction
* Cloud-based artifact storage
* Advanced production monitoring and observability

---

## 📌 Note

This project is intended to demonstrate an end-to-end machine learning and MLOps workflow for offline signature verification, from dataset ingestion and model training to model evaluation, API serving, containerization, and cloud deployment.
