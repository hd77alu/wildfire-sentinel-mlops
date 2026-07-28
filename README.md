# Wildfire-Sentinel-MLOps
## Project Overview
This project, Wildfire-Sentinel,is a scalable, containerized MLOps pipeline for real-time wildfire detection from aerial imagery, built with FastAPI, Streamlit, MobileNet, and Docker Compose.

## Public Deployments URLs
- [Wildfire Sentinel API Documentation](https://wildfire-sentinel-api-75b7885f.fastapicloud.dev/docs#/) (deployed via FastAPI Cloud).
- [Streamlit UI](https://wildfire-sentinel-mlops-6gpsxpjak9k96osnaisjvg.streamlit.app/) (deployed via Streamlit Community Cloud).

## Dataset
This project uses Wildfire Prediction Dataset (Satellite Images) by Abdelghani Aaba.

- Source: [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)

### Dataset characteristics
This dataset contains satellite images (350x350px) in 2 classes :

- Wildfire : 22710 images
- No wildfire : 20140 images

The data was divided into train, test and validation with these percentages :

- Train : ~70%
- Test : ~15%
- Validation : ~15%

## Repository Structure

```
wildfire-sentinel-mlops/
│
├── .streamlit/                             # UI Configuration
├── assets/                                 # Media Files
├── a-video-demo/                           # Demonstration Video
├── database/                               # SQLite Storage
├── models/                                 # Trained model
├── notebook/                               # Jupyter Notebook
│ 
├── src/                                    # Source Code
│   ├── api.py                              # Backend Endpoints
│   ├── app.py                              # Frontend Dashboard
│   ├── database.py                         # SQLite Operations
│   ├── prediction.py                       # Model Inference
│   ├── preprocessing.py                    # Data Transformation
│   └── retrain.py                          # Model Retraining
│ 
├── tests/                                  # Integration Tests
│
├── wildfire-dataset/                       # Image Data
│   ├── test                                # Evaluation Images
│   ├── train                               # Training Images
│   └── valid                               # Validation Images
│
└── .dockerignore                           # Docker Exclusions
└── .gitattributes                          # Git Attribute
└── .gitignore                              # Untracked Files
└── docker-compose.yml                      # Container Orchestration
└── Dockerfile.api                          # API Container
└── Dockerfile.ui                           # UI Container
└── locustfile.py                           # Load Testing
└── nginx.conf                              # Load Balancer
└── README.md                               # Project Overview
└── requirements.txt                        # Dependencies List
```

## Set-up

### 1. Clone the Repository

```bash
git clone https://github.com/hd77alu/wildfire-sentinel-mlops
cd wildfire-sentinel-mlops
```
### 2. Create and Activate Virtual Environment (With Python 3.12.10)
```bash
# Create virtual environment specifying Python 3.12.10
python3.12 -m venv venv

# Activate the environment
source venv/bin/activate
```

## Running the Local Application

### Method 1: Two Terminals

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API (First Terminal)

```bash
uvicorn src.api:app --reload --port 8000
```

### 3. Run Streamlit (Seconed Terminal)
```bash
# Ensure the virtual environment is also activated in your second terminal
streamlit run src/app.py
```

### Method 2: Containerized Stack

> Prerequisites: Ensure Docker Desktop is installed

### 1. Build and start containers in detached mode:
```bash
docker-compose up --build -d
```

### 2. Verify running services:
```bash
docker-compose ps
```
### 3.Access Services:
```bash
Streamlit UI: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
```

### 4. Stop containers:
```bash
docker-compose down
```

## Results from Flood Request Simulation via Locust
image
> Figure 1: Single API Replica (scale api=1)

image
> Figure 2: Load-Balanced Cluster (scale api=3)

**Interpretation:** From Figures (1-2), From Figures (1-2), we observed that under a sustained load of 20 concurrent users for 1 minute, a single API container experiences complete process saturation due to the CPU-intensive nature of MobileNet image inference. This queue backlog causes incoming client requests to time out (HTTP 499), forcing Nginx to return HTTP 502 Bad Gateway errors as the backend worker becomes unresponsive, resulting in severe latency spikes up to 11,000 ms. Horizontally scaling our backend architecture to three API replicas behind Nginx resolves this single-node bottleneck, distributing inference tasks across independent container instances, restoring system stability to a 0% failure rate, and maintaining 50th percentile latencies well under 1 second.

## FastAPI
The following endpoints have been implemented for our FastAPI:

| Endpoint | Method | Request Payload | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | *None* | System health check and model status verification. |
| `/predict` | `POST` | `file` (Image multipart/form-data) | Performs inference on a single image file for wildfire detection. |
| `/predict-batch` | `POST` | `files` (List of image files) | Performs batch inference on multiple image files simultaneously. |
| `/upload-retrain-data` | `POST` | `files` (Images), `label` (Query param) | Uploads labeled images to augment dataset for future retraining. |
| `/trigger-retraining` | `POST` | `epochs`, `learning_rate` (JSON body) | Triggers background model retraining pipeline on updated dataset. |

### API Tests
Our API test suite inside [tests/test_api.py](tests/test_api.py) evaluates system reliability across 14 automated unit and integration test cases using `pytest`. The test suite validates endpoint health, single/batch inference pipelines, image preprocessing error handling (invalid files, oversized payloads, corrupted uploads), retraining workflows, and CORS security configurations — achieving a **100% pass rate**.

image

## Streamlit UI
Our Streamlit UI contains four tabs- Dataset Insights, Single Image Analysis, Batch Processing, and Model Retraining- as visible below:

 image
