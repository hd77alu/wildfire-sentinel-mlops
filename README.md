# Wildfire-Sentinel-MLOps
## Project Overview
This project, Wildfire-Sentinel, is a scalable, containerized MLOps pipeline for real-time wildfire detection from satellite imagery, built with FastAPI, Streamlit, MobileNet, and Docker Compose.

- You can access demo video from [here.](https://somup.com/cOiUVmVngAd)

## Table of Content

- [Public Deployment URLs](#public-deployment-urls)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Set-up](#set-up)
- [Model Evaluation Metrics](#model-evaluation-metrics-results)
- [Flood Request Results](#results-from-flood-request-simulation-via-locust)
- [FastAPI](#fastapi)
- [Streamlit UI](#streamlit-ui)

## Public Deployment URLs
- [Wildfire Sentinel API Documentation](https://wildfire-sentinel-api-75b7885f.fastapicloud.dev/docs) (deployed via FastAPI Cloud).
- [Streamlit UI](https://wildfire-sentinel-mlops-6gpsxpjak9k96osnaisjvg.streamlit.app) (deployed via Streamlit Community Cloud).

## Dataset
This project uses the Wildfire Prediction Dataset (Satellite Images) by Abdelghani Aaba.

- Source: [Kaggle Datasets](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)

### Dataset characteristics
This dataset contains satellite images (350x350px) in 2 classes :

- Wildfire: 22710 images
- No wildfire: 20140 images

The data was divided into train, test, and validation with these percentages :

- Train: ~70%
- Test: ~15%
- Validation: ~15%

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

### 3. Run Streamlit (Second Terminal)
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

## Model Evaluation Metrics Results
[![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd77alu/wildfire-sentinel-mlops/blob/main/notebook/wildfire_sentinel.ipynb)

The best model (**MobileNetV2**) was evaluated on the test dataset, achieving strong classification performance across all key metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.9735** (97.35%) |
| **Precision** | **0.9926** (99.26%) |
| **Recall** | **0.9592** (95.92%) |
| **F1-Score** | **0.9756** (97.56%) |
| **AUC** | **0.9984** (99.84%) |

> **Key Takeaway:** The model achieved strong Accuracy and Recall, high **Precision (0.9926)**, which minimizes false positives, while an **AUC of 0.9984** demonstrates near-optimal class separation between wildfire and non-wildfire imagery.

## Results from Flood Request Simulation via Locust
![Locust Charts for single API Replica](https://github.com/hd77alu/wildfire-sentinel-mlops/blob/4a3229ff095755b748f52d08eff6ce49da36cb97/assets/report1-locust-charts.png)
> Figure 1: Single API Replica (scale api=1)

![Locust Charts for Load-Balanced Cluster](https://github.com/hd77alu/wildfire-sentinel-mlops/blob/4a3229ff095755b748f52d08eff6ce49da36cb97/assets/report2-locust-charts.png)
> Figure 2: Load-Balanced Cluster (scale api=3)

### Comparison Table

| Metric | Single API Replica (`scale api=1`) | Load-Balanced Cluster (`scale api=3`) | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Failures / Errors** | **High Failure Rate** (Failures/s tracks directly with total RPS during load spikes) | **0 Failures/s** (Flat red line across the entire run) | Scaling eliminated the 502 Bad Gateway / upstream drop bottleneck. |
| **Throughput (RPS)** | Fluctuates heavily between 0 and 15 RPS (most total RPS counted as failures) | **Sustained ~8–10 RPS** of purely successful requests | Traffic delivery smoothed out into a stable, reliable stream. |
| **Median Latency (50th %)** | 3,100 ms – 6,500 ms (Spike at 12:23:36 AM shows 3,100 ms) | **500 ms – 840 ms** (Tooltip shows 840 ms at peak ramp-up) | **~75%–85% reduction** in median latency for end users. |
| **Tail Latency (95th %)** | 5,100 ms – 11,000 ms (Tooltip shows 5,100 ms at 12:23:36 AM) | **1,200 ms – 2,800 ms** (Tooltip shows 2,400 ms) | Eliminates severe worst-case queue delays under full concurrency. |

**Interpretation:** From Figures (1-2), From Figures (1-2), we observed that under a sustained load of 20 concurrent users for 1 minute, a single API container experiences complete process saturation due to the CPU-intensive nature of MobileNet image inference. This queue backlog causes incoming client requests to time out (HTTP 499), forcing Nginx to return HTTP 502 Bad Gateway errors as the backend worker becomes unresponsive, resulting in severe latency spikes up to 11,000 ms. Horizontally scaling our backend architecture to three API replicas behind Nginx resolves this single-node bottleneck by distributing inference tasks across independent container instances, restoring system stability to a 0% failure rate and maintaining 50th-percentile latencies well under 1 second.

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

![API Tests Results](https://github.com/hd77alu/wildfire-sentinel-mlops/blob/4a3229ff095755b748f52d08eff6ce49da36cb97/assets/api-test-results.png)

## Streamlit UI
Our Streamlit UI contains four tabs- Dataset Insights, Single Image Analysis, Batch Processing, and Model Retraining- as visible below:

![Streamlit UI](https://github.com/hd77alu/wildfire-sentinel-mlops/blob/4a3229ff095755b748f52d08eff6ce49da36cb97/assets/wildfire-streamlit-ui.png)
