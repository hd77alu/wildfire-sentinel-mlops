# Wildfire-Sentinel-MLOps
This project, Wildfire-Sentinel, focuses on the development and optimization of Deep Learning models designed to identify wildfires from high-resolution satellite imagery datasets.

## Project Objectives
- Automated Detection: Build a robust computer vision pipeline to classify satellite images into 'Wildfire' or 'No Wildfire' categories.
- Architectural Benchmarking: Compare custom Convolutional Neural Networks (CNNs) against pre-trained Transfer Learning architectures.
- Model Optimization: Evaluate the impact of different hyper parameter tuning techniques on model generalization.
- Reliability for Deployment: Achieve high recall, precision and AUC-ROC scores to ensure the system is reliable for real-world monitoring with minimal false alarms.

## Repository Structure

```
wildfire-sentinel-mlops/
│
├── .streamlit/                             # 
├── assets/                                 # 
├── a-video-demo/                           #
├── database/                               #
├── models/                                 # 
├── notebook/                               #
│ 
├── src/                                    # 
│   ├── api.py                              #
│   └── app.py                              #
│   └── database.py                         #
│   └── prediction.py                       #
│   └── preprocessing.py                    #
│   └── retrain.py                          #
│ 
├── tests/                                  #  
│
├── wildfire-dataset/                       # 
│   ├── test                                
│   └── train                               
│   └── valid                               
│
└── .dockerignore                           #
└── .gitattributes                          #
└── .gitignore                              #
└── docker-compose.yml                      #
└── Dockerfile.api                          #
└── Dockerfile.ui                           #
└── README.md                               #
└── requirements.txt                        #
```

## Set-up

### 1. Clone or Download the Repository

```bash
git clone https://github.com/hd77alu/wildfire-sentinel-mlops
cd wildfire-sentinel-mlops
```
## Running the Local Application

### Method 1: Terminal

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
Access Services:
Streamlit UI: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
```

### 3. Stop containers:
```bash
docker-compose down
```