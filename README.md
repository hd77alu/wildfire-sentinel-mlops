# Wildfire-Sentinel-MLOps
This project, Wildfire-Sentinel, focuses on the development and optimization of Deep Learning models designed to identify wildfires from high-resolution satellite imagery datasets.

## Project Objectives
- Automated Detection: Build a robust computer vision pipeline to classify satellite images into 'Wildfire' or 'No Wildfire' categories.
- Architectural Benchmarking: Compare custom Convolutional Neural Networks (CNNs) against pre-trained Transfer Learning architectures.
- Model Optimization: Evaluate the impact of different hyper parameter tuning techniques on model generalization.
- Reliability for Deployment: Achieve high recall, precision and AUC-ROC scores to ensure the system is reliable for real-world monitoring with minimal false alarms.

## Set-up

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