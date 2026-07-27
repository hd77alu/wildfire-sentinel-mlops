# Wildfire-Sentinel-MLOps
This branch will be used for deployment.

## Repository Structure

```
wildfire-sentinel-mlops/
│
├── .streamlit/                              
├── assets/                                  
├── database/                               
├── models/                                  
│ 
├── src/                                     
│   ├── api.py                              
│   └── app.py                              
│   └── database.py                         
│   └── prediction.py                       
│   └── preprocessing.py                    
│   └── retrain.py                            
│                         
└── .gitattributes                          
└── .gitignore
└── pyproject.toml                                                          
└── README.md                               
└── requirements.txt                                                     
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