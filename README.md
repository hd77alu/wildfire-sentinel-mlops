# Wildfire-Sentinel-MLOps

This branch is used for Wildfire Sentinel deployment.

- The API was deployed via FastApI Cloud: You can access the Wildfire Sentinel API Documentation from [here.](https://wildfire-sentinel-api-75b7885f.fastapicloud.dev/docs)
- The Streamlit UI was deployed via Streamlit Community Cloud: You can access the UI from [here.](https://wildfire-sentinel-mlops-6gpsxpjak9k96osnaisjvg.streamlit.app/)

## Repository Structure

```
wildfire-sentinel-mlops/
│
├── .streamlit/        # Streamlit config                       
├── assets/                                  
├── database/                               
├── models/                                  
│ 
├── src/                                     
│   ├── api.py         # FastAPI                    
│   └── app.py         # Streamlit UI                   
│   └── database.py                         
│   └── prediction.py                       
│   └── preprocessing.py                    
│   └── retrain.py                            
│                                                  
└── .gitignore
└── pyproject.toml     # FastAPI deployment config                                                  
└── README.md                               
└── requirements.txt                                                     
```
