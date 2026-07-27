#!/bin/bash

# 1. Start FastAPI backend in the background on port 8000
echo "Starting FastAPI backend..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait 3 seconds for the API & model to initialize
sleep 3

# 2. Start Streamlit frontend on port 7860 (HF Spaces public port)
# Note: Change 'src/app.py' to 'src/ui.py' if your Streamlit file is named ui.py
echo "Starting Streamlit frontend on port 7860..."
exec streamlit run src/app.py --server.port=7860 --server.address=0.0.0.0