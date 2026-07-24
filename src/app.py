import os
import requests
import streamlit as st
from PIL import Image

# API Endpoint Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Wildfire Sentinel - Early Detection System",
    layout="wide",
)

st.title("Wildfire Sentinel")
st.markdown("Automated wildfire detection from satellite and aerial imagery, powered by deep learning models.")

# --- Sidebar Configuration ---
st.sidebar.header("Inference Settings")

# Check API Health
try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend API Status: Online")
    else:
        st.sidebar.error("Backend API Status: Unhealthy")
except requests.exceptions.RequestException:
    st.sidebar.error("Backend API Status: Offline")
    st.sidebar.caption(f"Ensure the FastAPI service is running at `{API_BASE_URL}`")

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05,
    help="Confidence score threshold required to classify an image as a wildfire.",
)

# --- Main App Tabs ---
tab1, tab2 = st.tabs(["Single Image", "Batch Processing"])

# TAB 1: Single Image
with tab1:
    st.subheader("Single Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload a satellite or aerial image file (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="Source Image", use_container_width=True)

        with col2:
            st.markdown("### Model Assessment")
            if st.button("Run Detection", type="primary", key="btn_single"):
                with st.spinner("Executing inference..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        response = requests.post(
                            f"{API_BASE_URL}/predict?threshold={threshold}",
                            files=files,
                            timeout=10,
                        )

                        if response.status_code == 200:
                            data = response.json()
                            pred_class = data["predicted_class"].upper()
                            confidence = data["confidence"]

                            if pred_class in ["WILDFIRE", "FIRE"]:
                                st.error(f"Classification Result: {pred_class} Detected")
                            else:
                                st.success(f"Classification Result: {pred_class}")

                            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                            st.progress(float(confidence))

                            st.json(data)
                        else:
                            st.error(f"API Error ({response.status_code}): {response.json().get('detail')}")

                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")


# TAB 2: Batch Processing
with tab2:
    st.subheader("Batch Image Analysis")
    batch_files = st.file_uploader(
        "Upload multiple image files for rapid screening",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.info(f"{len(batch_files)} image file(s) loaded for analysis.")

        if st.button("Process Batch", type="primary", key="btn_batch"):
            with st.spinner(f"Processing {len(batch_files)} image files..."):
                try:
                    payload = []
                    for f in batch_files:
                        f.seek(0)
                        payload.append(("files", (f.name, f.getvalue(), f.type)))

                    response = requests.post(
                        f"{API_BASE_URL}/predict-batch?threshold={threshold}",
                        files=payload,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        batch_res = response.json()
                        st.success(f"Successfully processed {batch_res['total_images']} images.")

                        cols = st.columns(3)
                        for idx, item in enumerate(batch_res["predictions"]):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                matching_file = next((f for f in batch_files if f.name == item["filename"]), None)
                                if matching_file:
                                    matching_file.seek(0)
                                    st.image(matching_file, use_container_width=True)

                                is_fire = item["predicted_class"].lower() in ["wildfire", "fire"]
                                badge = "WILDFIRE DETECTED" if is_fire else "CLEAR"
                                
                                st.markdown(f"**Filename:** {item['filename']}")
                                st.markdown(f"**Status:** {badge} ({item['confidence']*100:.1f}%)")
                                st.divider()
                    else:
                        st.error(f"Batch Processing Error ({response.status_code}): {response.json().get('detail')}")

                except Exception as e:
                    st.error(f"Batch request failed: {str(e)}")