import os
import requests
import streamlit as st
from PIL import Image

# API Endpoint Configuration
API_BASE_URL = os.getenv("API_URL", "https://wildfire-sentinel-api-75b7885f.fastapicloud.dev")

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
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Insights", "Single Image Analysis", "Batch Processing", "Model Retraining"])

# --- TAB 1: DATASET INSIGHTS ---
with tab1:
    # CSS for image alignment
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            height: 280px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Decoding the Visual Signatures of Wildfire Detection")
    st.caption(
        "An analysis exploring the spectral, thermal, and"
        " spatial features driving our MobileNet model."
    )
    st.markdown("---")
    
    # Two-Column Grid for Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Active Flame Mask Ratio")
        st.image("assets/hsv_fire_ratio.png", width='stretch')
        st.info("""
        **Interpretation:**
        Isolating flame pixels via HSV thresholding shows distinct positive spikes in wildfire samples, providing strong activation signals for CNNs.
        """)

        st.subheader("3. Red Dominance Index")
        st.image("assets/red_dominance.png", width='stretch')
        st.info("""
        **Interpretation:**
        The $R / (G + B)$ ratio shows a clear class separation, shifting rightward for wildfires due to intense flame and smoke illumination.
        """)

    with col2:
        st.subheader("2. Edge Variance (Smoke Blur)")
        st.image("assets/edge_variance.png", width='stretch')
        st.info("""
        **Interpretation:**
        Lower Laplacian variance in wildfire frames proves dense smoke plumes obscure fine canopy details, serving as a key visual marker.
        """)

        st.subheader("4. Spectral RGB Distribution")
        st.image("assets/spectral_rgb_distribution.png", width='stretch')
        st.info("""
        **Interpretation:**
        Wildfires show elevated multi-channel intensity, driven by bright flame emissions and reflective smoke plumes.
        """)

    # Conclusion Section
    st.markdown("### Conclusion: What Story Does Our Dataset Tell?")
    st.info("""
        Wildfire classification relies on two main visual indicators: **chromatic shift** and **spatial texture disruption**. 
        
        * **Thermal & Color Markers:** Active combustion shifts light emission toward the warm end of the optical spectrum. Non-fire images display balanced vegetation tones, whereas wildfire imagery is marked by high red-spectrum saturation and localized flame pixels.
        * **Atmospheric Smoke Masking:** Dense smoke plumes smooth out fine spatial details (leaves, canopy lines, terrain), reducing image edge variance. 
        
        *Together, these patterns confirm that our model leverages both color intensity shifts and texture-smoothing features to accurately classify wildfire threats.*
        """)
    
    st.markdown("---")

# TAB 2: Single Image
with tab2:
    st.subheader("Single Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload a satellite or aerial image file (.jpg, .png), max size 10MB.",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="Source Image", width='stretch')

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


# TAB 3: Batch Processing
with tab3:
    st.subheader("Batch Processing")
    batch_files = st.file_uploader(
        "Upload multiple image files for rapid screening (.jpg, .png), max size 10MB.",
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
                                    st.image(matching_file, width='stretch')

                                is_fire = item["predicted_class"].lower() in ["wildfire", "fire"]
                                badge = "WILDFIRE DETECTED" if is_fire else "CLEAR"
                                
                                st.markdown(f"**Filename:** {item['filename']}")
                                st.markdown(f"**Status:** {badge} ({item['confidence']*100:.1f}%)")
                                st.divider()
                    else:
                        st.error(f"Batch Processing Error ({response.status_code}): {response.json().get('detail')}")

                except Exception as e:
                    st.error(f"Batch request failed: {str(e)}")

# TAB 4: Model Retraining
with tab4:
    st.subheader("Dataset Ingestion & Model Fine-Tuning")
    st.caption("Upload raw satellite image tiles (.jpg, .png up to 10MB) or a compressed dataset archive (.zip up to 100MB).")

    col_up1, col_up2 = st.columns([2, 1])

    with col_up1:
        retrain_files = st.file_uploader(
            "Upload Bulk Retraining Images or ZIP Archive",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
            key="retrain_uploader",
            help="Accepted formats: JPG, PNG, or ZIP archives containing image tiles."
        )

    with col_up2:
        label_selection = st.selectbox(
            "Target Class Label",
            ["wildfire", "no_wildfire"],
            help="Ground truth category for the uploaded dataset samples."
        )

        if st.button("Upload to Database", type="primary"):
            if retrain_files:
                with st.spinner("Extracting and storing dataset samples..."):
                    payload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in retrain_files
                    ]
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/upload-retrain-data?label={label_selection}",
                            files=payload
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.success(data["message"])
                        else:
                            st.error(f"Upload failed (Status {res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Could not connect to API backend: {e}")
            else:
                st.warning("Please select at least one file or ZIP archive to upload.")

    st.divider()
    st.subheader("Trigger Pipeline Execution")
    st.caption("Preprocesses new database entries and unfreezes pretrained transfer learning layers.")

    if st.button("Start Retraining Pipeline"):
        with st.spinner("Initiating background retraining worker..."):
            try:
                res = requests.post(f"{API_BASE_URL}/trigger-retraining")
                if res.status_code == 200:
                    st.info("Retraining job queued successfully in the background.")
                else:
                    st.error("Failed to trigger retraining process.")
            except Exception as e:
                st.error(f"Could not connect to API backend: {e}")