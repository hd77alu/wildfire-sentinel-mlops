# --- run tests with command: python -m pytest tests/test_api.py -v ---
import io
import zipfile
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Fixture that initializes TestClient with lifespan context execution."""
    with TestClient(app) as c:
        yield c


def create_test_image_bytes(format: str = "JPEG", size: tuple = (224, 224), color: tuple = (255, 0, 0)) -> bytes:
    """Helper utility to generate raw bytes for an image file in memory."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def create_test_zip_bytes(filenames=["test_1.jpg", "test_2.png"]) -> bytes:
    """Helper utility to generate raw bytes for a ZIP archive containing images in memory."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            img_format = "PNG" if fname.endswith(".png") else "JPEG"
            img_data = create_test_image_bytes(format=img_format)
            zf.writestr(fname, img_data)
    return zip_buf.getvalue()


# --- General & Health Checks ---

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


# --- Single Image Inference Tests ---

def test_predict_single_valid_image(client):
    img_bytes = create_test_image_bytes(format="JPEG")
    files = {"file": ("test_fire.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/predict?threshold=0.40", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["filename"] == "test_fire.jpg"
    assert data["predicted_class"] in ["wildfire", "nowildfire"]


def test_predict_single_invalid_file_type(client):
    files = {"file": ("document.txt", b"Hello, world!", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415


def test_predict_single_corrupted_image(client):
    files = {"file": ("corrupted.jpg", b"NOT_AN_IMAGE_BYTE_STREAM", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_single_payload_too_large(client):
    large_bytes = b"0" * (11 * 1024 * 1024)
    files = {"file": ("huge_image.jpg", large_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413


# --- Batch Image Inference Tests ---

def test_predict_batch_images(client):
    img1 = create_test_image_bytes(format="JPEG", color=(255, 0, 0))
    img2 = create_test_image_bytes(format="PNG", color=(0, 255, 0))
    
    files = [
        ("files", ("img1.jpg", img1, "image/jpeg")),
        ("files", ("img2.png", img2, "image/png")),
    ]
    
    response = client.post("/predict-batch?threshold=0.50", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_images"] == 2


# --- Retraining Dataset Ingestion & ZIP Extraction Tests ---

def test_upload_retrain_data_direct_images(client):
    img_bytes = create_test_image_bytes(format="JPEG")
    files = [("files", ("tile_01.jpg", img_bytes, "image/jpeg"))]
    
    response = client.post("/upload-retrain-data?label=wildfire", files=files)
    assert response.status_code == 200
    assert "message" in response.json()


def test_upload_retrain_data_zip_extraction(client):
    zip_bytes = create_test_zip_bytes(filenames=["wildfire_tile_1.jpg", "wildfire_tile_2.png"])
    files = [("files", ("dataset_batch.zip", zip_bytes, "application/zip"))]
    
    response = client.post("/upload-retrain-data?label=wildfire", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_upload_retrain_data_invalid_label(client):
    img_bytes = create_test_image_bytes(format="JPEG")
    files = [("files", ("tile_01.jpg", img_bytes, "image/jpeg"))]
    
    response = client.post("/upload-retrain-data?label=invalid_class", files=files)
    assert response.status_code in [400, 422]


def test_upload_retrain_data_corrupt_zip(client):
    corrupt_zip_bytes = b"PK\x03\x04CorruptDataHereNotAZipArchive"
    files = [("files", ("corrupt.zip", corrupt_zip_bytes, "application/zip"))]
    
    response = client.post("/upload-retrain-data?label=nowildfire", files=files)
    assert response.status_code in [400, 422, 500]


# --- Retraining Execution Endpoint Tests ---

def test_trigger_retraining_pipeline(client):
    response = client.post("/trigger-retraining")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "message" in data


# --- CORS Middleware Security Tests ---

def test_cors_allowed_origin(client):
    headers = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/predict", headers=headers)
    assert response.status_code == 200


def test_cors_disallowed_origin(client):
    headers = {
        "Origin": "http://untrusted-malicious-site.com",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/predict", headers=headers)
    assert response.headers.get("access-control-allow-origin") != "http://untrusted-malicious-site.com"