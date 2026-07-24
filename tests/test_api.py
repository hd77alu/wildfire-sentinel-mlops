import io
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
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


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