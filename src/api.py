import io
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.prediction import WildfirePredictor

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB payload limit
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

predictor: WildfirePredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Initializing Wildfire Sentinel API...")
    try:
        predictor = WildfirePredictor()
        print("WildfirePredictor loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load WildfirePredictor model: {e}")
        raise e
    yield
    print("Shutting down Wildfire Sentinel API...")


app = FastAPI(
    title="Wildfire Sentinel API",
    description="Secure REST API for real-time wildfire detection from satellite and aerial imagery.",
    version="1.0.0",
    lifespan=lifespan,
)

raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8000"
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# --- Pydantic Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_path: str = Field(..., json_schema_extra={"example": "models/mobilenet_wildfire_model.keras"})


class PredictionResult(BaseModel):
    filename: str = Field(..., json_schema_extra={"example": "satellite_capture_01.jpg"})
    predicted_class: str = Field(..., json_schema_extra={"example": "wildfire"})
    confidence: float = Field(..., json_schema_extra={"example": 0.9842})


class BatchPredictionResponse(BaseModel):
    total_images: int = Field(..., json_schema_extra={"example": 2})
    predictions: List[PredictionResult]


# --- Payload Validation Helper ---
async def validate_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {list(ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File '{file.filename}' exceeds maximum allowed payload limit of 10MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file: '{file.filename}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating image file: {str(e)}",
        )


# --- Endpoints ---

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to the Wildfire Sentinel API",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unhealthy.",
        )

    return HealthResponse(
        status="healthy",
        model_path=predictor.model_path,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict_single(
    file: UploadFile = File(...),
    threshold: float = Query(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for wildfire classification.",
    ),
):
    image = await validate_uploaded_image(file)

    try:
        result = predictor.predict_image(image, threshold=threshold)

        pred_class = result.get("class") or result.get("predicted_class") or "unknown"
        confidence = result.get("confidence", 0.0)

        return PredictionResult(
            filename=file.filename,
            predicted_class=pred_class,
            confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Query(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for batch wildfire classification.",
    ),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in the batch request.",
        )

    results = []
    for file in files:
        try:
            image = await validate_uploaded_image(file)
            res = predictor.predict_image(image, threshold=threshold)

            pred_class = res.get("class") or res.get("predicted_class") or "unknown"
            confidence = res.get("confidence", 0.0)

            results.append(
                PredictionResult(
                    filename=file.filename,
                    predicted_class=pred_class,
                    confidence=confidence,
                )
            )
        except HTTPException:
            continue
        except Exception:
            continue

    return BatchPredictionResponse(
        total_images=len(results),
        predictions=results,
    )