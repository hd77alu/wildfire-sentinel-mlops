import io
import os
import zipfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.database import init_db, log_sample
from src.retrain import preprocess_and_retrain
from src.prediction import WildfirePredictor

MAX_INFERENCE_FILE_SIZE = 10 * 1024 * 1024   # 10 MB limit for inference images
MAX_RETRAIN_PAYLOAD_SIZE = 100 * 1024 * 1024 # 100 MB limit for retraining ZIP/bulk uploads
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

predictor: Optional[WildfirePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Initializing Wildfire Sentinel API...")
    try:
        init_db()
        predictor = WildfirePredictor()
        print("WildfirePredictor loaded successfully.\n")
    except Exception as e:
        print(f"Failed to initialize API resources: {e}")
        raise e
    yield
    print("Shutting down Wildfire Sentinel API...")


app = FastAPI(
    title="Wildfire Sentinel API",
    description="Secure REST API for real-time wildfire detection and model retraining.",
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
    filename = file.filename or "uploaded_image"

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {list(ALLOWED_IMAGE_TYPES)}",
        )

    contents = await file.read()

    if len(contents) > MAX_INFERENCE_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File '{filename}' exceeds maximum allowed payload limit of 10MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file: '{filename}'",
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
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not initialized.",
        )

    filename = file.filename or "uploaded_image"
    image = await validate_uploaded_image(file)

    try:
        result = predictor.predict_image(image, threshold=threshold)

        pred_class = result.get("class") or result.get("predicted_class") or "unknown"
        confidence = result.get("confidence", 0.0)

        return PredictionResult(
            filename=filename,
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
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not initialized.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in the batch request.",
        )

    results = []
    for file in files:
        filename = file.filename or "uploaded_image"
        try:
            image = await validate_uploaded_image(file)
            res = predictor.predict_image(image, threshold=threshold)

            pred_class = res.get("class") or res.get("predicted_class") or "unknown"
            confidence = res.get("confidence", 0.0)

            results.append(
                PredictionResult(
                    filename=filename,
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


ALLOWED_LABELS = {"wildfire", "no_wildfire"}
@app.post("/upload-retrain-data")
async def upload_retrain_data(
    files: list[UploadFile] = File(...),
    label: str = Query(..., description="Target class label for retraining data")
):
    if label not in ALLOWED_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{label}'. Allowed labels are: {', '.join(ALLOWED_LABELS)}"
        )
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for retraining upload.",
        )

    raw_dir = "database/retrain_raw"
    os.makedirs(raw_dir, exist_ok=True)
    saved_count = 0

    for file in files:
        filename = file.filename or "uploaded_file"
        contents = await file.read()

        if len(contents) > MAX_RETRAIN_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File '{filename}' exceeds maximum allowed payload limit of 100MB.",
            )

        # Handle ZIP Archive Extraction
        if filename.endswith(".zip") or file.content_type in {"application/zip", "application/x-zip-compressed"}:
            try:
                with zipfile.ZipFile(io.BytesIO(contents)) as z:
                    for zip_info in z.infolist():
                        if zip_info.is_dir() or zip_info.filename.startswith("__MACOSX"):
                            continue

                        ext = os.path.splitext(zip_info.filename)[1].lower()
                        if ext in ALLOWED_IMAGE_EXTENSIONS:
                            extracted_name = os.path.basename(zip_info.filename)
                            file_path = os.path.join(raw_dir, extracted_name)

                            with open(file_path, "wb") as f:
                                f.write(z.read(zip_info.filename))

                            log_sample(filename=extracted_name, filepath=file_path, label=label)
                            saved_count += 1

            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupted or invalid ZIP archive: '{filename}'",
                )

        # Handle Individual Image Files
        elif file.content_type in ALLOWED_IMAGE_TYPES or os.path.splitext(filename)[1].lower() in ALLOWED_IMAGE_EXTENSIONS:
            file_path = os.path.join(raw_dir, filename)
            with open(file_path, "wb") as f:
                f.write(contents)

            log_sample(filename=filename, filepath=file_path, label=label)
            saved_count += 1

    return {
        "message": f"Successfully extracted and indexed {saved_count} image samples for retraining.",
        "label_assigned": label,
        "destination": raw_dir,
    }


@app.post("/trigger-retraining", tags=["Retraining"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(preprocess_and_retrain)
    return {
        "status": "queued",
        "message": "Model retraining pipeline triggered successfully in the background.",
    }