import os
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image

# Global Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def preprocess_single_image(image_input, target_size=IMG_SIZE):
    """
    Preprocesses a single image (filepath, PIL Image, or Bytes) into a batch tensor
    suitable for model inference.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif hasattr(image_input, "read"):  # Handles file-like streams (Streamlit/FastAPI)
        img = Image.open(image_input)
    else:
        raise ValueError("Unsupported image input type. Expected file path or PIL Image.")

    # Ensure 3-channel RGB image
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to model input dimensions
    img = img.resize(target_size)

    # Convert to array and expand dimensions for batching -> (1, 224, 224, 3)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch


def create_datasets(data_dir="wildfire-dataset", batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Loads train, validation, and test tf.data.Datasets from the directory structure.
    Applies performance optimization (prefetch).
    """
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    print(f"Loading datasets from {data_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="binary",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Class names detected: {class_names}")

    # Prefetch optimization
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=2)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def process_uploaded_zip(zip_file_stream, extract_to="wildfire-dataset/train"):
    """
    Processes an uploaded ZIP archive containing new training images for retraining.
    Extracts files directly to the training directory.
    """
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_stream, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print(f"Extracted uploaded batch to {extract_to}")
    return f"Extraction complete. Files saved to {extract_to}."