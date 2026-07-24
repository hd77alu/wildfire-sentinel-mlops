import os
import keras
import numpy as np
from .preprocessing import preprocess_single_image

# Point directly to our modern .keras model file (.h5 file was causing issues when loading the model)
DEFAULT_MODEL_PATH = os.path.join("models", "mobilenet_wildfire_model.keras")
CLASS_NAMES = ["nowildfire", "wildfire"]
CONFIDENCE_THRESHOLD = float(os.getenv("WILDFIRE_THRESHOLD", 0.40)) # lower threshold to increase sensitivity for wildfire detection


class WildfirePredictor:
    """Inference wrapper for loading our wildfire detection model and making predictions."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Loads our saved Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Ensure our model file exists in the 'model/' directory."
            )
        
        print(f"Loading Wildfire Sentinel model from {self.model_path}...")
        model = keras.models.load_model(self.model_path, compile=False)
        print("✓ Model loaded successfully.")
        return model

    def predict_image(self, image_input, threshold=CONFIDENCE_THRESHOLD):
        """Runs inference on a single image (file path, PIL Image, or numpy array)."""
        img_tensor = preprocess_single_image(image_input)
        raw_output = self.model.predict(img_tensor, verbose=0)
        
        raw_score = float(raw_output[0][0]) if raw_output.ndim > 1 else float(raw_output[0])

        if raw_score >= threshold:
            predicted_class = "wildfire"
            confidence = raw_score
        else:
            predicted_class = "nowildfire"
            confidence = 1.0 - raw_score

        return {
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "raw_score": round(raw_score, 4)
        }

    def predict_batch(self, image_tensors, threshold=CONFIDENCE_THRESHOLD):
        """Runs batch inference on a preprocessed tensor batch."""
        raw_outputs = self.model.predict(image_tensors, verbose=0)
        results = []

        for score in raw_outputs.flatten():
            score = float(score)
            p_class = "wildfire" if score >= threshold else "nowildfire"
            conf = score if p_class == "wildfire" else (1.0 - score)
            results.append({
                "class": p_class,
                "confidence": round(conf, 4),
                "raw_score": round(score, 4)
            })

        return results