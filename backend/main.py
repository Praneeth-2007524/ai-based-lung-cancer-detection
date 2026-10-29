
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "lung_cancer_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

MAX_FILE_SIZE = 15 * 1024 * 1024

# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AI-Based Lung Cancer Detection API",
    description=(
        "Educational CT image classification API "
        "using a trained TensorFlow CNN model."
    ),
    version="1.0.0",
)

# --------------------------------------------------
# 3. CORS CONFIGURATION
# --------------------------------------------------

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()

if frontend_origin:
    origins.extend(
        origin.strip()
        for origin in frontend_origin.split(",")
        if origin.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------

model = None
model_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH))

        print("Model loaded successfully.")
        print("Model input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)

        # Check whether the model matches this API.
        input_shape = model.input_shape
        output_shape = model.output_shape

        if (
            len(input_shape) != 4
            or input_shape[-1] != 1
            or tuple(input_shape[1:3]) != IMAGE_SIZE
        ):
            raise ValueError(
                "The model must accept grayscale images "
                "with shape (224, 224, 1)."
            )

        if output_shape[-1] != len(CLASS_NAMES):
            raise ValueError(
                "The model must return three class scores: "
                "Benign, Malignant, and Normal."
            )

    else:
        model_load_error = (
            f"Model file not found: {MODEL_PATH}"
        )
        print(model_load_error)

except Exception as exc:
    model = None
    model_load_error = str(exc)
    print("Model loading failed:", model_load_error)

# --------------------------------------------------
# 5. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "AI-Based Lung Cancer Detection",
        "service": "FastAPI",
        "status": "ready" if model is not None else "model_missing",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict",
        "documentation": "/docs",
        "model_loaded": model is not None,
    }

# --------------------------------------------------
# 6. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "model_missing",
        "model_loaded": model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "classes": CLASS_NAMES,
        "detail": (
            None if model is not None else model_load_error
        ),
    }

# --------------------------------------------------
# 7. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """
    Convert the uploaded image to grayscale,
    resize to 224x224, normalize pixels to [0, 1],
    and create a batch for CNN prediction.
    """

    try:
        image = Image.open(io.BytesIO(raw_bytes))

        # Match the grayscale training pipeline.
        image = image.convert("L")

        # Resize to the model's expected image size.
        image = image.resize(IMAGE_SIZE)

        # Convert pixels to float values from 0 to 1.
        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        # Shape: (224, 224) -> (1, 224, 224, 1)
        image_array = np.expand_dims(
            image_array,
            axis=(0, -1),
        )

        return image_array

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read this image. "
                "Please upload a valid JPG, PNG, or WebP image."
            ),
        ) from exc

# --------------------------------------------------
# 8. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Confirm the model loaded successfully.
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model is not available. "
                "Check backend/models/lung_cancer_model.keras "
                "and the server logs."
            ),
        )

    # Validate file type.
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    # Read the uploaded file safely.
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 15 MB.",
        )

    # Prepare the image for the CNN.
    image_batch = preprocess_image(raw_bytes)

    try:
        # Run model inference.
        predictions = model.predict(
            image_batch,
            verbose=0,
        )

        scores = np.asarray(
            predictions[0],
            dtype=np.float64,
        ).ravel()

        if scores.size != len(CLASS_NAMES):
            raise ValueError(
                "The model did not return exactly three class scores."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "The model returned invalid prediction values."
            )

        # Softmax models normally return probabilities.
        # For logits, convert them to probabilities.
        if (
            np.all(scores >= 0)
            and np.all(scores <= 1)
            and np.isclose(np.sum(scores), 1.0, atol=1e-3)
        ):
            probabilities = scores
        else:
            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)
            probabilities = exp_scores / np.sum(exp_scores)

        # Find the class with the highest score.
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(probabilities[predicted_index])

        class_scores = {
            name: round(float(score), 4)
            for name, score in zip(
                CLASS_NAMES,
                probabilities,
            )
        }

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "scores": class_scores,
            "notice": (
                "Educational research output only. "
                "This result is not a medical diagnosis. "
                "Do not use it to make health decisions."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Prediction error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

# --------------------------------------------------
# 9. RUN DIRECTLY (OPTIONAL)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
