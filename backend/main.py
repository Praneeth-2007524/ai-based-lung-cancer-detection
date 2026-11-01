
import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PDFImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lung_cancer_model.keras"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
MALIGNANT_INDEX = 1
MAX_FILE_SIZE = 15 * 1024 * 1024

# Demonstration-only thresholds, NOT clinical risk thresholds.
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60

app = FastAPI(
    title="AI-Based Lung Cancer Detection API",
    description=(
        "Educational CT image classification with experimental "
        "scoring and Grad-CAM."
    ),
    version="2.0.0",
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()

if frontend_origin:
    origins.extend(
        x.strip()
        for x in frontend_origin.split(",")
        if x.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = None
model_load_error: Optional[str] = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Initialize the model before accessing intermediate layers.
    dummy_input = tf.zeros(
        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
        dtype=tf.float32,
    )

    _ = model(dummy_input, training=False)

    print("Model loaded successfully.")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    if (
        len(model.input_shape) != 4
        or model.input_shape[-1] != 1
        or tuple(model.input_shape[1:3]) != IMAGE_SIZE
    ):
        raise ValueError(
            "Expected grayscale model input shape "
            "(224, 224, 1)."
        )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Expected three output scores: "
            "Benign, Malignant, Normal."
        )

    # Confirm the expected Grad-CAM layer exists.
    model.get_layer("conv2d_2")

except Exception as exc:
    model = None
    model_load_error = str(exc)
    print("Model loading failed:", model_load_error)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    try:
        image = (
            Image.open(io.BytesIO(raw_bytes))
            .convert("L")
            .resize(IMAGE_SIZE)
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        return np.expand_dims(array, axis=(0, -1))

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read image. Upload a valid "
                "JPG, PNG, or WebP."
            ),
        ) from exc


# --------------------------------------------------
# MODEL PREDICTION SCORES
# --------------------------------------------------

def read_probabilities(
    image_batch: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(
        model.predict(image_batch, verbose=0)[0],
        dtype=np.float64,
    ).ravel()

    if (
        scores.size != 3
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError(
            "Model returned invalid prediction scores."
        )

    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(scores.sum(), 1.0, atol=1e-3)
    ):
        return scores

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)

    return exp_scores / exp_scores.sum()


# --------------------------------------------------
# DEMONSTRATION RISK CATEGORY
# --------------------------------------------------

def risk_category(score: float) -> str:

    if score < LOW_THRESHOLD:
        return "Low"

    if score < HIGH_THRESHOLD:
        return "Moderate"

    return "High"


# --------------------------------------------------
# IMAGE BASE64 CONVERSION
# --------------------------------------------------

def image_data_uri(image: Image.Image) -> str:

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def make_gradcam(image_batch: np.ndarray):
    """
    Grad-CAM for the Malignant class.

    This is an attention explanation map.
    It is NOT a tumour segmentation mask.
    """

    if model is None:
        raise RuntimeError("Model is not loaded.")

    tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32,
    )

    conv_layer = model.get_layer("conv2d_2")

    # Extract convolutional features.
    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=conv_layer.output,
    )

    # Find the convolution layer position.
    layer_index = model.layers.index(conv_layer)

    # Create a model for the layers after conv2d_2.
    tail_input = tf.keras.Input(
        shape=tuple(conv_layer.output.shape[1:])
    )

    x = tail_input

    for layer in model.layers[layer_index + 1:]:
        x = layer(x)

    tail_model = tf.keras.Model(
        inputs=tail_input,
        outputs=x,
    )

    # Calculate gradients from the malignant score
    # back to the convolutional feature maps.
    with tf.GradientTape() as tape:

        conv_output = feature_model(
            tensor,
            training=False,
        )

        tape.watch(conv_output)

        predictions = tail_model(
            conv_output,
            training=False,
        )

        target = predictions[:, MALIGNANT_INDEX]

    gradients = tape.gradient(
        target,
        conv_output,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated. "
            "Check the model architecture and target layer."
        )

    # Average gradients across spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    features = conv_output[0]

    heat = tf.reduce_sum(
        features * weights,
        axis=-1,
    )

    heat = tf.maximum(heat, 0)

    maximum = tf.reduce_max(heat)

    heat = tf.where(
        maximum > 0,
        heat / (maximum + 1e-8),
        heat,
    )

    # Resize heatmap.
    gray = Image.fromarray(
        np.uint8(
            np.clip(heat.numpy(), 0, 1) * 255
        )
    ).resize(IMAGE_SIZE)

    intensity = np.asarray(
        gray,
        dtype=np.float32,
    ) / 255.0

    # Create colour heatmap.
    rgb = np.zeros(
        (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
        dtype=np.uint8,
    )

    rgb[..., 0] = np.uint8(
        np.clip(
            255 * np.minimum(1, 2 * intensity),
            0,
            255,
        )
    )

    rgb[..., 1] = np.uint8(
        np.clip(
            255 * (
                1 - np.abs(2 * intensity - 1)
            ),
            0,
            255,
        )
    )

    rgb[..., 2] = np.uint8(
        np.clip(
            255 * (1 - 2 * intensity),
            0,
            255,
        )
    )

    heatmap = Image.fromarray(
        rgb,
        mode="RGB",
    )

    # Original grayscale image converted to RGB.
    original = Image.fromarray(
        np.uint8(
            np.clip(
                image_batch[0, :, :, 0],
                0,
                1,
            ) * 255
        ),
        mode="L",
    ).convert("RGB")

    # Blend original image and heatmap.
    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.40,
    )

    return heatmap, overlay


# --------------------------------------------------
# COMPLETE IMAGE ANALYSIS
# --------------------------------------------------

def analyze(raw_bytes: bytes) -> dict:

    batch = preprocess_image(raw_bytes)

    probabilities = read_probabilities(batch)

    predicted_index = int(
        np.argmax(probabilities)
    )

    heatmap, overlay = make_gradcam(batch)

    malignant = float(
        probabilities[MALIGNANT_INDEX]
    )

    return {
        "probabilities": probabilities,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(
            probabilities[predicted_index]
        ),
        "malignant_score": malignant,
        "risk_category": risk_category(malignant),
        "heatmap": heatmap,
        "overlay": overlay,
    }


# --------------------------------------------------
# UPLOAD VALIDATION
# --------------------------------------------------

async def read_upload(file: UploadFile) -> bytes:

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

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 15 MB.",
        )

    return raw


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "AI-Based Lung Cancer Detection",
        "status": (
            "ready"
            if model is not None
            else "model_missing"
        ),
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict",
        "report_endpoint": "/report",
        "documentation": "/docs",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": (
            "ok"
            if model is not None
            else "model_missing"
        ),
        "model_loaded": model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "classes": CLASS_NAMES,
        "gradcam_layer": "conv2d_2",
        "detail": (
            None
            if model is not None
            else model_load_error
        ),
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_load_error}",
        )

    raw = await read_upload(file)

    try:
        result = analyze(raw)

        probs = result["probabilities"]

        return {
            "success": True,

            "predicted_class": result["predicted_class"],

            "confidence": round(
                result["confidence"],
                4,
            ),

            "confidence_percent": round(
                result["confidence"] * 100,
                2,
            ),

            "scores": {
                name: round(float(score), 4)
                for name, score in zip(
                    CLASS_NAMES,
                    probs,
                )
            },

            "malignant_score_percent": round(
                result["malignant_score"] * 100,
                2,
            ),

            "risk_category": result["risk_category"],

            "risk_score_note": (
                "This is the model's Malignant-class score, "
                "not a clinically validated probability of "
                "cancer or personal medical risk."
            ),

            "heatmap_image": image_data_uri(
                result["heatmap"]
            ),

            "overlay_image": image_data_uri(
                result["overlay"]
            ),

            "visualization_note": (
                "Grad-CAM highlights image regions that "
                "influenced the model. It is not a tumour "
                "segmentation mask and does not confirm "
                "tumour location or boundaries."
            ),

            "notice": (
                "Educational research output only. "
                "Not a medical diagnosis; do not use "
                "for health decisions."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Prediction error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


# --------------------------------------------------
# PDF REPORT ENDPOINT
# --------------------------------------------------

@app.post("/report")
async def download_report(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model is unavailable.",
        )

    raw = await read_upload(file)

    try:
        result = analyze(raw)
        probs = result["probabilities"]

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
        )

        styles = getSampleStyleSheet()

        story = [
            Paragraph(
                "LungSight AI - Image Analysis Report",
                styles["Title"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "Educational research report - "
                "not a medical diagnosis",
                styles["Heading2"],
            ),
            Spacer(1, 4 * mm),
        ]

        rows = [
            ["Measure", "Model output"],
            ["Predicted class", result["predicted_class"]],
            [
                "Prediction confidence",
                f'{result["confidence"] * 100:.2f}%',
            ],
            [
                "Malignant-class score",
                f'{result["malignant_score"] * 100:.2f}%',
            ],
            [
                "Demonstration risk category",
                result["risk_category"],
            ],
            ["Benign score", f"{probs[0] * 100:.2f}%"],
            ["Malignant score", f"{probs[1] * 100:.2f}%"],
            ["Normal score", f"{probs[2] * 100:.2f}%"],
        ]

        table = Table(
            rows,
            colWidths=[65 * mm, 95 * mm],
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#243b53"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ])
        )

        story.extend([
            table,
            Spacer(1, 5 * mm),
            Paragraph(
                "Grad-CAM attention overlay",
                styles["Heading2"],
            ),
            Spacer(1, 2 * mm),
        ])

        image_buffer = io.BytesIO()

        result["overlay"].save(
            image_buffer,
            format="PNG",
        )

        image_buffer.seek(0)

        story.append(
            PDFImage(
                image_buffer,
                width=100 * mm,
                height=100 * mm,
            )
        )

        story.extend([
            Spacer(1, 4 * mm),
            Paragraph(
                "Grad-CAM is an explanation map showing "
                "regions that influenced the model. It is "
                "not a tumour segmentation mask and does "
                "not establish tumour boundaries or "
                "affected lung area.",
                styles["BodyText"],
            ),
            Spacer(1, 3 * mm),
            Paragraph(
                "The malignant-class score and "
                "Low/Moderate/High category use "
                "demonstration-only thresholds. They are "
                "not clinically validated cancer-risk "
                "estimates. Consult a qualified medical "
                "professional for interpretation.",
                styles["BodyText"],
            ),
        ])

        doc.build(story)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    'attachment; filename="lungsight_analysis_report.pdf"'
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("Report generation error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {exc}",
        ) from exc


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
