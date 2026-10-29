
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

ROOT = Path("data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
CLASSES = ["Benign", "Malignant", "Normal"]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_split(split, shuffle):
    path = ROOT / split
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset folder: {path}")

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds

train_raw = load_split("train", True)
val_raw = load_split("val", False)
test_raw = load_split("test", False)

# Calculate weights to help with class imbalance.
train_labels = np.concatenate([
    labels.numpy() for _, labels in train_raw
])
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=train_labels,
)
class_weights = {
    i: float(weight) for i, weight in enumerate(weights)
}
print("Class weights:", class_weights)

# Improve input pipeline performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw.prefetch(AUTOTUNE)
val_ds = val_raw.prefetch(AUTOTUNE)
test_ds = test_raw.prefetch(AUTOTUNE)

# Mild augmentation. Do not flip or distort scans excessively.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.04, 0.04),
], name="augmentation")

inputs = tf.keras.layers.Input(shape=(224, 224, 1))
x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
x = augmentation(x)

for filters in [32, 64, 128]:
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    len(CLASSES), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_path = MODEL_DIR / "lung_cancer_model_improved.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(best_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Load the best validation model.
model = tf.keras.models.load_model(best_path)

# Evaluate the test set only after training is complete.
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("Class order:", CLASSES)

summary = {
    "class_names": CLASSES,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "epochs_completed": len(history.history["loss"]),
}

with open(
    MODEL_DIR / "improved_training_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print(f"Saved improved model to {best_path}")
print("Original model was not overwritten.")
