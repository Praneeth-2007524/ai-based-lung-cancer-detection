from pathlib import Path
import json
import tensorflow as tf

ROOT = Path("data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
CLASSES = ["Benign", "Malignant", "Normal"]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_split(split, shuffle):
    path = ROOT / split
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See README.md for required folder structure.")
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )

train_ds = load_split("train", True)
val_ds = load_split("val", False)
test_ds = load_split("test", False)

# Scale pixel intensities to [0, 1].
normalization = tf.keras.layers.Rescaling(1.0 / 255)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),
    normalization,
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "lung_cancer_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print("Class order:", CLASSES)
with open(MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump({"class_names": CLASSES, "test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=2)
print("Saved model to models/lung_cancer_model.keras")
