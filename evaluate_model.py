
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

CLASSES = ["Benign", "Malignant", "Normal"]
IMG_SIZE = (224, 224)

model_path = Path("models/lung_cancer_model_improved.keras")
test_path = Path("data/test")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not test_path.exists():
    raise FileNotFoundError(f"Test folder not found: {test_path}")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=16,
    shuffle=False,
)

model = tf.keras.models.load_model(model_path)

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0,
    digits=4,
))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("CONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES,
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Improved Lung Cancer Model - Test Set")
plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.show()

print("\nSaved confusion matrix to results/confusion_matrix.png")
