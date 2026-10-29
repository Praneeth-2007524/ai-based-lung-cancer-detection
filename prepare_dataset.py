
from pathlib import Path
import random
import shutil

# Change this path only if your dataset is stored elsewhere.
SOURCE = Path(
    r"C:\Users\kanur\Downloads\LungDataset\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"
)

PROJECT = Path(__file__).resolve().parent
DESTINATION = PROJECT / "data"

# Original dataset folder names and model class names
CLASSES = {
    "Bengin cases": "Benign",
    "Malignant cases": "Malignant",
    "Normal cases": "Normal",
}

# Split ratios: 70% training, 15% validation, 15% testing
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    if not SOURCE.is_dir():
        print("ERROR: Dataset folder was not found:")
        print(SOURCE)
        print("Please check the SOURCE path in this script.")
        return

    # Check all source folders before copying anything.
    for original_name in CLASSES:
        if not (SOURCE / original_name).is_dir():
            print(f"ERROR: Missing folder: {SOURCE / original_name}")
            print("Check the spelling of the original dataset folders.")
            return

    # Prevent accidental overwriting of an existing dataset.
    if any(DESTINATION.rglob("*")):
        print("ERROR: The data folder is not empty.")
        print("To avoid overwriting files, this script has stopped.")
        print("Use an empty data folder or choose another destination.")
        return

    random.seed(RANDOM_SEED)
    total = 0

    for original_name, class_name in CLASSES.items():
        source_folder = SOURCE / original_name
        images = sorted(
            p for p in source_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            print(f"ERROR: Found only {len(images)} images in {original_name}.")
            print("Check the dataset folder before continuing.")
            return

        random.shuffle(images)

        n = len(images)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            target_folder = DESTINATION / split_name / class_name
            target_folder.mkdir(parents=True, exist_ok=True)

            for index, image_path in enumerate(split_images):
                # Unique filenames avoid collisions.
                new_name = f"{class_name}_{index:04d}{image_path.suffix.lower()}"
                shutil.copy2(image_path, target_folder / new_name)

        print(f"{class_name}: {n} images")
        for split_name, split_images in splits.items():
            print(f"  {split_name}: {len(split_images)}")

        total += n

    print(f"\nFinished copying {total} images.")
    print(f"Dataset location: {DESTINATION}")
    print("IMPORTANT: Review patient-level separation before research evaluation.")

if __name__ == "__main__":
    main()
