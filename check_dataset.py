
from pathlib import Path

for split in ["train", "val", "test"]:
    for category in ["Benign", "Malignant", "Normal"]:
        folder = Path("data") / split / category

        if folder.exists():
            count = sum(1 for p in folder.rglob("*") if p.is_file())
            print(f"{split}/{category}: {count} files")
        else:
            print(f"{split}/{category}: FOLDER MISSING")
