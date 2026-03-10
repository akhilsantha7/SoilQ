"""
Generate augmented copies of dataset images to increase effective training set size.
Use when you have few images per class (e.g. ~35). Run once, then train on dataset_augmented/.

Usage:
  python generate_augmented_dataset.py

Creates dataset_augmented/ with originals + augmented copies. Then in train.py
use: full_dataset = datasets.ImageFolder("dataset_augmented", ...)
"""
import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

SRC_DIR = "dataset"
OUT_DIR = "dataset_augmented"
COPIES_PER_IMAGE = 4   # 4 augmented copies per image (35 -> 35 + 140 = 175 per class)

# Augmentations that stay on PIL (no normalization – that's for training only)
augment = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.1),
])

def main():
    src = Path(SRC_DIR)
    out = Path(OUT_DIR)
    if not src.is_dir():
        print(f"Error: {SRC_DIR} not found")
        return

    out.mkdir(parents=True, exist_ok=True)
    total_orig = 0
    total_aug = 0

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        out_class = out / class_name
        out_class.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue

            base = img_path.stem
            # Save original once into augmented set (so all data is in one place)
            orig_out = out_class / f"{base}_orig{img_path.suffix}"
            if not orig_out.exists():
                img.copy().save(orig_out)
                total_orig += 1

            # Save augmented copies
            for i in range(COPIES_PER_IMAGE):
                aug_img = augment(img)
                out_path = out_class / f"{base}_aug{i}{img_path.suffix}"
                if aug_img.mode != "RGB":
                    aug_img = aug_img.convert("RGB")
                aug_img.save(out_path, quality=92)
                total_aug += 1

        print(f"  {class_name}: {len(images)} originals -> {len(images) * (1 + COPIES_PER_IMAGE)} files")

    print(f"\nDone. Total: {total_orig} originals + {total_aug} augmented = {total_orig + total_aug} in {OUT_DIR}/")
    print(f"Next: in train.py use  full_dataset = datasets.ImageFolder('{OUT_DIR}', ...)  to train on augmented set.")

if __name__ == "__main__":
    main()
