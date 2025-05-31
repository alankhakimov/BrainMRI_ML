import os
from PIL import Image
from pathlib import Path
import shutil
import random

def resize_images_in_folder(input_folder, output_folder, size=(128, 128), image_format="JPEG"):
    """
    Resizes all images in a folder to the specified size and saves them to a new folder.

    :param input_folder: Path to the folder containing the original images.
    :param output_folder: Path where resized images will be saved.
    :param size: Tuple (width, height), e.g., (128, 128)
    :param image_format: Image format to save as, e.g., "JPEG" or "PNG"
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(root, filename)
                # Preserve subdirectory structure
                rel_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, rel_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    img = Image.open(input_path)
                    img = img.convert("RGB")  # Ensure 3 channels
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(output_path, format=image_format)
                    print(f"Resized: {rel_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")
def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # For each class subdirectory
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, split_images in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(split_class_dir, img)
                shutil.copy2(src_path, dst_path)

        print(f"{class_name}: {len(images)} images split into train/val/test.")

def main():
    input_path = Path("C:\Documents\MRI Machine Learning Project\\3 category brain MRI\Brain_Cancer raw MRI data\Resized_Images")
    output_path = Path("C:\Documents\MRI Machine Learning Project\\3 category brain MRI\Brain_Cancer raw MRI data\Split_Dataset")
    split_dataset(input_path, output_path)

if __name__ == "__main__":
    main()