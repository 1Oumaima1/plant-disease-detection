import os
from pathlib import Path
from collections import Counter

from PIL import Image
import matplotlib.pyplot as plt


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_DIR = BASE_DIR / "dataset" / "PlantVillage" / "PlantVillage"

# Results directories
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"}


# 1. CHECK DATASET

def check_dataset():
    """Check if the dataset directory exists."""

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at:")
        print(DATASET_DIR)
        return False

    print("Dataset found successfully!")
    print(f"Path: {DATASET_DIR}")

    return True


# 2. GET CLASSES

def get_classes():
    """Return the list of class folders."""

    classes = sorted(
        [
            folder.name
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    return classes


# 3. COUNT IMAGES PER CLASS

def count_images_per_class(classes):
    """Count images inside each class."""

    class_counts = {}

    for class_name in classes:

        class_path = DATASET_DIR / class_name

        images = [
            file
            for file in class_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        class_counts[class_name] = len(images)

    return class_counts


# 4. DISPLAY DATASET INFORMATION
def display_dataset_info(classes, class_counts):

    total_images = sum(class_counts.values())

    print("\n" + "=" * 60)
    print("PLANT DISEASE DATASET EXPLORATION")
    print("=" * 60)

    print(f"\nNumber of classes : {len(classes)}")
    print(f"Total images     : {total_images}")

    print("\nClasses:")
    print("-" * 60)

    for class_name in classes:
        print(f"{class_name:<55} {class_counts[class_name]}")


# 5. CLASS DISTRIBUTION

def plot_class_distribution(class_counts):

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(14, 8))

    plt.bar(range(len(classes)), counts)

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90
    )

    plt.xlabel("Plant Disease Class")
    plt.ylabel("Number of Images")
    plt.title("PlantVillage - Class Distribution")

    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"\nSaved: {output_path}")


# 6. SAMPLE IMAGES

def display_sample_images(classes, samples_per_class=1):

    print("\nGenerating sample images...")

    selected_images = []

    for class_name in classes:

        class_path = DATASET_DIR / class_name

        images = [
            file
            for file in class_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        if images:
            selected_images.append(
                (class_name, images[0])
            )

    n = len(selected_images)

    fig, axes = plt.subplots(
        5,
        4,
        figsize=(14, 16)
    )

    axes = axes.flatten()

    for i, (class_name, image_path) in enumerate(selected_images):

        try:

            image = Image.open(image_path)

            axes[i].imshow(image)
            axes[i].set_title(
                class_name,
                fontsize=8
            )
            axes[i].axis("off")

        except Exception as e:

            print(f"Could not open: {image_path}")
            print(f"Error: {e}")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "Sample Images from PlantVillage Dataset",
        fontsize=16
    )

    plt.tight_layout()

    output_path = FIGURES_DIR / "sample_images.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"Saved: {output_path}")


# 7. CHECK IMAGE DIMENSIONS

def check_image_dimensions(classes):

    dimensions = Counter()

    print("\nChecking image dimensions...")

    for class_name in classes:

        class_path = DATASET_DIR / class_name

        for image_path in class_path.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp"}
            ):

                try:

                    with Image.open(image_path) as image:

                        dimensions[image.size] += 1

                except Exception:
                    pass

    print("\nMost common image dimensions:")

    for dimension, count in dimensions.most_common(10):

        print(
            f"{dimension[0]} x {dimension[1]} : {count} images"
        )


# 8. CHECK CORRUPTED IMAGES

def check_corrupted_images(classes):

    corrupted_images = []

    print("\nChecking for corrupted images...")

    for class_name in classes:

        class_path = DATASET_DIR / class_name

        for image_path in class_path.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp"}
            ):

                try:

                    with Image.open(image_path) as image:

                        image.verify()

                except Exception:

                    corrupted_images.append(
                        str(image_path)
                    )

    print(
        f"Corrupted images: {len(corrupted_images)}"
    )

    if corrupted_images:

        print("\nCorrupted files:")

        for image in corrupted_images:
            print(image)

    return corrupted_images


# MAIN

def main():

    # Check dataset
    if not check_dataset():
        return

    # Get classes
    classes = get_classes()

    # Count images
    class_counts = count_images_per_class(classes)

    # Display information
    display_dataset_info(
        classes,
        class_counts
    )

    # Class distribution
    plot_class_distribution(
        class_counts
    )

    # Sample images
    display_sample_images(
        classes
    )

    # Image dimensions
    check_image_dimensions(
        classes
    )

    # Corrupted images
    check_corrupted_images(
        classes
    )

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
