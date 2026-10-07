from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

# Original dataset
DATASET_DIR = BASE_DIR / "dataset" / "PlantVillage" / "PlantVillage"

# Output directories
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
TEST_DIR = BASE_DIR / "dataset" / "test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 1. GET ALL IMAGES

def get_images():

    images = []
    labels = []

    classes = sorted(
        [
            folder.name
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    for class_name in classes:

        class_dir = DATASET_DIR / class_name

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                images.append(image_path)
                labels.append(class_name)

    return images, labels, classes


# 2. CREATE DIRECTORIES

def create_directories(classes):

    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        for class_name in classes:

            class_dir = base_dir / class_name
            class_dir.mkdir(
                parents=True,
                exist_ok=True
            )

# 3. COPY IMAGES
def copy_images(images, labels, output_dir):

    for image_path, class_name in zip(images, labels):

        destination = output_dir / class_name / image_path.name

        shutil.copy2(
            image_path,
            destination
        )

# 4. DISPLAY DISTRIBUTION

def display_distribution(
    train_labels,
    val_labels,
    test_labels
):

    print("\nDataset distribution:")
    print("-" * 60)

    print(f"Train      : {len(train_labels)} images")
    print(f"Validation : {len(val_labels)} images")
    print(f"Test       : {len(test_labels)} images")

    print("-" * 60)

    total = (
        len(train_labels)
        + len(val_labels)
        + len(test_labels)
    )

    print(f"Total      : {total} images")


# 5. MAIN PREPROCESSING

def main():

    print("=" * 60)
    print("PLANT DISEASE DATA PREPROCESSING")
    print("=" * 60)

    # Check dataset
    if not DATASET_DIR.exists():

        print("\nERROR: Dataset not found!")
        print(DATASET_DIR)

        return

    print("\nLoading images...")

    images, labels, classes = get_images()

    print(f"Number of classes : {len(classes)}")
    print(f"Total images     : {len(images)}")

    # --------------------------------------------------------
    # First split:
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------------

    train_images, temp_images, train_labels, temp_labels = (
        train_test_split(
            images,
            labels,
            test_size=0.30,
            random_state=42,
            stratify=labels
        )
    )

    # --------------------------------------------------------
    # Second split:
    # 15% Validation
    # 15% Test
    # --------------------------------------------------------

    val_images, test_images, val_labels, test_labels = (
        train_test_split(
            temp_images,
            temp_labels,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels
        )
    )

    # Create directories

    print("\nCreating directories...")

    create_directories(classes)

    # Copy files
    print("\nCopying training images...")
    copy_images(
        train_images,
        train_labels,
        TRAIN_DIR
    )

    print("Copying validation images...")
    copy_images(
        val_images,
        val_labels,
        VAL_DIR
    )

    print("Copying test images...")
    copy_images(
        test_images,
        test_labels,
        TEST_DIR
    )

    # Display result

    display_distribution(
        train_labels,
        val_labels,
        test_labels
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
