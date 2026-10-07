from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"

FEATURES_DIR = BASE_DIR / "features"

TRAIN_FEATURES = FEATURES_DIR / "train_features.pt"
VAL_FEATURES = FEATURES_DIR / "val_features.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 8

# DEVICE

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# CREATE MODEL

def create_feature_extractor(device):

    print("\nLoading pretrained ResNet50...")

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    )

    # Remove the final classification layer
    # ResNet50 output before FC = 2048 features
    model.fc = torch.nn.Identity()

    model = model.to(device)

    # Evaluation mode
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print("Feature extractor ready.")

    return model

# EXTRACT FEATURES

def extract_features(
    model,
    dataset_dir,
    output_file,
    device,
    dataset_name
):

    print("\n" + "=" * 60)
    print(f"EXTRACTING FEATURES: {dataset_name}")
    print("=" * 60)

    # Load dataset
    dataset = datasets.ImageFolder(
        dataset_dir,
        transform=transform
    )

    print(f"Images: {len(dataset)}")
    print(f"Classes: {len(dataset.classes)}")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_labels = []

    total_batches = len(loader)

    print("\nStarting feature extraction...")

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            # Move images to CPU/GPU
            images = images.to(device)

            # Extract features
            features = model(images)

            # Move features back to CPU
            features = features.cpu()

            # Store batch
            all_features.append(features)
            all_labels.append(labels)

            # Progress
            if (
                (batch_index + 1) % 50 == 0
                or batch_index == total_batches - 1
            ):

                progress = (
                    (batch_index + 1)
                    / total_batches
                    * 100
                )

                print(
                    f"Progress: "
                    f"{batch_index + 1}/{total_batches} "
                    f"({progress:.1f}%)"
                )

    # Combine all batches
    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # Save features and labels
    torch.save(
        {
            "features": features,
            "labels": labels,
            "class_names": dataset.classes
        },
        output_file
    )

    print("\nExtraction completed!")

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    print(
        f"Saved to:\n{output_file}"
    )

    # Free memory
    del all_features
    del all_labels
    del features
    del labels


# MAIN

def main():

    print("=" * 60)
    print("PLANT DISEASE FEATURE EXTRACTION")
    print("=" * 60)

    # Create output directory
    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Device
    device = get_device()

    # Create ResNet50 feature extractor
    model = create_feature_extractor(
        device
    )

    # TRAIN FEATURES

    extract_features(
        model=model,
        dataset_dir=TRAIN_DIR,
        output_file=TRAIN_FEATURES,
        device=device,
        dataset_name="TRAIN"
    )

    # VALIDATION FEATURES

    extract_features(
        model=model,
        dataset_dir=VAL_DIR,
        output_file=VAL_FEATURES,
        device=device,
        dataset_name="VALIDATION"
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    print(TRAIN_FEATURES)
    print(VAL_FEATURES)


# RUN

if __name__ == "__main__":
    main()
