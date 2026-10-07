from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

TRAIN_FEATURES = FEATURES_DIR / "train_features.pt"
VAL_FEATURES = FEATURES_DIR / "val_features.pt"

MODEL_PATH = MODELS_DIR / "plant_disease_classifier.pth"

BATCH_SIZE = 64
NUM_CLASSES = 15
NUM_EPOCHS = 20
LEARNING_RATE = 0.001


# CLASSIFIER

class PlantDiseaseClassifier(nn.Module):

    def __init__(self, input_size=2048, num_classes=15):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.classifier(x)


# LOAD FEATURES

def load_features():

    print("Loading extracted features...")

    train_data = torch.load(
        TRAIN_FEATURES,
        weights_only=True
    )

    val_data = torch.load(
        VAL_FEATURES,
        weights_only=True
    )

    train_features = train_data["features"]
    train_labels = train_data["labels"]

    val_features = val_data["features"]
    val_labels = val_data["labels"]

    class_names = train_data["class_names"]

    print(f"Training features   : {train_features.shape}")
    print(f"Training labels     : {train_labels.shape}")

    print(f"Validation features : {val_features.shape}")
    print(f"Validation labels   : {val_labels.shape}")

    print(f"Number of classes   : {len(class_names)}")

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_names
    )


# CREATE DATALOADERS

def create_dataloaders(
    train_features,
    train_labels,
    val_features,
    val_labels
):

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    val_dataset = TensorDataset(
        val_features,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


# TRAIN

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * features.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    loss = total_loss / total

    accuracy = correct / total

    return loss, accuracy

# VALIDATION

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * features.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    loss = total_loss / total

    accuracy = correct / total

    return loss, accuracy


# MAIN

def main():

    print("=" * 60)
    print("PLANT DISEASE CLASSIFIER TRAINING")
    print("=" * 60)

    # Check feature files
    if not TRAIN_FEATURES.exists():

        print("\nERROR:")
        print(f"Training features not found:")
        print(TRAIN_FEATURES)

        return

    if not VAL_FEATURES.exists():

        print("\nERROR:")
        print(f"Validation features not found:")
        print(VAL_FEATURES)

        return

    # Create models directory
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load features

    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_names
    ) = load_features()

    # DataLoaders

    train_loader, val_loader = create_dataloaders(
        train_features,
        train_labels,
        val_features,
        val_labels
    )

    # Model

    model = PlantDiseaseClassifier(
        input_size=2048,
        num_classes=NUM_CLASSES
    )

    # Loss

    criterion = nn.CrossEntropyLoss()

    # Optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training

    best_val_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": 2048,
                    "num_classes": NUM_CLASSES,
                    "val_accuracy": best_val_accuracy
                },
                MODEL_PATH
            )

            print(
                f"  → Best model saved "
                f"(Val Acc: {val_accuracy:.4f})"
            )

    # Final result

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Model saved at:\n{MODEL_PATH}"
    )

# RUN

if __name__ == "__main__":
    main()
