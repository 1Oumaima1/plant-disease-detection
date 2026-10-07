import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration


TEST_DIR = "dataset/test"
MODEL_PATH = "models/plant_disease_classifier.pth"

RESULTS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# Load Test Dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print(f"Number of test images: {len(test_dataset)}")
print(f"Number of classes: {len(class_names)}")


# Load ResNet50

print("\nLoading pretrained ResNet50...")

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove final classification layer
resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)
resnet.eval()



# Extract Test Features

all_features = []
all_labels = []

print("\nExtracting test features...")

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(test_loader):

        images = images.to(DEVICE)

        features = resnet(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Processed {batch_idx + 1}/{len(test_loader)} batches"
            )


test_features = torch.cat(all_features)
test_labels = torch.cat(all_labels)

print("\nFeature extraction completed.")
print(f"Feature shape: {test_features.shape}")
print(f"Labels shape: {test_labels.shape}")


# Save Test Features

torch.save(
    {
        "features": test_features,
        "labels": test_labels,
        "class_names": class_names
    },
    "features/test_features.pt"
)

print("Test features saved.")


# Classifier Definition

class DiseaseClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
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


# Load Trained Classifier

print("\nLoading trained classifier...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classifier = DiseaseClassifier(
    checkpoint["input_size"],
    checkpoint["num_classes"]
)

classifier.load_state_dict(
    checkpoint["model_state_dict"]
)

classifier = classifier.to(DEVICE)
classifier.eval()

print(
    f"Best validation accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# Prediction

test_features = test_features.to(DEVICE)
test_labels = test_labels.to(DEVICE)

all_predictions = []

with torch.no_grad():

    outputs = classifier(test_features)

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    all_predictions = predictions.cpu().numpy()

true_labels = test_labels.cpu().numpy()


# Metrics
accuracy = accuracy_score(
    true_labels,
    all_predictions
)

precision = precision_score(
    true_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-score : {f1 * 100:.2f}%")


# Classification Report

report = classification_report(
    true_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# Save Metrics

metrics = {
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "classification_report": classification_report(
        true_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
}

with open(
    os.path.join(RESULTS_DIR, "test_metrics.json"),
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    f"\nMetrics saved to: "
    f"{RESULTS_DIR}/test_metrics.json"
)


# Confusion Matrix

cm = confusion_matrix(
    true_labels,
    all_predictions
)

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - Plant Disease Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()

cm_path = os.path.join(
    FIGURES_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print(
    f"Confusion matrix saved to: {cm_path}"
)

print("\nEvaluation completed successfully!")

