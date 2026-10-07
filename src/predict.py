import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Configuration

MODEL_PATH = "models/plant_disease_classifier.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


# Image Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load ResNet50

print("Loading ResNet50...")

resnet = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)
resnet.eval()


# Load Classifier

print("Loading trained classifier...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

classifier = DiseaseClassifier(
    checkpoint["input_size"],
    checkpoint["num_classes"]
)

classifier.load_state_dict(
    checkpoint["model_state_dict"]
)

classifier = classifier.to(DEVICE)
classifier.eval()


# Prediction Function

def predict_image(image_path):

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # Feature extraction
    with torch.no_grad():
        features = resnet(image_tensor)

        # Classification
        outputs = classifier(features)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()
    confidence_value = confidence.item() * 100

    predicted_label = class_names[predicted_index]

    print("\n==============================")
    print("PREDICTION RESULT")
    print("==============================")

    print(f"Image     : {image_path}")
    print(f"Disease   : {predicted_label}")
    print(f"Confidence: {confidence_value:.2f}%")

    # Top 3 predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    print("\nTop 3 predictions:")

    for i in range(3):

        index = top_indices[0][i].item()
        probability = top_probabilities[0][i].item() * 100

        print(
            f"{i + 1}. "
            f"{class_names[index]} "
            f"({probability:.2f}%)"
        )


# Main

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src\\predict.py "
            "path_to_image.jpg"
        )

        print(
            "\nExample:"
        )

        print(
            "python src\\predict.py "
            "dataset\\test\\Tomato_healthy\\image.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)
