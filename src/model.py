import torch
import torch.nn as nn
from torchvision import models


# MODEL

def create_model(num_classes=15):

    print("Loading pretrained ResNet50...")

    # Load pretrained ResNet50
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Get number of input features
    num_features = model.fc.in_features

    # Replace the final classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )

    return model


# DEVICE

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


# TEST MODEL

if __name__ == "__main__":

    device = get_device()

    model = create_model(
        num_classes=15
    )

    model = model.to(device)

    print("\nModel created successfully!")

    print(f"Device: {device}")

    print("\nTrainable parameters:")

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Trainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")