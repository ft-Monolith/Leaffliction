import os

import torch
import torch.nn as nn

from srcs.utils import error_exit


def make_CNN(num_classes):
    model = LeafCNN(num_classes)
    return model


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, num_classes, device="cpu"):
    if not os.path.isfile(model_path):
        error_exit(f"model '{model_path}' missing")
    model = make_CNN(num_classes).to(device)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        error_exit(f"'{model_path}' is not a valid model "
                   f"for {num_classes} classes")
    model.eval()
    return model


class LeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define the CNN architecture 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3->64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # ->128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # ->256
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Pooling layer to reduce spatial dimensions
        self.relu = nn.ReLU() # Non-linear activation function (negative values to zero)
        self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pooling layer to reduce each feature map to a single value
        self.fc = nn.Linear(256, num_classes) # Fully connected layer to map the 256 features to the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
