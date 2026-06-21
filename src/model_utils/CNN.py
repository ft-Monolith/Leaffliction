"""Modèle CNN.

TEMPORAIRE : on utilise un resnet18 tout fait de torchvision pour
pouvoir tester la boucle d'entraînement. À remplacer par notre propre
CNN maison à la fin.
"""

import torch.nn as nn
from torchvision import models


def make_CNN(num_classes):
    """Retourne un modèle prêt à entraîner pour `num_classes` classes."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
