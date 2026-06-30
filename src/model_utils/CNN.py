import torch.nn as nn


def make_CNN(num_classes):
    model = LeafCNN(num_classes)
    return model


class LeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Convolutions : détectent des motifs (bords, textures, formes).
        # Le 1er nb = canaux d'entrée, le 2e = nb de filtres (sortie).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3->64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # ->128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # ->256
        # BatchNorm : normalise les sorties de conv -> entraînement stable.
        #self.bn1 = nn.BatchNorm2d(64)
        #self.bn2 = nn.BatchNorm2d(128)
        #self.bn3 = nn.BatchNorm2d(256)
        # MaxPool : divise la taille de l'image par 2 (résume l'info).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU : non-linéarité (met les valeurs négatives à 0).
        self.relu = nn.ReLU()
        # GAP : moyenne chaque carte -> 1 valeur par canal (256 valeurs).
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Linear : couche finale, transforme les 256 valeurs en N classes.
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
