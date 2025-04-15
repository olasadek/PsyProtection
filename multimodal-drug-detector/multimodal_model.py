#%%
# ====================== 0. Import Libraries ======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#%%
# ====================== 1. Define Networks ======================

class ViTImageEncoder(nn.Module):
    """Vision Transformer (ViT) for extracting features from grayscale MRI images."""
    def __init__(self):
        super(ViTImageEncoder, self).__init__()
        vit = models.vit_b_16(pretrained=True)
        vit.heads = nn.Identity()  # Remove classifier head
        self.vit = vit

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels
        return self.vit(x)  # Output: (B, 768)


class EHRFeatureEncoder(nn.Module):
    """Simple feedforward network for extracting features from EHR (tabular) data."""
    def __init__(self, input_dim):
        super(EHRFeatureEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return x  # Output: (B, 50)


class MultiModalDiagnosticNet(nn.Module):
    """Model that fuses image and EHR features for binary classification."""
    def __init__(self, ehr_input_dim):
        super(MultiModalDiagnosticNet, self).__init__()
        self.image_encoder = ViTImageEncoder()
        self.ehr_encoder = EHRFeatureEncoder(ehr_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 50, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, features):
        img_features = self.image_encoder(images)
        ehr_features = self.ehr_encoder(features)
        combined = torch.cat([img_features, ehr_features], dim=1)
        return self.classifier(combined)
