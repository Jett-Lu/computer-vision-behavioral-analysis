"""Small classifier heads and frozen Torchvision feature extractors."""

from __future__ import annotations

import torch
from torch import nn


MODEL_NAMES = ("mlp", "cnn", "advanced")


class ActivityClassifier(nn.Module):
    """A linear head or one-hidden-layer MLP for four activity classes."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_activity_classifier(model_name: str, input_dim: int) -> ActivityClassifier:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported activity model: {model_name}")
    hidden_dim = 128 if model_name == "mlp" else 0
    return ActivityClassifier(input_dim=input_dim, hidden_dim=hidden_dim)


def build_mobilenet_extractor(pretrained: bool = True) -> nn.Module:
    """Return frozen MobileNetV2 features with the ImageNet head removed."""
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_s3d_extractor(pretrained: bool = True) -> nn.Module:
    """Return frozen S3D features with the Kinetics-400 head removed."""
    from torchvision.models.video import S3D_Weights, s3d

    weights = S3D_Weights.KINETICS400_V1 if pretrained else None
    model = s3d(weights=weights)
    model.classifier = nn.Sequential()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
