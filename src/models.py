import torch.nn as nn
from torchvision.models import (
    resnet50,
    efficientnet_b0,
    vit_b_16,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    ViT_B_16_Weights,
)


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vit_base":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")