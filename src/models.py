import torch.nn as nn
from torchvision.models import (
    resnet50,
    efficientnet_b0,
    vit_b_16,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    ViT_B_16_Weights,
)

from wsdan import get_wsdan_config, WSDANModel

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def get_model_weights(model_name: str, pretrained: bool = True):
    model_name = model_name.lower()

    if not pretrained:
        return None

    if model_name == "resnet50":
        return ResNet50_Weights.DEFAULT

    if model_name == "efficientnet_b0":
        return EfficientNet_B0_Weights.DEFAULT

    if model_name == "vit_base":
        return ViT_B_16_Weights.DEFAULT

    return None


def freeze_backbone(model_name: str, model) -> None:
    model_name = model_name.lower()

    if isinstance(model, WSDANModel):
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.attention_head.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if model_name == "resnet50":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if model_name == "efficientnet_b0":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if model_name == "vit_base":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.heads.parameters():
            parameter.requires_grad = True
        return

    if HAS_TIMM and hasattr(model, "get_classifier"):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported model for freezing: {model_name}")


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.0,
    config: dict | None = None,
):
    model_name = model_name.lower()
    wsdan_config = get_wsdan_config(config)

    if wsdan_config.enabled:
        return WSDANModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            num_attention_maps=wsdan_config.num_attention_maps,
            dropout_rate=dropout_rate,
            feature_source=wsdan_config.feature_source,
        )

    weights = get_model_weights(model_name, pretrained)

    if model_name == "resnet50":
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if dropout_rate > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vit_base":
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        if dropout_rate > 0:
            model.heads.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if not HAS_TIMM:
        raise ValueError(f"Unsupported model: {model_name}. Install timm for additional models: pip install timm")

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)
    return model
