import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    resnet50,
    vit_b_16,
)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


@dataclass
class WSDANConfig:
    enabled: bool = False
    start_epoch: int = 0
    num_attention_maps: int = 16
    crop_threshold: float = 0.6
    drop_threshold: float = 0.4
    crop_weight: float = 1.0
    drop_weight: float = 1.0
    base_weight: float = 1.0
    bbox_padding_ratio: float = 0.1
    feature_source: str = "auto"
    inference_crop_weight: float = 1.0


def get_wsdan_config(config: Optional[dict]) -> WSDANConfig:
    wsdan = dict(config.get("wsdan") or {}) if config else {}
    return WSDANConfig(
        enabled=bool(wsdan.get("enabled", False)),
        start_epoch=int(wsdan.get("start_epoch", 0)),
        num_attention_maps=int(wsdan.get("num_attention_maps", 16)),
        crop_threshold=float(wsdan.get("crop_threshold", 0.6)),
        drop_threshold=float(wsdan.get("drop_threshold", 0.4)),
        crop_weight=float(wsdan.get("crop_weight", 1.0)),
        drop_weight=float(wsdan.get("drop_weight", 1.0)),
        base_weight=float(wsdan.get("base_weight", 1.0)),
        bbox_padding_ratio=float(wsdan.get("bbox_padding_ratio", 0.1)),
        feature_source=str(wsdan.get("feature_source", "auto")),
        inference_crop_weight=float(wsdan.get("inference_crop_weight", 1.0)),
    )


def is_wsdan_enabled(config: Optional[dict]) -> bool:
    return get_wsdan_config(config).enabled


def get_logits_from_output(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def select_peak_attention_map(attention_maps: torch.Tensor) -> torch.Tensor:
    attention_scores = attention_maps.flatten(2).amax(dim=2)
    indices = attention_scores.argmax(dim=1)
    selected = attention_maps[torch.arange(attention_maps.size(0), device=attention_maps.device), indices]
    return selected.unsqueeze(1)


class TimmFeatureBackbone(nn.Module):
    CONV_LIKE_HINTS = (
        "convnext",
        "resnet",
        "efficientnet",
        "densenet",
        "mobilenet",
        "regnet",
        "rexnet",
        "coatnet",
        "inception",
        "vgg",
        "dla",
        "csp",
        "darknet",
    )

    TOKEN_HINTS = (
        "eva",
        "vit",
        "deit",
        "beit",
        "mae",
        "mvit",
        "cait",
    )

    def __init__(self, model_name: str, pretrained: bool, feature_source: str = "auto"):
        super().__init__()
        if not HAS_TIMM:
            raise ValueError("WS-DAN with timm backbones requires timm to be installed.")

        self.model_name = model_name.lower()
        self.feature_source = feature_source
        self.backbone = self._build_backbone(model_name, pretrained, feature_source)
        self.num_features = self._infer_num_features()

    def _build_backbone(self, model_name: str, pretrained: bool, feature_source: str):
        if feature_source == "features_only" or (
            feature_source == "auto" and any(hint in self.model_name for hint in self.CONV_LIKE_HINTS)
        ):
            try:
                return timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(-1,),
                )
            except Exception:
                pass

        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

    def _infer_num_features(self) -> int:
        if hasattr(self.backbone, "feature_info"):
            channels = self.backbone.feature_info.channels()
            if channels:
                return int(channels[-1])

        num_features = getattr(self.backbone, "num_features", None)
        if num_features is not None:
            return int(num_features)

        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)

        raise ValueError(f"Unable to infer feature dimension for WS-DAN backbone: {self.model_name}")

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0))
        if num_prefix_tokens > 0 and tokens.size(1) > num_prefix_tokens:
            tokens = tokens[:, num_prefix_tokens:, :]
        elif hasattr(self.backbone, "cls_token") and tokens.size(1) > 1:
            tokens = tokens[:, 1:, :]

        side = int(math.sqrt(tokens.size(1)))
        if side * side != tokens.size(1):
            raise ValueError(
                f"Cannot reshape token sequence of length {tokens.size(1)} into a square feature map for WS-DAN."
        )
        return tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), side, side)

    def _extract_from_dict(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        preferred_keys = (
            "x_norm_patchtokens",
            "patch_tokens",
            "tokens",
            "feature_map",
            "features",
            "last_hidden_state",
        )
        for key in preferred_keys:
            value = features.get(key)
            if isinstance(value, torch.Tensor):
                return value

        for value in features.values():
            if isinstance(value, torch.Tensor):
                return value

        raise ValueError(f"WS-DAN backbone dict output did not contain a tensor feature map: {self.model_name}")

    def _extract_tensor_feature_map(self, features) -> torch.Tensor:
        if isinstance(features, dict):
            features = self._extract_from_dict(features)

        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError(f"WS-DAN backbone returned an empty feature list: {self.model_name}")
            for candidate in reversed(features):
                if isinstance(candidate, torch.Tensor):
                    features = candidate
                    break
            else:
                raise ValueError(f"WS-DAN backbone list output did not contain a tensor feature map: {self.model_name}")

        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Unsupported feature output type for WS-DAN backbone {self.model_name}: {type(features)!r}")

        if features.dim() == 4:
            return features

        if features.dim() == 3:
            return self._reshape_tokens(features)

        raise ValueError(
            f"Unsupported feature shape for WS-DAN backbone {self.model_name}: {tuple(features.shape)}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(images)
        else:
            features = self.backbone(images)

        return self._extract_tensor_feature_map(features)


class ConvNeXtFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        if not HAS_TIMM:
            raise ValueError("ConvNeXt WS-DAN support requires timm to be installed.")

        self.model_name = model_name.lower()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        channels = self.backbone.feature_info.channels()
        if not channels:
            raise ValueError(f"Unable to infer ConvNeXt feature dimension for WS-DAN: {self.model_name}")
        self.num_features = int(channels[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError(f"Unexpected ConvNeXt feature output for WS-DAN: {self.model_name}")
        feature_map = features[-1]
        if not isinstance(feature_map, torch.Tensor) or feature_map.dim() != 4:
            raise ValueError(
                f"ConvNeXt WS-DAN expected a 4D feature map, got {type(feature_map)!r} "
                f"with shape {tuple(feature_map.shape) if isinstance(feature_map, torch.Tensor) else 'n/a'}"
            )
        return feature_map


class EVA02FeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        if not HAS_TIMM:
            raise ValueError("EVA02 WS-DAN support requires timm to be installed.")

        self.model_name = model_name.lower()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.num_features = self._infer_num_features()

    def _infer_num_features(self) -> int:
        for attr_name in ("num_features", "embed_dim"):
            attr_value = getattr(self.backbone, attr_name, None)
            if attr_value is not None:
                return int(attr_value)
        raise ValueError(f"Unable to infer EVA02 feature dimension for WS-DAN: {self.model_name}")

    def _reshape_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        side = int(math.sqrt(patch_tokens.size(1)))
        if side * side != patch_tokens.size(1):
            raise ValueError(
                f"EVA02 WS-DAN expected square patch tokens, got sequence length {patch_tokens.size(1)}"
            )
        return patch_tokens.transpose(1, 2).reshape(patch_tokens.size(0), patch_tokens.size(2), side, side)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "forward_features"):
            raise ValueError(f"EVA02 backbone does not expose forward_features(): {self.model_name}")

        features = self.backbone.forward_features(images)

        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "last_hidden_state"):
                value = features.get(key)
                if isinstance(value, torch.Tensor):
                    if value.dim() == 4:
                        return value
                    if value.dim() == 3:
                        return self._reshape_patch_tokens(value)
            raise ValueError(f"EVA02 WS-DAN could not find patch tokens in dict output: {self.model_name}")

        if isinstance(features, torch.Tensor):
            if features.dim() == 4:
                return features
            if features.dim() == 3:
                num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
                if features.size(1) > num_prefix_tokens:
                    features = features[:, num_prefix_tokens:, :]
                return self._reshape_patch_tokens(features)

        raise ValueError(f"Unsupported EVA02 feature output for WS-DAN: {self.model_name}")


class ResNetFeatureBackbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_features = model.fc.in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class EfficientNetFeatureBackbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.num_features = model.classifier[1].in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class ViTFeatureBackbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        self.num_features = self.backbone.hidden_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone._process_input(images)
        batch_size = tokens.shape[0]
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.backbone.encoder(tokens)
        patch_tokens = tokens[:, 1:, :]
        side = int(math.sqrt(patch_tokens.size(1)))
        return patch_tokens.transpose(1, 2).reshape(batch_size, patch_tokens.size(2), side, side)


class WSDANModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        num_attention_maps: int,
        dropout_rate: float = 0.0,
        feature_source: str = "auto",
    ):
        super().__init__()
        self.model_name = model_name.lower()

        if self.model_name == "resnet50":
            self.backbone = ResNetFeatureBackbone(pretrained)
        elif self.model_name == "efficientnet_b0":
            self.backbone = EfficientNetFeatureBackbone(pretrained)
        elif self.model_name == "vit_base":
            self.backbone = ViTFeatureBackbone(pretrained)
        elif "convnext" in self.model_name:
            self.backbone = ConvNeXtFeatureBackbone(model_name, pretrained)
        elif "eva02" in self.model_name or self.model_name.startswith("eva"):
            self.backbone = EVA02FeatureBackbone(model_name, pretrained)
        else:
            self.backbone = TimmFeatureBackbone(
                model_name,
                pretrained,
                feature_source=feature_source,
            )

        self.num_features = self.backbone.num_features
        self.num_attention_maps = num_attention_maps
        self.attention_head = nn.Conv2d(self.num_features, num_attention_maps, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(self.num_features * num_attention_maps, num_classes)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_maps = self.backbone(images)
        attention_maps = torch.sigmoid(self.attention_head(feature_maps))

        weighted_features = torch.einsum("bkhw,bchw->bkc", attention_maps, feature_maps)
        attention_norm = attention_maps.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        weighted_features = weighted_features / attention_norm

        descriptor = weighted_features.flatten(1)
        descriptor = torch.sign(descriptor) * torch.sqrt(descriptor.abs() + 1e-9)
        descriptor = F.normalize(descriptor, dim=1)
        logits = self.classifier(self.dropout(descriptor))

        return {
            "logits": logits,
            "attention_maps": attention_maps,
            "feature_maps": feature_maps,
            "descriptor": descriptor,
        }


def sample_attention_map(attention_maps: torch.Tensor) -> torch.Tensor:
    batch_size, num_attention_maps, _, _ = attention_maps.shape
    indices = torch.randint(0, num_attention_maps, (batch_size,), device=attention_maps.device)
    selected = attention_maps[torch.arange(batch_size, device=attention_maps.device), indices]
    return selected.unsqueeze(1)


def attention_crop(images: torch.Tensor, attention_maps: torch.Tensor, threshold: float, padding_ratio: float) -> torch.Tensor:
    resized_attention = F.interpolate(attention_maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
    cropped_images = []
    image_height, image_width = images.shape[-2:]

    for image, attention in zip(images, resized_attention):
        attention_2d = attention[0]
        max_value = attention_2d.max()
        if max_value <= 0:
            cropped_images.append(image)
            continue

        mask = attention_2d >= (max_value * threshold)
        if not mask.any():
            cropped_images.append(image)
            continue

        coords = mask.nonzero(as_tuple=False)
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item()) + 1
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item()) + 1

        pad_y = int((y_max - y_min) * padding_ratio)
        pad_x = int((x_max - x_min) * padding_ratio)
        y_min = max(0, y_min - pad_y)
        y_max = min(image_height, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(image_width, x_max + pad_x)

        cropped = image[:, y_min:y_max, x_min:x_max].unsqueeze(0)
        cropped = F.interpolate(cropped, size=(image_height, image_width), mode="bilinear", align_corners=False)
        cropped_images.append(cropped.squeeze(0))

    return torch.stack(cropped_images, dim=0)


def attention_drop(images: torch.Tensor, attention_maps: torch.Tensor, threshold: float) -> torch.Tensor:
    resized_attention = F.interpolate(attention_maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
    keep_mask = (resized_attention < threshold).float()
    return images * keep_mask


def compute_wsdan_inference_logits(model, images: torch.Tensor, wsdan_config: WSDANConfig) -> torch.Tensor:
    base_outputs = model(images)
    base_logits = get_logits_from_output(base_outputs)

    if not isinstance(base_outputs, dict):
        return base_logits

    if wsdan_config.inference_crop_weight <= 0:
        return base_logits

    attention_maps = base_outputs.get("attention_maps")
    if attention_maps is None:
        return base_logits

    selected_attention = select_peak_attention_map(attention_maps.detach())
    crop_images = attention_crop(
        images,
        selected_attention,
        threshold=wsdan_config.crop_threshold,
        padding_ratio=wsdan_config.bbox_padding_ratio,
    )
    crop_outputs = model(crop_images)
    crop_logits = get_logits_from_output(crop_outputs)
    return base_logits + wsdan_config.inference_crop_weight * crop_logits


def compute_wsdan_loss(
    criterion,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    crop_logits: Optional[torch.Tensor],
    drop_logits: Optional[torch.Tensor],
    wsdan_config: WSDANConfig,
):
    base_loss = criterion(base_logits, labels)
    total_loss = wsdan_config.base_weight * base_loss

    if crop_logits is not None and wsdan_config.crop_weight > 0:
        crop_loss = criterion(crop_logits, labels)
        total_loss = total_loss + wsdan_config.crop_weight * crop_loss
    else:
        crop_loss = base_loss.detach().new_zeros(())

    if drop_logits is not None and wsdan_config.drop_weight > 0:
        drop_loss = criterion(drop_logits, labels)
        total_loss = total_loss + wsdan_config.drop_weight * drop_loss
    else:
        drop_loss = base_loss.detach().new_zeros(())

    return total_loss, {
        "base_loss": base_loss.detach(),
        "crop_loss": crop_loss.detach(),
        "drop_loss": drop_loss.detach(),
    }
