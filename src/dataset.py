import os
import random
from collections import defaultdict
from typing import Tuple, List, Optional, Dict

import torch
from PIL import ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import get_model_weights


def load_bounding_boxes(data_root: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Load CUB-200-2011 bounding boxes indexed by relative image path.

    Returns dict mapping 'NNN.Species/filename.jpg' -> (x, y, width, height).
    """
    cub_root = os.path.dirname(data_root.rstrip(os.sep).rstrip("/").rstrip("\\"))
    images_file = os.path.join(cub_root, "images.txt")
    bbox_file = os.path.join(cub_root, "bounding_boxes.txt")

    if not os.path.exists(bbox_file):
        raise FileNotFoundError(f"Bounding box file not found: {bbox_file}")

    id_to_path = {}
    with open(images_file) as f:
        for line in f:
            img_id, path = line.strip().split(" ", 1)
            id_to_path[int(img_id)] = path

    bboxes = {}
    with open(bbox_file) as f:
        for line in f:
            tokens = line.strip().split()
            img_id = int(tokens[0])
            x, y, w, h = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
            bboxes[id_to_path[img_id]] = (x, y, w, h)

    return bboxes


class CUBBBoxCropDataset(datasets.ImageFolder):
    """ImageFolder that crops to bird bounding box before transforms.

    Uses ground-truth bounding boxes from CUB-200-2011 annotations.
    Adds configurable padding around the bbox to include context.
    """

    def __init__(self, root, transform=None, bboxes=None, pad_fraction=0.1):
        super().__init__(root, transform=None)
        self.custom_transform = transform
        self.bboxes = bboxes or {}
        self.pad_fraction = pad_fraction

        self._path_to_rel = {}
        for idx, (full_path, _label) in enumerate(self.samples):
            rel = os.path.join(
                os.path.basename(os.path.dirname(full_path)),
                os.path.basename(full_path),
            )
            self._path_to_rel[idx] = rel

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        rel_path = self._path_to_rel.get(index, "")
        bbox = self.bboxes.get(rel_path)
        if bbox is not None:
            img_w, img_h = image.size
            x, y, w, h = bbox
            pad_w = w * self.pad_fraction
            pad_h = h * self.pad_fraction
            x1 = max(0, int(x - pad_w))
            y1 = max(0, int(y - pad_h))
            x2 = min(img_w, int(x + w + pad_w))
            y2 = min(img_h, int(y + h + pad_h))
            image = image.crop((x1, y1, x2, y2))

        if self.custom_transform:
            image = self.custom_transform(image)

        return image, label


def load_part_locations(data_root: str) -> Dict[str, List[Tuple[float, float, int]]]:
    """Load CUB-200-2011 part locations indexed by relative image path.

    Returns dict mapping 'NNN.Species/filename.jpg' -> [(x, y, visible), ...] for 15 parts.
    """
    cub_root = os.path.dirname(data_root.rstrip(os.sep).rstrip("/").rstrip("\\"))
    images_file = os.path.join(cub_root, "images.txt")
    parts_file = os.path.join(cub_root, "parts", "part_locs.txt")

    if not os.path.exists(parts_file):
        return {}

    id_to_path = {}
    with open(images_file) as f:
        for line in f:
            img_id, path = line.strip().split(" ", 1)
            id_to_path[int(img_id)] = path

    parts_by_path = defaultdict(list)
    with open(parts_file) as f:
        for line in f:
            tokens = line.strip().split()
            img_id = int(tokens[0])
            x = float(tokens[2])
            y = float(tokens[3])
            visible = int(tokens[4])
            rel_path = id_to_path[img_id]
            parts_by_path[rel_path].append((x, y, visible))

    return dict(parts_by_path)


class CUBPartMaskingDataset(datasets.ImageFolder):
    """ImageFolder that masks random body parts on PIL images before transforms.

    Uses ground-truth part locations from CUB-200-2011 annotations.
    Masking happens on the original PIL image before any resize/crop transforms,
    so coordinates map directly.
    """

    def __init__(self, root, transform=None, part_locs=None, mask_prob=0.3, mask_radius_frac=0.08):
        super().__init__(root, transform=None)
        self.custom_transform = transform
        self.part_locs = part_locs or {}
        self.mask_prob = mask_prob
        self.mask_radius_frac = mask_radius_frac

        # Build index from sample path to relative path for part lookup
        self._path_to_rel = {}
        for idx, (full_path, _label) in enumerate(self.samples):
            rel = os.path.join(
                os.path.basename(os.path.dirname(full_path)),
                os.path.basename(full_path),
            )
            self._path_to_rel[idx] = rel

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        # Apply part masking on PIL image before transforms
        if random.random() < self.mask_prob:
            rel_path = self._path_to_rel.get(index, "")
            parts = self.part_locs.get(rel_path, [])
            visible_parts = [(x, y) for x, y, v in parts if v == 1]

            if visible_parts:
                n_mask = random.randint(1, min(2, len(visible_parts)))
                selected = random.sample(visible_parts, n_mask)
                w, h = image.size
                radius = int(self.mask_radius_frac * min(w, h))
                draw = ImageDraw.Draw(image)
                for px, py in selected:
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=(0, 0, 0),
                    )

        if self.custom_transform:
            image = self.custom_transform(image)

        return image, label


def build_transforms(
    model_name: str,
    pretrained: bool = True,
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    weights = get_model_weights(model_name, pretrained)

    if weights is None and not pretrained:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return train_transform, val_transform

    if weights is not None:
        eval_transform = weights.transforms()
        normalize = transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def get_labels_from_samples(samples) -> List[int]:
    return [label for _, label in samples]


def stratified_sample_indices(
    labels: List[int],
    max_samples: Optional[int],
    seed: int = 42,
) -> List[int]:
    total_size = len(labels)

    if max_samples is None or max_samples >= total_size:
        return list(range(total_size))

    rng = random.Random(seed)

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_to_indices.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)

        ratio = len(indices) / total_size
        take_n = max(1, round(ratio * max_samples))
        take_n = min(take_n, len(indices))

        selected_indices.extend(shuffled[:take_n])

    if len(selected_indices) > max_samples:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:max_samples]

    if len(selected_indices) < max_samples:
        selected_set = set(selected_indices)
        remaining = [i for i in range(total_size) if i not in selected_set]
        rng.shuffle(remaining)
        need = max_samples - len(selected_indices)
        selected_indices.extend(remaining[:need])

    rng.shuffle(selected_indices)
    return selected_indices


def stratified_train_val_split(
    indices: List[int],
    labels: List[int],
    val_split: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)

    class_to_indices = defaultdict(list)
    for idx in indices:
        class_to_indices[labels[idx]].append(idx)

    train_indices = []
    val_indices = []

    for label, class_indices in class_to_indices.items():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue

        val_count = int(len(shuffled) * val_split)
        val_count = max(1, val_count)
        val_count = min(val_count, len(shuffled) - 1)

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def load_official_split(data_root: str) -> Tuple[List[str], List[str]]:
    """Load CUB-200-2011 official train/test split.

    Derives the CUB root from data_root (which points to .../CUB_200_2011/images/).
    Reads images.txt and train_test_split.txt to return train and test image paths.
    """
    cub_root = os.path.dirname(data_root.rstrip(os.sep).rstrip("/").rstrip("\\"))

    images_file = os.path.join(cub_root, "images.txt")
    split_file = os.path.join(cub_root, "train_test_split.txt")

    if not os.path.exists(images_file) or not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Official split files not found. Expected:\n"
            f"  {images_file}\n"
            f"  {split_file}\n"
            f"Derived CUB root: {cub_root}"
        )

    image_paths = {}
    with open(images_file, "r") as f:
        for line in f:
            img_id, img_path = line.strip().split(" ", 1)
            image_paths[int(img_id)] = img_path

    train_paths = []
    test_paths = []
    with open(split_file, "r") as f:
        for line in f:
            img_id, is_train = line.strip().split(" ", 1)
            img_id = int(img_id)
            path = image_paths[img_id]
            if int(is_train) == 1:
                train_paths.append(path)
            else:
                test_paths.append(path)

    return train_paths, test_paths


def official_split_to_indices(
    dataset: datasets.ImageFolder,
    train_paths: List[str],
) -> List[int]:
    """Map official train paths to ImageFolder sample indices."""
    path_to_index = {}
    for idx, (full_path, _label) in enumerate(dataset.samples):
        rel_path = os.path.join(
            os.path.basename(os.path.dirname(full_path)),
            os.path.basename(full_path),
        )
        path_to_index[rel_path] = idx

    train_indices = []
    for rel_path in train_paths:
        if rel_path in path_to_index:
            train_indices.append(path_to_index[rel_path])

    return train_indices


def build_dataloaders(
    data_root: str,
    model_name: str,
    pretrained: bool = True,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.1,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42,
    prefetch_factor: int = 2,
    use_official_split: bool = False,
    use_part_masking: bool = False,
    part_mask_prob: float = 0.3,
    part_mask_radius: float = 0.08,
    use_bbox_crop: bool = False,
    bbox_pad_fraction: float = 0.1,
):
    train_transform, val_transform = build_transforms(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
    )

    if use_bbox_crop:
        bboxes = load_bounding_boxes(data_root)
        print(f"Bbox crop enabled: {len(bboxes)} images with bounding boxes, "
              f"pad_fraction={bbox_pad_fraction}")
        train_dataset_full = CUBBBoxCropDataset(
            root=data_root, transform=train_transform,
            bboxes=bboxes, pad_fraction=bbox_pad_fraction,
        )
        val_dataset_full = CUBBBoxCropDataset(
            root=data_root, transform=val_transform,
            bboxes=bboxes, pad_fraction=bbox_pad_fraction,
        )
    elif use_part_masking:
        part_locs = load_part_locations(data_root)
        print(f"Part masking enabled: {len(part_locs)} images with part annotations, "
              f"prob={part_mask_prob}, radius={part_mask_radius}")
        train_dataset_full = CUBPartMaskingDataset(
            root=data_root, transform=train_transform,
            part_locs=part_locs, mask_prob=part_mask_prob,
            mask_radius_frac=part_mask_radius,
        )
        val_dataset_full = datasets.ImageFolder(root=data_root, transform=val_transform)
    else:
        train_dataset_full = datasets.ImageFolder(root=data_root, transform=train_transform)
        val_dataset_full = datasets.ImageFolder(root=data_root, transform=val_transform)

    labels = get_labels_from_samples(train_dataset_full.samples)
    class_names = train_dataset_full.classes

    if use_official_split:
        train_paths, _test_paths = load_official_split(data_root)
        official_train_indices = official_split_to_indices(train_dataset_full, train_paths)
        print(f"Official split: {len(official_train_indices)} train images "
              f"(out of {len(labels)} total)")

        selected_indices = stratified_sample_indices(
            labels=[labels[i] for i in official_train_indices],
            max_samples=max_samples,
            seed=seed,
        )
        selected_indices = [official_train_indices[i] for i in selected_indices]
    else:
        selected_indices = stratified_sample_indices(
            labels=labels,
            max_samples=max_samples,
            seed=seed,
        )

    train_indices, val_indices = stratified_train_val_split(
        indices=selected_indices,
        labels=labels,
        val_split=val_split,
        seed=seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    use_pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, class_names
