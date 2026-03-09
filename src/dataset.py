import random
from collections import defaultdict
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import get_model_weights


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
):
    train_transform, val_transform = build_transforms(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
    )

    train_dataset_full = datasets.ImageFolder(root=data_root, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=data_root, transform=val_transform)

    labels = get_labels_from_samples(train_dataset_full.samples)
    class_names = train_dataset_full.classes

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
