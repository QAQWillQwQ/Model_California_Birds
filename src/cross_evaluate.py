"""Cross-evaluate models on both full-image and bbox-cropped val sets.

Answers: is the 92.25% bbox model truly better, or did bbox val just make it easier?

Usage:
    python src/cross_evaluate.py

Evaluates:
  1. Bbox model on bbox-cropped val set (training distribution)
  2. Bbox model on full-image val set (cross-evaluation)
  3. Baseline model on full-image val set (training distribution)
  4. Baseline model on bbox-cropped val set (cross-evaluation)
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dataset import (
    build_transforms, load_official_split, official_split_to_indices,
    load_bounding_boxes, CUBBBoxCropDataset, stratified_sample_indices,
    stratified_train_val_split, get_labels_from_samples,
)
from models import build_model, get_model_weights
from utils import get_device, load_checkpoint, set_seed


def compute_topk(outputs, labels, topk=(1, 5)):
    max_k = min(max(topk), outputs.size(1))
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    results = {}
    for k in topk:
        actual_k = min(k, outputs.size(1))
        results[k] = correct[:actual_k].reshape(-1).float().sum().item()
    return results


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    top1_correct = 0.0
    top5_correct = 0.0
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        topk = compute_topk(outputs, labels, topk=(1, 5))
        top1_correct += topk[1]
        top5_correct += topk[5]
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "top1": top1_correct / total,
        "top5": top5_correct / total,
        "total": total,
    }


def build_val_loader(data_root, model_name, image_size, batch_size, num_workers,
                     use_bbox_crop=False, bbox_pad_fraction=0.1):
    """Build val loader with or without bbox crops."""
    weights = get_model_weights(model_name, True)
    if weights is not None:
        eval_transform = weights.transforms()
        normalize = transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if use_bbox_crop:
        bboxes = load_bounding_boxes(data_root)
        dataset = CUBBBoxCropDataset(
            root=data_root, transform=val_transform,
            bboxes=bboxes, pad_fraction=bbox_pad_fraction,
        )
    else:
        dataset = datasets.ImageFolder(root=data_root, transform=val_transform)

    labels = get_labels_from_samples(dataset.samples)

    # Use official split, then carve val from train
    train_paths, _ = load_official_split(data_root)
    official_train_indices = official_split_to_indices(dataset, train_paths)

    selected_indices = stratified_sample_indices(
        labels=[labels[i] for i in official_train_indices],
        max_samples=None,
        seed=42,
    )
    selected_indices = [official_train_indices[i] for i in selected_indices]

    _, val_indices = stratified_train_val_split(
        indices=selected_indices,
        labels=labels,
        val_split=0.15,
        seed=42,
    )

    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return val_loader, dataset.classes


def main():
    set_seed(42)
    device = get_device()

    # Paths
    data_root = "/mnt/c/Users/ljh20/MCS/CS273P/Model_California_Birds/data/CUB_200_2011/images"
    bbox_checkpoint = "/mnt/c/Users/ljh20/MCS/CS273P/Model_California_Birds/outputs/output_20260314_113627/checkpoints/convnextv2_base_best.pth"
    baseline_checkpoint = "/mnt/c/Users/ljh20/MCS/CS273P/Model_California_Birds/outputs/output_20260314_090351/checkpoints/convnextv2_base_best.pth"
    model_name = "convnextv2_base"
    image_size = 384
    batch_size = 12
    num_workers = 4

    # Build both val sets
    print("Building full-image val set...")
    full_loader, class_names = build_val_loader(
        data_root, model_name, image_size, batch_size, num_workers,
        use_bbox_crop=False,
    )
    print(f"  Val images: {len(full_loader.dataset)}")

    print("Building bbox-cropped val set...")
    bbox_loader, _ = build_val_loader(
        data_root, model_name, image_size, batch_size, num_workers,
        use_bbox_crop=True, bbox_pad_fraction=0.1,
    )
    print(f"  Val images: {len(bbox_loader.dataset)}")

    num_classes = len(class_names)

    # Load both models
    print(f"\nLoading bbox model: {bbox_checkpoint}")
    bbox_model = build_model(model_name, num_classes, pretrained=False).to(device)
    bbox_model = load_checkpoint(bbox_model, bbox_checkpoint, device)

    print(f"Loading baseline model: {baseline_checkpoint}")
    baseline_model = build_model(model_name, num_classes, pretrained=False).to(device)
    baseline_model = load_checkpoint(baseline_model, baseline_checkpoint, device)

    # Cross-evaluate
    print("\n" + "=" * 70)
    print("  CROSS-EVALUATION: Models x Val Sets")
    print("=" * 70)

    results = {}

    configs = [
        ("Bbox model",     bbox_model,     "bbox val",  bbox_loader),
        ("Bbox model",     bbox_model,     "full val",  full_loader),
        ("Baseline model", baseline_model, "full val",  full_loader),
        ("Baseline model", baseline_model, "bbox val",  bbox_loader),
    ]

    for model_label, model, val_label, loader in configs:
        key = f"{model_label} → {val_label}"
        print(f"\n  Evaluating: {key}...")
        start = time.perf_counter()
        r = evaluate_model(model, loader, device)
        elapsed = time.perf_counter() - start
        r["time"] = elapsed
        results[key] = r
        print(f"    Top-1: {r['top1']*100:.2f}%  |  Top-5: {r['top5']*100:.2f}%  |  "
              f"Loss: {r['loss']:.4f}  |  {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model → Val Set':<35} {'Top-1':>8} {'Top-5':>8} {'Loss':>8}")
    print("  " + "-" * 63)
    for key, r in results.items():
        marker = " *" if r["top1"] == max(v["top1"] for v in results.values()) else ""
        print(f"  {key:<35} {r['top1']*100:>7.2f}% {r['top5']*100:>7.2f}% {r['loss']:>8.4f}{marker}")

    print("\n  * = best top-1 accuracy")
    print(f"\n  Val set size: {len(full_loader.dataset)} images")
    print("=" * 70)


if __name__ == "__main__":
    main()
