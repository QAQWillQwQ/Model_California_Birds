import os
import sys
import time

import torch
import torch.nn as nn

from dataset import build_transforms, load_official_split, official_split_to_indices
from models import build_model
from utils import load_config, get_device, load_checkpoint

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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
def evaluate(model, loader, criterion, device, num_classes=200):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

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

        _, preds = outputs.topk(1, dim=1)
        for t, p in zip(labels.cpu(), preds.squeeze(1).cpu()):
            confusion[t.item()][p.item()] += 1

    loss_val = running_loss / total
    top1 = top1_correct / total
    top5 = top5_correct / total
    return loss_val, top1, top5, confusion


def build_tta_transforms(image_size, normalize):
    """Build 10-crop TTA transforms: center + 4 corners, each with horizontal flip."""
    base_size = int(image_size * 1.143)  # ~256 for 224, ~438 for 384

    return transforms.Compose([
        transforms.Resize(base_size),
        transforms.TenCrop(image_size),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([transforms.ToTensor(), normalize])(crop) for crop in crops
        ])),
    ])


@torch.no_grad()
def evaluate_tta(model, loader, criterion, device, num_classes=200):
    """Evaluate with 10-crop TTA (center + 4 corners + horizontal flips)."""
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        # images shape: (batch, 10, C, H, W) from TenCrop
        labels = labels.to(device)
        bs, ncrops, c, h, w = images.size()
        images = images.view(-1, c, h, w).to(device)

        outputs = model(images)
        outputs = outputs.view(bs, ncrops, -1).mean(dim=1)  # average over crops

        loss = criterion(outputs, labels)
        running_loss += loss.item() * bs

        topk = compute_topk(outputs, labels, topk=(1, 5))
        top1_correct += topk[1]
        top5_correct += topk[5]
        total += bs

        _, preds = outputs.topk(1, dim=1)
        for t, p in zip(labels.cpu(), preds.squeeze(1).cpu()):
            confusion[t.item()][p.item()] += 1

    loss_val = running_loss / total
    top1 = top1_correct / total
    top5 = top5_correct / total
    return loss_val, top1, top5, confusion


def build_test_loader(data_root, model_name, pretrained, image_size, batch_size, num_workers, use_tta=False):
    """Build a DataLoader for the official CUB-200-2011 test set."""
    from models import get_model_weights

    weights = get_model_weights(model_name, pretrained)
    if weights is not None:
        eval_transform = weights.transforms()
        normalize = transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if use_tta:
        test_transform = build_tta_transforms(image_size, normalize)
    else:
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    dataset_full = datasets.ImageFolder(root=data_root, transform=test_transform)
    _train_paths, test_paths = load_official_split(data_root)
    test_indices = official_split_to_indices(dataset_full, test_paths)

    print(f"Official test set: {len(test_indices)} images")

    test_dataset = Subset(dataset_full, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size if not use_tta else max(1, batch_size // 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_names = dataset_full.classes
    return test_loader, class_names


def print_top_confused(confusion, class_names, top_n=20):
    """Print the most confused class pairs from confusion matrix."""
    conf = confusion.clone().float()
    conf.fill_diagonal_(0)
    class_counts = confusion.sum(dim=1).float()
    conf_rate = conf / (class_counts.unsqueeze(1) + 1e-8)

    flat_topk = conf_rate.flatten().topk(top_n).indices
    print(f"\nTop {top_n} most confused class pairs:")
    print(f"{'True Class':<35} {'Predicted As':<35} {'Rate':>6} {'Count':>6}")
    print("-" * 90)
    for idx in flat_topk:
        true_cls = (idx // len(class_names)).item()
        pred_cls = (idx % len(class_names)).item()
        rate = conf_rate.flatten()[idx].item()
        count = int(conf.flatten()[idx].item())
        if count > 0:
            print(f"{class_names[true_cls]:<35} {class_names[pred_cls]:<35} {rate:>5.1%} {count:>6}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py <config_path> [checkpoint_path] [--tta] [--official]")
        print("")
        print("Options:")
        print("  checkpoint_path   Path to .pth file (default: auto-detect from config)")
        print("  --tta             Enable 10-crop test-time augmentation")
        print("  --official        Evaluate on CUB-200-2011 official test set (5794 images)")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    device = get_device()

    use_tta = "--tta" in sys.argv
    use_official = "--official" in sys.argv

    # Find checkpoint
    checkpoint_path = None
    for arg in sys.argv[2:]:
        if arg.endswith(".pth"):
            checkpoint_path = arg
            break

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config["output_dir"], "checkpoints", f'{config["model"]}_best.pth'
        )

    print(f"Device: {device}")
    print(f"Model: {config['model']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image size: {config['image_size']}")
    print(f"TTA: {use_tta}")
    print(f"Official test set: {use_official}")
    print()

    if use_official:
        loader, class_names = build_test_loader(
            data_root=config["data_root"],
            model_name=config["model"],
            pretrained=config.get("pretrained", True),
            image_size=config["image_size"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            use_tta=use_tta,
        )
    else:
        from dataset import build_dataloaders
        _, loader, class_names = build_dataloaders(
            data_root=config["data_root"],
            model_name=config["model"],
            pretrained=config.get("pretrained", True),
            batch_size=config["batch_size"],
            image_size=config["image_size"],
            val_split=config["val_split"],
            num_workers=config["num_workers"],
            use_official_split=config.get("use_official_split", False),
        )

    num_classes = len(class_names)
    model = build_model(
        model_name=config["model"],
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model = load_checkpoint(model, checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()

    print(f"Evaluating on {len(loader.dataset)} images...")
    start = time.perf_counter()

    if use_tta:
        loss_val, top1, top5, confusion = evaluate_tta(model, loader, criterion, device, num_classes)
        mode = "TTA (10-crop)"
    else:
        loss_val, top1, top5, confusion = evaluate(model, loader, criterion, device, num_classes)
        mode = "Single-crop"

    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"  {mode} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Loss:      {loss_val:.4f}")
    print(f"  Top-1:     {top1:.4f} ({top1*100:.2f}%)")
    print(f"  Top-5:     {top5:.4f} ({top5*100:.2f}%)")
    print(f"  Images:    {len(loader.dataset)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'='*60}")

    # Show most confused pairs
    print_top_confused(confusion, class_names, top_n=20)

    # Per-class accuracy
    per_class_correct = confusion.diag().float()
    per_class_total = confusion.sum(dim=1).float()
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

    worst_classes = per_class_acc.argsort()[:10]
    print(f"\n10 worst-performing classes:")
    print(f"{'Class':<40} {'Accuracy':>8} {'Count':>6}")
    print("-" * 60)
    for idx in worst_classes:
        acc = per_class_acc[idx].item()
        count = int(per_class_total[idx].item())
        print(f"{class_names[idx]:<40} {acc:>7.1%} {count:>6}")


if __name__ == "__main__":
    main()
