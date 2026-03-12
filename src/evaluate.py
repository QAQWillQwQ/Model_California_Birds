import os
import sys

import torch
import torch.nn as nn

from dataset import build_dataloaders
from models import build_model
from utils import load_config, get_device, load_checkpoint


def compute_topk_correct(outputs, labels, topk=(1, 5)):
    max_k = min(max(topk), outputs.size(1))
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    results = {}
    for k in topk:
        actual_k = min(k, outputs.size(1))
        correct_k = correct[:actual_k].reshape(-1).float().sum().item()
        results[k] = correct_k
    return results


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        topk_result = compute_topk_correct(outputs, labels, topk=(1, 5))
        top1_correct += topk_result[1]
        top5_correct += topk_result[5]
        total += batch_size

    loss_value = running_loss / total
    top1_value = top1_correct / total
    top5_value = top5_correct / total
    return loss_value, top1_value, top5_value


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/evaluate.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    device = get_device()

    _, val_loader, class_names = build_dataloaders(
        data_root=config["data_root"],
        model_name=config["model"],
        pretrained=bool(config.get("pretrained", True)),
        batch_size=int(config["batch_size"]),
        image_size=int(config["image_size"]),
        val_split=float(config["val_split"]),
        num_workers=int(config["num_workers"]),
        max_samples=config.get("max_samples", None),
        seed=int(config["seed"]),
        prefetch_factor=int(config.get("prefetch_factor", 2)),
    )

    model = build_model(
        model_name=config["model"],
        num_classes=len(class_names),
        pretrained=False,
        dropout_rate=float(config.get("dropout_rate", 0.0)),
    ).to(device)

    best_full_path = os.path.join(
        config["output_dir"],
        "checkpoints",
        f'{config["model"]}_best_full.pth'
    )

    best_path = os.path.join(
        config["output_dir"],
        "checkpoints",
        f'{config["model"]}_best.pth'
    )

    checkpoint_path = best_full_path if os.path.exists(best_full_path) else best_path
    model = load_checkpoint(model, checkpoint_path, device)

    criterion = nn.CrossEntropyLoss()
    loss_value, top1_value, top5_value = evaluate(model, val_loader, criterion, device)

    print(f"Evaluation checkpoint: {checkpoint_path}")
    print(f"Evaluation Loss      : {loss_value:.4f}")
    print(f"Evaluation Top1      : {top1_value:.4f}")
    print(f"Evaluation Top5      : {top5_value:.4f}")


if __name__ == "__main__":
    main()