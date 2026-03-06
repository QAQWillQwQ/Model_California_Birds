import csv
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders
from models import build_model
from utils import load_config, set_seed, get_device, ensure_dir, save_checkpoint


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


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=20):
    model.train()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        topk_result = compute_topk_correct(outputs, labels, topk=(1, 5))
        top1_correct += topk_result[1]
        top5_correct += topk_result[5]
        total += batch_size

        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            avg_loss = running_loss / total
            avg_top1 = top1_correct / total
            avg_top5 = top5_correct / total
            print(
                f"  Train Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {avg_loss:.4f} | Top1: {avg_top1:.4f} | Top5: {avg_top5:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_top1 = top1_correct / total
    epoch_top5 = top5_correct / total
    return epoch_loss, epoch_top1, epoch_top5


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, log_interval=20):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        topk_result = compute_topk_correct(outputs, labels, topk=(1, 5))
        top1_correct += topk_result[1]
        top5_correct += topk_result[5]
        total += batch_size

        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            avg_loss = running_loss / total
            avg_top1 = top1_correct / total
            avg_top5 = top5_correct / total
            print(
                f"  Val Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {avg_loss:.4f} | Top1: {avg_top1:.4f} | Top5: {avg_top5:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_top1 = top1_correct / total
    epoch_top5 = top5_correct / total
    return epoch_loss, epoch_top1, epoch_top5


def append_experiment_result(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    output_dir = config["output_dir"]
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    logs_dir = os.path.join(output_dir, "logs")

    ensure_dir(output_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    train_loader, val_loader, class_names = build_dataloaders(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        val_split=config["val_split"],
        num_workers=config["num_workers"],
        max_samples=config.get("max_samples", None),
        seed=config["seed"],
    )

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = build_model(
        model_name=config["model"],
        num_classes=len(class_names),
        pretrained=config["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0

    last_train_top1 = 0.0
    last_train_top5 = 0.0
    last_val_top1 = 0.0
    last_val_top5 = 0.0

    log_path = os.path.join(logs_dir, f'{config["model"]}_train_log.txt')

    with open(log_path, "w") as log_file:
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Num classes: {len(class_names)}\n")
        log_file.write(f"Model: {config['model']}\n\n")

        for epoch in range(config["epochs"]):
            print(f"\n===== Epoch {epoch + 1}/{config['epochs']} =====")

            train_loss, train_top1, train_top5 = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                log_interval=config.get("log_interval", 20),
            )

            val_loss, val_top1, val_top5 = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                log_interval=config.get("log_interval", 20),
            )

            last_train_top1 = train_top1
            last_train_top5 = train_top5
            last_val_top1 = val_top1
            last_val_top5 = val_top5

            line = (
                f"Epoch [{epoch + 1}/{config['epochs']}] | "
                f"Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.4f} | Train Top5: {train_top5:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.4f} | Val Top5: {val_top5:.4f}"
            )

            print(line)
            log_file.write(line + "\n")

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_val_top5 = val_top5
                best_epoch = epoch + 1

                best_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_best.pth')
                save_checkpoint(model, best_model_path)

        final_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_last.pth')
        save_checkpoint(model, final_model_path)

        summary_line = (
            f"\nBest Epoch: {best_epoch} | Best Val Top1: {best_val_top1:.4f} | Best Val Top5: {best_val_top5:.4f}\n"
        )
        print(summary_line)
        log_file.write(summary_line)

    csv_path = os.path.join(logs_dir, "experiment_results.csv")
    append_experiment_result(
        csv_path,
        {
            "model": config["model"],
            "pretrained": config["pretrained"],
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "val_split": config["val_split"],
            "num_workers": config["num_workers"],
            "max_samples": config.get("max_samples", ""),
            "best_epoch": best_epoch,
            "best_val_top1": round(best_val_top1, 6),
            "best_val_top5": round(best_val_top5, 6),
            "last_train_top1": round(last_train_top1, 6),
            "last_train_top5": round(last_train_top5, 6),
            "last_val_top1": round(last_val_top1, 6),
            "last_val_top5": round(last_val_top5, 6),
            "device": str(device),
        },
    )

    print(f"Experiment result appended to: {csv_path}")


if __name__ == "__main__":
    main()