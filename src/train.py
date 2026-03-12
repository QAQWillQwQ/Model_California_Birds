import csv
import os
import shutil
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.transforms import v2 as transforms_v2

from dataset import build_dataloaders
from models import build_model, freeze_backbone
from utils import (
    load_config,
    set_seed,
    get_device,
    ensure_dir,
    save_checkpoint,
    save_full_checkpoint,
    load_full_checkpoint,
    load_checkpoint,
)


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


def build_mixup_cutmix(num_classes, config):
    mixup_alpha = float(config.get("mixup_alpha", 0.0))
    cutmix_alpha = float(config.get("cutmix_alpha", 0.0))

    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None

    transforms_list = []

    if mixup_alpha > 0:
        transforms_list.append(
            transforms_v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
        )

    if cutmix_alpha > 0:
        transforms_list.append(
            transforms_v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
        )

    if len(transforms_list) == 1:
        return transforms_list[0]

    return transforms_v2.RandomChoice(transforms_list)


def build_scheduler(optimizer, config):
    scheduler_name = str(config.get("scheduler", "cosine")).lower()
    total_epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    min_lr = float(config.get("min_lr", 1e-6))

    if scheduler_name != "cosine":
        return None

    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    return CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=min_lr,
    )


def append_experiment_result(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    amp_enabled,
    log_interval=50,
    mixup_cutmix=None,
):
    model.train()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        labels_for_acc = labels

        if mixup_cutmix is not None:
            images, labels = mixup_cutmix(images, labels)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels_for_acc.size(0)
        running_loss += loss.item() * batch_size

        topk_result = compute_topk_correct(outputs, labels_for_acc, topk=(1, 5))
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
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    log_interval=50,
):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    set_seed(int(config["seed"]))
    device = get_device()

    if device.type == "cuda" and bool(config.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True

    matmul_precision = str(config.get("float32_matmul_precision", "")).lower()
    if hasattr(torch, "set_float32_matmul_precision") and matmul_precision in {"high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)

    output_dir = config["output_dir"]
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    logs_dir = os.path.join(output_dir, "logs")

    ensure_dir(output_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    log_path = os.path.join(logs_dir, f'{config["model"]}_train_log.txt')
    csv_path = os.path.join(logs_dir, "experiment_results.csv")

    if bool(config.get("clear_logs_before_run", False)) and not config.get("resume_from"):
        if os.path.exists(log_path):
            os.remove(log_path)

    train_loader, val_loader, class_names = build_dataloaders(
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
        pretrained=bool(config.get("pretrained", True)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
    ).to(device)

    if bool(config.get("freeze_backbone", False)):
        freeze_backbone(config["model"], model)

    if bool(config.get("compile_model", False)) and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    label_smoothing = float(config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    scheduler = build_scheduler(optimizer, config)

    amp_enabled = bool(config.get("use_amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    mixup_cutmix = build_mixup_cutmix(len(class_names), config)

    start_epoch = 0
    best_val_top1 = 0.0
    best_val_top5 = 0.0

    resume_from = config.get("resume_from", None)
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from path not found: {resume_from}")

        if resume_from.endswith("_full.pth"):
            model, optimizer, scheduler, start_epoch, best_val_top1, best_val_top5 = load_full_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=resume_from,
                device=device,
            )
            print(f"Resumed full checkpoint from: {resume_from}")
            print(f"Start epoch      : {start_epoch}")
            print(f"Best Val Top1    : {best_val_top1:.4f}")
            print(f"Best Val Top5    : {best_val_top5:.4f}")
        else:
            model = load_checkpoint(model, resume_from, device)
            print(f"Loaded model-only checkpoint from: {resume_from}")

    print(f"Using device      : {device}")
    print(f"Model             : {config['model']}")
    print(f"Image size        : {config['image_size']}")
    print(f"Batch size        : {config['batch_size']}")
    print(f"Train batches     : {len(train_loader)}")
    print(f"Val batches       : {len(val_loader)}")
    print(f"Num classes       : {len(class_names)}")
    print(f"AMP enabled       : {amp_enabled}")
    print(f"Resume from       : {resume_from if resume_from else 'None'}")

    best_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_best.pth')
    best_full_path = os.path.join(checkpoints_dir, f'{config["model"]}_best_full.pth')
    last_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_last.pth')
    last_full_path = os.path.join(checkpoints_dir, f'{config["model"]}_last_full.pth')

    val_interval = max(1, int(config.get("val_interval", 1)))
    patience = int(config.get("early_stopping_patience", 0))
    no_improve_count = 0

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Num classes: {len(class_names)}\n")
        log_file.write(f"Model: {config['model']}\n")
        log_file.write(f"Resume from: {resume_from}\n\n")

        for epoch in range(start_epoch, int(config["epochs"])):
            print(f"\n===== Epoch {epoch + 1}/{config['epochs']} =====")

            train_loss, train_top1, train_top5 = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                log_interval=int(config.get("log_interval", 50)),
                mixup_cutmix=mixup_cutmix,
            )

            if (epoch + 1) % val_interval == 0:
                val_loss, val_top1, val_top5 = validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    log_interval=int(config.get("log_interval", 50)),
                )
            else:
                val_loss, val_top1, val_top5 = 0.0, 0.0, 0.0

            if scheduler is not None:
                scheduler.step()

            line = (
                f"Epoch [{epoch + 1}/{config['epochs']}] | "
                f"Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.4f} | Train Top5: {train_top5:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.4f} | Val Top5: {val_top5:.4f}"
            )
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            improved = val_top1 > best_val_top1
            if improved:
                best_val_top1 = val_top1
                best_val_top5 = val_top5
                no_improve_count = 0

                save_checkpoint(model, best_model_path)
                save_full_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_top1=best_val_top1,
                    best_val_top5=best_val_top5,
                    path=best_full_path,
                )
            else:
                no_improve_count += 1

            save_checkpoint(model, last_model_path)
            save_full_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_top1=best_val_top1,
                best_val_top5=best_val_top5,
                path=last_full_path,
            )

            if patience > 0 and no_improve_count >= patience:
                print(f"Early stopping triggered. No improvement for {patience} validation checks.")
                break

        summary_line = (
            f"\nBest Val Top1: {best_val_top1:.4f} | "
            f"Best Val Top5: {best_val_top5:.4f}\n"
        )
        print(summary_line)
        log_file.write(summary_line)

    append_experiment_result(
        csv_path,
        {
            "model": config["model"],
            "image_size": int(config["image_size"]),
            "batch_size": int(config["batch_size"]),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config.get("weight_decay", 0.0)),
            "label_smoothing": float(config.get("label_smoothing", 0.0)),
            "dropout_rate": float(config.get("dropout_rate", 0.0)),
            "mixup_alpha": float(config.get("mixup_alpha", 0.0)),
            "cutmix_alpha": float(config.get("cutmix_alpha", 0.0)),
            "scheduler": str(config.get("scheduler", "none")),
            "warmup_epochs": int(config.get("warmup_epochs", 0)),
            "resume_from": resume_from if resume_from else "",
            "best_val_top1": round(best_val_top1, 6),
            "best_val_top5": round(best_val_top5, 6),
            "device": str(device),
        },
    )

    print(f"Experiment result saved to: {csv_path}")


if __name__ == "__main__":
    main()