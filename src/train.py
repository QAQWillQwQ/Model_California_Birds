import csv
import os
import signal
import shutil
import sys
import time
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders
from models import build_model, freeze_backbone
from utils import load_config, set_seed, get_device, ensure_dir, save_checkpoint


RUN_CONTEXT = {
    "config": None,
    "archive_started": False,
    "termination_requested": False,
}


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


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def autocast_context(device, enabled):
    if not enabled or device.type not in ("cuda", "cpu"):
        return _nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    return _nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled, log_interval=20):
    model.train()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    epoch_start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    epoch_time = time.perf_counter() - epoch_start_time
    print(f"  Train Epoch Time: {epoch_time:.3f}s")
    return epoch_loss, epoch_top1, epoch_top5, epoch_time


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, amp_enabled, log_interval=20):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    epoch_start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        with autocast_context(device, amp_enabled):
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
    epoch_time = time.perf_counter() - epoch_start_time
    print(f"  Val Epoch Time: {epoch_time:.3f}s")
    return epoch_loss, epoch_top1, epoch_top5, epoch_time


def append_experiment_result(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def parse_args():
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


def prepare_output_dirs(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"output_{timestamp}")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")

    ensure_dir(run_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    print(f"Output directory: {run_dir}")
    return run_dir, checkpoints_dir, logs_dir


def clear_logs_dir(logs_dir):
    if not os.path.isdir(logs_dir):
        return

    for entry in os.listdir(logs_dir):
        entry_path = os.path.join(logs_dir, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)


def archive_output_dir(output_dir):
    output_dir = os.path.abspath(output_dir)
    parent_dir = os.path.dirname(output_dir)
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    archive_dir = os.path.join(parent_dir, f"output_{unique_id}")
    shutil.copytree(output_dir, archive_dir)
    return archive_dir


def archive_outputs_if_enabled(config):
    if not config or not config.get("archive_outputs", True):
        return None

    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        return None

    print(f"Starting archive of outputs from: {output_dir}")
    return archive_output_dir(output_dir)


def handle_termination_signal(signum, _frame):
    signal_name = signal.Signals(signum).name

    if RUN_CONTEXT["termination_requested"]:
        print(f"Termination already in progress after {signal_name}; waiting for archive to finish.")
        return

    RUN_CONTEXT["termination_requested"] = True
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    print(f"\nReceived {signal_name}. Archiving outputs before shutdown...")

    config = RUN_CONTEXT["config"]
    archive_dir = None
    if not RUN_CONTEXT["archive_started"]:
        RUN_CONTEXT["archive_started"] = True
        try:
            archive_dir = archive_outputs_if_enabled(config)
        except Exception as exc:
            print(f"Failed to archive outputs during shutdown: {exc}")

    if archive_dir:
        print(f"Archived outputs to: {archive_dir}")

    raise KeyboardInterrupt


def register_termination_handlers(config):
    RUN_CONTEXT["config"] = config
    RUN_CONTEXT["archive_started"] = False
    RUN_CONTEXT["termination_requested"] = False
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)


def build_training_state(config, device, num_classes):
    dropout_rate = float(config.get("dropout_rate", 0.0))
    model = build_model(
        model_name=config["model"],
        num_classes=num_classes,
        pretrained=config["pretrained"],
        dropout_rate=dropout_rate,
    ).to(device)

    freeze_backbone_enabled = config.get("freeze_backbone", False)
    if freeze_backbone_enabled:
        freeze_backbone(config["model"], model)

    compile_enabled = bool(config.get("compile_model", False))
    if compile_enabled and hasattr(torch, "compile"):
        model = torch.compile(model)

    label_smoothing = float(config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer = optim.AdamW(trainable_parameters, lr=config["learning_rate"], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    amp_enabled = bool(config.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_interval = max(1, int(config.get("val_interval", 1)))

    early_stopping_patience = int(config.get("early_stopping_patience", 0))

    return model, criterion, optimizer, scheduler, scaler, amp_enabled, val_interval, early_stopping_patience


def build_experiment_row(config, device, best_epoch, best_val_top1, best_val_top5, last_metrics):
    return {
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
        "last_train_top1": round(last_metrics["train_top1"], 6),
        "last_train_top5": round(last_metrics["train_top5"], 6),
        "last_val_top1": round(last_metrics["val_top1"], 6),
        "last_val_top5": round(last_metrics["val_top5"], 6),
        "device": str(device),
    }


def write_log_header(log_file, device, num_classes, config, amp_enabled, val_interval):
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Num classes: {num_classes}\n")
    log_file.write(f"Model: {config['model']}\n\n")
    log_file.write(f"AMP Enabled: {amp_enabled}\n")
    log_file.write(f"Compile Enabled: {config.get('compile_model', False)}\n")
    log_file.write(f"CUDNN Benchmark: {config.get('cudnn_benchmark', False)}\n")
    log_file.write(f"Freeze Backbone: {config.get('freeze_backbone', False)}\n")
    log_file.write(f"Validation Interval: {val_interval}\n")
    log_file.write(f"Weight Decay: {config.get('weight_decay', 0.0)}\n")
    log_file.write(f"Label Smoothing: {config.get('label_smoothing', 0.0)}\n")
    log_file.write(f"Dropout Rate: {config.get('dropout_rate', 0.0)}\n")
    log_file.write(f"Early Stopping Patience: {config.get('early_stopping_patience', 0)}\n")
    log_file.write(f"Scheduler: CosineAnnealingLR (T_max={config['epochs']}, eta_min=1e-6)\n\n")
    log_file.flush()


def run_validation(epoch, total_epochs, val_interval, model, val_loader, criterion, device, amp_enabled, log_interval):
    should_validate = ((epoch + 1) % val_interval == 0) or ((epoch + 1) == total_epochs)

    if should_validate:
        val_loss, val_top1, val_top5, val_time = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            log_interval=log_interval,
        )
    else:
        val_loss = float("nan")
        val_top1 = float("nan")
        val_top5 = float("nan")
        val_time = 0.0
        print(f"  Skipping validation on epoch {epoch + 1}; val_interval={val_interval}")

    return should_validate, val_loss, val_top1, val_top5, val_time


def run_training_loop(
    config,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    scaler,
    amp_enabled,
    val_interval,
    early_stopping_patience,
    checkpoints_dir,
    log_path,
    num_classes,
):
    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    total_training_time = 0.0
    total_validation_time = 0.0
    loop_start_time = time.perf_counter()
    last_metrics = {
        "train_top1": 0.0,
        "train_top5": 0.0,
        "val_top1": 0.0,
        "val_top5": 0.0,
    }
    log_interval = config.get("log_interval", 20)

    with open(log_path, "w") as log_file:
        write_log_header(log_file, device, num_classes, config, amp_enabled, val_interval)

        for epoch in range(config["epochs"]):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n===== Epoch {epoch + 1}/{config['epochs']} (LR: {current_lr:.2e}) =====")

            train_loss, train_top1, train_top5, train_time = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                log_interval=log_interval,
            )
            total_training_time += train_time

            scheduler.step()

            should_validate, val_loss, val_top1, val_top5, val_time = run_validation(
                epoch=epoch,
                total_epochs=config["epochs"],
                val_interval=val_interval,
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                log_interval=log_interval,
            )
            total_validation_time += val_time

            last_metrics["train_top1"] = train_top1
            last_metrics["train_top5"] = train_top5
            last_metrics["val_top1"] = val_top1
            last_metrics["val_top5"] = val_top5

            epoch_total_time = train_time + val_time
            line = (
                f"Epoch [{epoch + 1}/{config['epochs']}] | "
                f"Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.4f} | Train Top5: {train_top5:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.4f} | Val Top5: {val_top5:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Train Time: {train_time:.3f}s | Val Time: {val_time:.3f}s | Epoch Time: {epoch_total_time:.3f}s"
            )

            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if should_validate and val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_val_top5 = val_top5
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                best_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_best.pth')
                save_checkpoint(model, best_model_path)
                print(f"Saved best checkpoint to: {best_model_path}")
            elif should_validate:
                epochs_without_improvement += 1

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                stop_msg = f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
                print(stop_msg)
                log_file.write(stop_msg + "\n")
                log_file.flush()
                break

        final_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_last.pth')
        save_checkpoint(model, final_model_path)
        print(f"Saved last checkpoint to: {final_model_path}")

        total_loop_time = time.perf_counter() - loop_start_time
        summary_line = (
            f"\nBest Epoch: {best_epoch} | Best Val Top1: {best_val_top1:.4f} | Best Val Top5: {best_val_top5:.4f}\n"
            f"Total Train Time: {total_training_time:.3f}s | Total Val Time: {total_validation_time:.3f}s | "
            f"Total Runtime: {total_loop_time:.3f}s\n"
        )
        print(summary_line)
        log_file.write(summary_line)
        log_file.flush()

    return best_epoch, best_val_top1, best_val_top5, last_metrics


def main():

    # basic initialize
    config_path = parse_args()
    config = load_config(config_path)
    register_termination_handlers(config)

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(config.get("float32_matmul_precision", "high"))

    run_dir, checkpoints_dir, logs_dir = prepare_output_dirs(config["output_dir"])
    if config.get("clear_logs_before_run", True):
        clear_logs_dir(logs_dir)

    # training initialize
    train_loader, val_loader, class_names = build_dataloaders(
        data_root=config["data_root"],
        model_name=config["model"],
        pretrained=config["pretrained"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        val_split=config["val_split"],
        num_workers=config["num_workers"],
        max_samples=config.get("max_samples", None),
        seed=config["seed"],
        prefetch_factor=config.get("prefetch_factor", 2),
    )

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model, criterion, optimizer, scheduler, scaler, amp_enabled, val_interval, early_stopping_patience = build_training_state(
        config=config,
        device=device,
        num_classes=len(class_names),
    )

    # run training loops
    log_path = os.path.join(logs_dir, f'{config["model"]}_train_log.log')
    best_epoch, best_val_top1, best_val_top5, last_metrics = run_training_loop(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        amp_enabled=amp_enabled,
        val_interval=val_interval,
        early_stopping_patience=early_stopping_patience,
        checkpoints_dir=checkpoints_dir,
        log_path=log_path,
        num_classes=len(class_names),
    )

    csv_path = os.path.join(logs_dir, "experiment_results.csv")
    append_experiment_result(
        csv_path,
        build_experiment_row(config, device, best_epoch, best_val_top1, best_val_top5, last_metrics),
    )

    print(f"Experiment result appended to: {csv_path}")

    if config.get("archive_outputs", True):
        RUN_CONTEXT["archive_started"] = True
        archive_dir = archive_output_dir(config["output_dir"])
        print(f"Archived outputs to: {archive_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. Shutdown completed after archive handling.")
        sys.exit(130)
