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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR
from torchvision.transforms import v2 as transforms_v2

from dataset import build_dataloaders
from models import build_model, freeze_backbone
from utils import load_config, set_seed, get_device, ensure_dir, save_checkpoint, save_full_checkpoint, load_full_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def get_cub_root(data_root):
    """Derive CUB-200-2011 root from data_root (which points to .../images/)."""
    return os.path.dirname(data_root.rstrip(os.sep).rstrip("/").rstrip("\\"))


class AttributeWeightedCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing distributed by attribute similarity.

    Instead of uniform smoothing across all classes, distributes smoothing mass
    proportionally to cosine similarity of 312-dim attribute vectors.
    """

    def __init__(self, cub_root, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

        attr_file = os.path.join(cub_root, "attributes", "class_attribute_labels_continuous.txt")
        attrs = []
        with open(attr_file) as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                attrs.append(vals)
        attrs = torch.tensor(attrs, dtype=torch.float32)  # (200, 312)

        # Cosine similarity matrix
        attrs_norm = F.normalize(attrs, p=2, dim=1)
        similarity = attrs_norm @ attrs_norm.T  # (200, 200)
        similarity.fill_diagonal_(0)

        # Normalize rows to probability distributions
        row_sums = similarity.sum(dim=1, keepdim=True)
        similarity = similarity / (row_sums + 1e-8)

        self.register_buffer("similarity", similarity)
        print(f"AttributeWeightedCrossEntropy: loaded {attrs.shape[0]} classes x {attrs.shape[1]} attributes")

    def forward(self, logits, targets):
        # MixUp/CutMix produces soft targets (2D) — use standard CE
        if targets.dim() > 1:
            log_probs = F.log_softmax(logits, dim=1)
            return -(targets * log_probs).sum(dim=1).mean()

        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            one_hot = F.one_hot(targets, num_classes).float()
            attr_dist = self.similarity[targets]  # (batch, num_classes)
            smooth_targets = (1.0 - self.smoothing) * one_hot + self.smoothing * attr_dist

        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


class ConfusionAdaptiveSampler(torch.utils.data.Sampler):
    """Sampler that oversamples classes involved in the most confused pairs.

    After each epoch, call update_confusion() with the validation confusion matrix.
    The sampler boosts sampling weight for classes that the model confuses most.
    """

    def __init__(self, labels, num_classes, topk=20, boost_factor=2.0):
        self.labels = labels
        self.num_classes = num_classes
        self.topk = topk
        self.boost_factor = boost_factor
        self.weights = torch.ones(len(labels))

    def update_confusion(self, confusion_matrix):
        """Update sampling weights based on validation confusion matrix."""
        conf = confusion_matrix.clone().float()
        conf.fill_diagonal_(0)

        # Normalize by class counts to get confusion rate
        class_counts = confusion_matrix.sum(dim=1, keepdim=True).float()
        conf_rate = conf / (class_counts + 1e-8)

        # Find top-K confused pairs
        flat_topk = conf_rate.flatten().topk(min(self.topk, conf_rate.numel())).indices
        confused_classes = set()
        for idx in flat_topk:
            true_cls = (idx // self.num_classes).item()
            pred_cls = (idx % self.num_classes).item()
            if conf_rate.flatten()[idx] > 0:
                confused_classes.add(true_cls)
                confused_classes.add(pred_cls)

        # Boost weights for confused classes
        weights = torch.ones(len(self.labels))
        for i, label in enumerate(self.labels):
            if label in confused_classes:
                weights[i] = self.boost_factor
        self.weights = weights

        print(f"  ConfusionAdaptiveSampler: boosting {len(confused_classes)} confused classes "
              f"(top-{self.topk} pairs, boost={self.boost_factor}x)")

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.labels), replacement=True).tolist())

    def __len__(self):
        return len(self.labels)


class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
    def flush(self):
        self.stream.flush()


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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled, log_interval=20, mixup_cutmix=None, scheduler=None):
    model.train()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    epoch_start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        labels_for_acc = labels
        if mixup_cutmix is not None:
            images, labels = mixup_cutmix(images, labels)

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

        if scheduler is not None:
            scheduler.step()

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
    epoch_time = time.perf_counter() - epoch_start_time
    print(f"  Train Epoch Time: {epoch_time:.3f}s")
    return epoch_loss, epoch_top1, epoch_top5, epoch_time


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, amp_enabled, log_interval=20, num_classes=0):
    model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0
    epoch_start_time = time.perf_counter()

    track_confusion = num_classes > 0
    if track_confusion:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

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

        if track_confusion:
            _, preds = outputs.topk(1, dim=1)
            for t, p in zip(labels.cpu(), preds.squeeze(1).cpu()):
                confusion[t.item()][p.item()] += 1

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

    if track_confusion:
        return epoch_loss, epoch_top1, epoch_top5, epoch_time, confusion
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


def build_training_state(config, device, num_classes, steps_per_epoch=None):
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
    use_attribute_smoothing = config.get("use_attribute_smoothing", False)
    if use_attribute_smoothing:
        cub_root = get_cub_root(config["data_root"])
        criterion = AttributeWeightedCrossEntropy(cub_root, num_classes, smoothing=label_smoothing).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer = optim.AdamW(trainable_parameters, lr=config["learning_rate"], weight_decay=weight_decay)

    scheduler_type = config.get("scheduler", "cosine")
    warmup_epochs = int(config.get("warmup_epochs", 0))

    if scheduler_type == "onecycle" and steps_per_epoch:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config["learning_rate"],
            epochs=config["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        scheduler_step_per_batch = True
    elif warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        scheduler_step_per_batch = False
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
        scheduler_step_per_batch = False

    amp_enabled = bool(config.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_interval = max(1, int(config.get("val_interval", 1)))

    early_stopping_patience = int(config.get("early_stopping_patience", 0))

    mixup_alpha = float(config.get("mixup_alpha", 0.0))
    cutmix_alpha = float(config.get("cutmix_alpha", 0.0))
    mixup_cutmix = None
    if mixup_alpha > 0 or cutmix_alpha > 0:
        mix_transforms = []
        if mixup_alpha > 0:
            mix_transforms.append(transforms_v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
        if cutmix_alpha > 0:
            mix_transforms.append(transforms_v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
        mixup_cutmix = transforms_v2.RandomChoice(mix_transforms)

    return model, criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_enabled, val_interval, early_stopping_patience, mixup_cutmix


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
    scheduler_type = config.get("scheduler", "cosine")
    if scheduler_type == "onecycle":
        log_file.write(f"Scheduler: OneCycleLR (max_lr={config['learning_rate']}, pct_start=0.1)\n")
    else:
        log_file.write(f"Scheduler: CosineAnnealingLR (T_max={config['epochs']}, eta_min=1e-6)\n")
    if scheduler_type != "onecycle":
        log_file.write(f"Warmup Epochs: {config.get('warmup_epochs', 0)}\n")
    log_file.write(f"Mixup Alpha: {config.get('mixup_alpha', 0.0)}\n")
    log_file.write(f"CutMix Alpha: {config.get('cutmix_alpha', 0.0)}\n")
    if config.get("resume_from"):
        log_file.write(f"Resume From: {config['resume_from']}\n")
    log_file.write("\n")
    log_file.flush()


def run_validation(epoch, total_epochs, val_interval, model, val_loader, criterion, device, amp_enabled, log_interval, num_classes=0):
    should_validate = ((epoch + 1) % val_interval == 0) or ((epoch + 1) == total_epochs)
    confusion = None

    if should_validate:
        result = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            log_interval=log_interval,
            num_classes=num_classes,
        )
        if num_classes > 0:
            val_loss, val_top1, val_top5, val_time, confusion = result
        else:
            val_loss, val_top1, val_top5, val_time = result
    else:
        val_loss = float("nan")
        val_top1 = float("nan")
        val_top5 = float("nan")
        val_time = 0.0
        print(f"  Skipping validation on epoch {epoch + 1}; val_interval={val_interval}")

    return should_validate, val_loss, val_top1, val_top5, val_time, confusion


def run_training_loop(
    config,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scheduler_step_per_batch,
    device,
    scaler,
    amp_enabled,
    val_interval,
    early_stopping_patience,
    mixup_cutmix,
    checkpoints_dir,
    log_path,
    num_classes,
    start_epoch=0,
    resume_best_val_top1=0.0,
    resume_best_val_top5=0.0,
    confusion_sampler=None,
    tb_writer=None,
):
    best_val_top1 = resume_best_val_top1
    best_val_top5 = resume_best_val_top5
    best_epoch = start_epoch if resume_best_val_top1 > 0 else 0
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

        for epoch in range(start_epoch, config["epochs"]):
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
                mixup_cutmix=mixup_cutmix,
                scheduler=scheduler if scheduler_step_per_batch else None,
            )
            total_training_time += train_time

            if not scheduler_step_per_batch:
                scheduler.step()

            should_validate, val_loss, val_top1, val_top5, val_time, confusion = run_validation(
                epoch=epoch,
                total_epochs=config["epochs"],
                val_interval=val_interval,
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                log_interval=log_interval,
                num_classes=num_classes if confusion_sampler is not None else 0,
            )
            total_validation_time += val_time

            if confusion_sampler is not None and confusion is not None:
                confusion_sampler.update_confusion(confusion)

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

            if tb_writer is not None:
                step = epoch + 1
                tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, step)
                tb_writer.add_scalars("Top1_Accuracy", {"train": train_top1, "val": val_top1}, step)
                tb_writer.add_scalars("Top5_Accuracy", {"train": train_top5, "val": val_top5}, step)
                tb_writer.add_scalar("Learning_Rate", current_lr, step)
                tb_writer.add_scalar("Loss_Gap", val_loss - train_loss, step)
                tb_writer.add_scalar("Accuracy_Gap", train_top1 - val_top1, step)
                tb_writer.add_scalar("Epoch_Time", epoch_total_time, step)
                tb_writer.flush()

            if should_validate and val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_val_top5 = val_top5
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                best_model_path = os.path.join(checkpoints_dir, f'{config["model"]}_best.pth')
                save_checkpoint(model, best_model_path)
                best_full_path = os.path.join(checkpoints_dir, f'{config["model"]}_best_full.pth')
                save_full_checkpoint(model, optimizer, scheduler, epoch, best_val_top1, best_val_top5, best_full_path)
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
        final_full_path = os.path.join(checkpoints_dir, f'{config["model"]}_last_full.pth')
        save_full_checkpoint(model, optimizer, scheduler, epoch, best_val_top1, best_val_top5, final_full_path)
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

    debug_log_file = open(os.path.join(logs_dir, "debug.log"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, debug_log_file)
    sys.stderr = Tee(sys.stderr, debug_log_file)

    # TensorBoard
    tb_writer = None
    if HAS_TENSORBOARD:
        tb_dir = os.path.join(logs_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard logging to: {tb_dir}")
    else:
        print("TensorBoard not available (install with: pip install tensorboard)")

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
        use_official_split=config.get("use_official_split", False),
        use_part_masking=config.get("use_part_masking", False),
        part_mask_prob=float(config.get("part_mask_prob", 0.3)),
        part_mask_radius=float(config.get("part_mask_radius", 0.08)),
        use_bbox_crop=config.get("use_bbox_crop", False),
        bbox_pad_fraction=float(config.get("bbox_pad_fraction", 0.1)),
    )

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model, criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_enabled, val_interval, early_stopping_patience, mixup_cutmix = build_training_state(
        config=config,
        device=device,
        num_classes=len(class_names),
        steps_per_epoch=len(train_loader),
    )

    # set up confusion-adaptive sampler if configured
    confusion_sampler = None
    use_confusion_sampling = config.get("use_confusion_sampling", False)
    if use_confusion_sampling:
        train_labels = [train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices]
        confusion_topk = int(config.get("confusion_topk", 20))
        confusion_sampler = ConfusionAdaptiveSampler(
            labels=train_labels,
            num_classes=len(class_names),
            topk=confusion_topk,
        )
        # Rebuild train loader with the sampler instead of shuffle
        from torch.utils.data import DataLoader
        dataloader_kwargs = {
            "num_workers": config["num_workers"],
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": config["num_workers"] > 0,
        }
        if config["num_workers"] > 0:
            dataloader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config["batch_size"],
            sampler=confusion_sampler,
            **dataloader_kwargs,
        )
        print(f"Confusion-adaptive sampling enabled (top-{confusion_topk} pairs)")

    # resume from checkpoint if configured
    start_epoch = 0
    resume_best_val_top1 = 0.0
    resume_best_val_top5 = 0.0
    resume_from = config.get("resume_from", None)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model, optimizer, scheduler, start_epoch, resume_best_val_top1, resume_best_val_top5 = load_full_checkpoint(
            model, optimizer, scheduler, resume_from, device,
        )
        print(f"Resumed at epoch {start_epoch}, best val top1: {resume_best_val_top1:.4f}")

        # Override optimizer LR and rebuild scheduler from current config
        # (the checkpoint's scheduler state uses the OLD config's LR)
        # Must set BOTH 'lr' (current) AND 'initial_lr' (scheduler base)
        # PyTorch schedulers use initial_lr to compute LR, not lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = config["learning_rate"]
            param_group["initial_lr"] = config["learning_rate"]
        print(f"Reset optimizer LR to config value: {config['learning_rate']}")

        # Rebuild scheduler with new config LR and remaining epochs
        scheduler_type = config.get("scheduler", "cosine")
        warmup_epochs = int(config.get("warmup_epochs", 0))
        remaining_epochs = config["epochs"] - start_epoch

        if scheduler_type == "onecycle" and len(train_loader) > 0:
            scheduler = OneCycleLR(
                optimizer, max_lr=config["learning_rate"],
                epochs=remaining_epochs, steps_per_epoch=len(train_loader),
                pct_start=0.1, anneal_strategy="cos",
            )
            scheduler_step_per_batch = True
        elif warmup_epochs > 0 and remaining_epochs > warmup_epochs:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs - warmup_epochs, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
            scheduler_step_per_batch = False
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining_epochs), eta_min=1e-6)
            scheduler_step_per_batch = False

        print(f"Rebuilt scheduler: {scheduler_type}, T_max={remaining_epochs}, warmup={warmup_epochs}")

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
        scheduler_step_per_batch=scheduler_step_per_batch,
        device=device,
        scaler=scaler,
        amp_enabled=amp_enabled,
        val_interval=val_interval,
        early_stopping_patience=early_stopping_patience,
        mixup_cutmix=mixup_cutmix,
        checkpoints_dir=checkpoints_dir,
        log_path=log_path,
        num_classes=len(class_names),
        start_epoch=start_epoch,
        resume_best_val_top1=resume_best_val_top1,
        resume_best_val_top5=resume_best_val_top5,
        confusion_sampler=confusion_sampler,
        tb_writer=tb_writer,
    )

    if tb_writer is not None:
        tb_writer.close()

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
