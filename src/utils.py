import os
import yaml
import random
import torch
import numpy as np


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "learning_rate" in config:
        config["learning_rate"] = float(config["learning_rate"])

    layerwise_lr = config.get("layerwise_lr")
    if isinstance(layerwise_lr, dict):
        layerwise_lr["enabled"] = bool(layerwise_lr.get("enabled", False))
        if "decay" in layerwise_lr:
            layerwise_lr["decay"] = float(layerwise_lr["decay"])
        config["layerwise_lr"] = layerwise_lr

    wsdan = config.get("wsdan")
    if isinstance(wsdan, dict):
        wsdan["enabled"] = bool(wsdan.get("enabled", False))
        if "start_epoch" in wsdan:
            wsdan["start_epoch"] = int(wsdan["start_epoch"])
        for key in (
            "crop_threshold",
            "drop_threshold",
            "crop_weight",
            "drop_weight",
            "base_weight",
            "bbox_padding_ratio",
        ):
            if key in wsdan:
                wsdan[key] = float(wsdan[key])
        if "num_attention_maps" in wsdan:
            wsdan["num_attention_maps"] = int(wsdan["num_attention_maps"])
        if "feature_source" in wsdan:
            wsdan["feature_source"] = str(wsdan["feature_source"])
        config["wsdan"] = wsdan

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def save_full_checkpoint(model, optimizer, scheduler, epoch, best_val_top1, best_val_top5, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
    }, path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model


def load_full_checkpoint(model, optimizer, scheduler, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_top1 = state["best_val_top1"]
        best_val_top5 = state["best_val_top5"]
    else:
        model.load_state_dict(state)
        start_epoch = 0
        best_val_top1 = 0.0
        best_val_top5 = 0.0
    return model, optimizer, scheduler, start_epoch, best_val_top1, best_val_top5
