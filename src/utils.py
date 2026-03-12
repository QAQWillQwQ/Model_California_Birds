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

def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        cleaned[clean_key] = value
    return cleaned


def _extract_model_state_dict(state):
    if not isinstance(state, dict):
        return state

    if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        return _clean_state_dict_keys(state["model_state_dict"])

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return _clean_state_dict_keys(state["state_dict"])

    if "model" in state and isinstance(state["model"], dict):
        return _clean_state_dict_keys(state["model"])

    if all(torch.is_tensor(v) for v in state.values()):
        return _clean_state_dict_keys(state)

    raise ValueError("Unable to extract model weights from checkpoint.")

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
    state_dict = _extract_model_state_dict(state)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[load_checkpoint] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[load_checkpoint] Unexpected keys: {unexpected_keys}")

    return model


def load_full_checkpoint(model, optimizer, scheduler, path, device):
    state = torch.load(path, map_location=device)

    state_dict = _extract_model_state_dict(state)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[load_full_checkpoint] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[load_full_checkpoint] Unexpected keys: {unexpected_keys}")

    start_epoch = 0
    best_val_top1 = 0.0
    best_val_top5 = 0.0

    if isinstance(state, dict):
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        start_epoch = int(state.get("epoch", -1)) + 1
        best_val_top1 = float(state.get("best_val_top1", 0.0))
        best_val_top5 = float(state.get("best_val_top5", 0.0))

    return model, optimizer, scheduler, start_epoch, best_val_top1, best_val_top5
