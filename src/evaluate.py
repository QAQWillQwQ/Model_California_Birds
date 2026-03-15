import os
import sys
import torch
import torch.nn as nn

from dataset import build_dataloaders
from models import build_model
from utils import load_config, get_device, load_checkpoint
from wsdan import get_logits_from_output


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = get_logits_from_output(outputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss_value = running_loss / total
    acc_value = correct / total
    return loss_value, acc_value


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
        pretrained=config["pretrained"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        val_split=config["val_split"],
        num_workers=config["num_workers"],
        config=config,
    )

    model = build_model(
        model_name=config["model"],
        num_classes=len(class_names),
        pretrained=False,
        config=config,
    ).to(device)

    checkpoint_path = os.path.join(
        config["output_dir"],
        "checkpoints",
        f'{config["model"]}_best.pth'
    )

    model = load_checkpoint(model, checkpoint_path, device)

    criterion = nn.CrossEntropyLoss()
    loss_value, acc_value = evaluate(model, val_loader, criterion, device)

    print(f"Evaluation Loss: {loss_value:.4f}")
    print(f"Evaluation Accuracy: {acc_value:.4f}")


if __name__ == "__main__":
    main()
