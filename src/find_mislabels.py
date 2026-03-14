"""
Scan the entire dataset using a trained model to find likely mislabeled images.

Approach: For each image, compare the model's predicted class vs the folder label.
Flag images where the model confidently predicts a DIFFERENT class than the label.

Usage:
    python src/find_mislabels.py <checkpoint_path> [--threshold 0.8] [--move]

Example:
    python src/find_mislabels.py outputs/output_20260308_135130/checkpoints/convnext_base.fb_in22k_ft_in1k_best.pth
"""

import argparse
import csv
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Find mislabeled images using a trained model")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-root", default=None, help="Path to data folder (default: auto-detect)")
    parser.add_argument("--model-name", default=None, help="Model architecture name (default: infer from checkpoint filename)")
    parser.add_argument("--image-size", type=int, default=448, help="Image size for inference (default: 448)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold to flag as mislabeled (default: 0.7)")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: mislabeled_report.csv)")
    parser.add_argument("--move", action="store_true", help="Move flagged images to a quarantine folder")
    return parser.parse_args()


def infer_model_name(checkpoint_path):
    basename = os.path.basename(checkpoint_path)
    for suffix in ("_best.pth", "_last.pth", "_best_full.pth", "_last_full.pth"):
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return basename.replace(".pth", "")


def build_inference_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the sample index."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


@torch.no_grad()
def scan_dataset(model, loader, device, class_names, dataset):
    model.eval()
    results = []
    total = len(dataset)
    processed = 0
    start_time = time.perf_counter()

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        top_probs, top_preds = probs.topk(3, dim=1)

        for i in range(images.size(0)):
            idx = indices[i].item()
            true_label = labels[i].item()
            file_path = dataset.samples[idx][0]

            pred_label = top_preds[i, 0].item()
            pred_conf = top_probs[i, 0].item()
            true_conf = probs[i, true_label].item()

            pred2_label = top_preds[i, 1].item()
            pred2_conf = top_probs[i, 1].item()

            results.append({
                "file_path": file_path,
                "true_class": class_names[true_label],
                "predicted_class": class_names[pred_label],
                "predicted_confidence": pred_conf,
                "true_class_confidence": true_conf,
                "second_pred_class": class_names[pred2_label],
                "second_pred_confidence": pred2_conf,
                "is_mismatch": pred_label != true_label,
            })

        processed += images.size(0)
        if processed % 1000 < images.size(0):
            elapsed = time.perf_counter() - start_time
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  Scanned {processed}/{total} images ({processed/total*100:.1f}%) | {rate:.0f} img/s | ETA: {eta:.0f}s")

    return results


def classify_issue(row, threshold):
    """Classify the type and severity of a suspected mislabel."""
    if not row["is_mismatch"]:
        return None, None

    pred_conf = row["predicted_confidence"]
    true_conf = row["true_class_confidence"]

    if pred_conf >= 0.9 and true_conf < 0.05:
        return "CRITICAL", "Model >90% confident it's a different class, <5% for labeled class"
    if pred_conf >= threshold and true_conf < 0.1:
        return "HIGH", f"Model >{threshold*100:.0f}% confident it's different, <10% for labeled class"
    if pred_conf >= threshold:
        return "MEDIUM", f"Model >{threshold*100:.0f}% confident it's a different class"
    if pred_conf >= 0.5 and true_conf < 0.1:
        return "LOW", "Model somewhat confident, very low confidence in labeled class"

    return None, None


def write_report(flagged, output_csv):
    if not flagged:
        print("No mislabeled images detected.")
        return

    fieldnames = [
        "severity", "reason", "file_path", "true_class", "predicted_class",
        "predicted_confidence", "true_class_confidence",
        "second_pred_class", "second_pred_confidence",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flagged:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Report written to: {output_csv}")


def quarantine_flagged(flagged, data_root):
    quarantine_root = os.path.join(os.path.dirname(data_root), "quarantine")
    moved = 0

    for row in flagged:
        src = row["file_path"]
        if not os.path.exists(src):
            continue

        rel = os.path.relpath(src, data_root)
        dst = os.path.join(quarantine_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved += 1

    print(f"Moved {moved} images to: {quarantine_root}")
    return moved


def main():
    args = parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    data_root = args.data_root or os.path.join(project_root, "data", "CUB_200_2011", "images")
    model_name = args.model_name or infer_model_name(checkpoint_path)
    output_csv = args.output_csv or os.path.join(project_root, "mislabeled_report.csv")

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")
    print(f"Image size: {args.image_size}")
    print(f"Confidence threshold: {args.threshold}")
    print()

    # Build dataset (no augmentation — just resize + normalize)
    transform = build_inference_transform(args.image_size)
    dataset = IndexedImageFolder(root=data_root, transform=transform)
    class_names = dataset.classes

    print(f"Dataset: {len(dataset)} images across {len(class_names)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build model and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    print(f"Loaded checkpoint successfully")
    print()

    # Scan all images
    print("Scanning dataset for mislabeled images...")
    results = scan_dataset(model, loader, device, class_names, dataset)

    # Classify and filter flagged images
    flagged = []
    for row in results:
        severity, reason = classify_issue(row, args.threshold)
        if severity is not None:
            row["severity"] = severity
            row["reason"] = reason
            flagged.append(row)

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    flagged.sort(key=lambda r: (severity_order.get(r["severity"], 99), -r["predicted_confidence"]))

    # Print summary
    total_mismatches = sum(1 for r in results if r["is_mismatch"])
    print()
    print(f"=== SCAN RESULTS ===")
    print(f"Total images scanned: {len(results)}")
    print(f"Total mismatches (pred != label): {total_mismatches} ({total_mismatches/len(results)*100:.1f}%)")
    print(f"Flagged as likely mislabeled: {len(flagged)} ({len(flagged)/len(results)*100:.2f}%)")
    print()

    severity_counts = {}
    for row in flagged:
        severity_counts[row["severity"]] = severity_counts.get(row["severity"], 0) + 1

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = severity_counts.get(sev, 0)
        if count > 0:
            print(f"  {sev}: {count}")

    print()

    # Show top 20 most confident mislabels
    print("Top 20 most confident mislabels:")
    print(f"{'Severity':<10} {'Pred Conf':>9} {'True Conf':>9}  {'True Class':<30} {'Predicted Class':<30} File")
    print("-" * 130)
    for row in flagged[:20]:
        true_cls = row['true_class'][:28]
        pred_cls = row['predicted_class'][:28]
        fname = os.path.basename(row['file_path'])
        print(f"{row['severity']:<10} {row['predicted_confidence']:>9.4f} {row['true_class_confidence']:>9.4f}  {true_cls:<30} {pred_cls:<30} {fname}")

    # Write full report
    write_report(flagged, output_csv)

    # Optionally move flagged images
    if args.move and flagged:
        print()
        quarantine_flagged(flagged, data_root)

    # Per-class summary
    class_error_counts = {}
    for row in flagged:
        cls = row["true_class"]
        class_error_counts[cls] = class_error_counts.get(cls, 0) + 1

    if class_error_counts:
        print()
        print("Classes with most flagged images:")
        sorted_classes = sorted(class_error_counts.items(), key=lambda x: -x[1])
        for cls, count in sorted_classes[:20]:
            print(f"  {cls}: {count} flagged images")


if __name__ == "__main__":
    main()
