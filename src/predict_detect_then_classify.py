import os
import sys
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import datasets

from dataset import build_transforms
from models import build_model
from utils import load_config, get_device, load_checkpoint


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def collect_image_files(input_path: str):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path]

    image_files = []
    for file_path in sorted(input_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file_path)

    return image_files


def print_section_title(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def build_classifier(config_path: str):
    config = load_config(config_path)
    device = get_device()

    dataset = datasets.ImageFolder(root=config["data_root"])
    class_names = dataset.classes

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
    model.eval()

    return config, device, class_names, model


@torch.no_grad()
def classify_crop(model, class_names, config, device, crop_image, top_k):
    _, val_transform = build_transforms(
        model_name=config["model"],
        pretrained=bool(config.get("pretrained", True)),
        image_size=int(config["image_size"]),
    )

    image_tensor = val_transform(crop_image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    actual_top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=actual_top_k, dim=1)

    return top_indices.squeeze(0).cpu().tolist(), top_probs.squeeze(0).cpu().tolist()


def detect_best_bird_crop(detector, image_path: Path):
    results = detector(str(image_path), verbose=False)
    result = results[0]

    names = result.names
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    bird_boxes = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = names[cls_id]
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()

        if cls_name.lower() == "bird":
            bird_boxes.append((conf, xyxy))

    if not bird_boxes:
        return None, None

    bird_boxes.sort(key=lambda x: x[0], reverse=True)
    best_conf, best_xyxy = bird_boxes[0]

    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in best_xyxy]
    crop = image.crop((x1, y1, x2, y2))

    return crop, {"confidence": best_conf, "bbox": [x1, y1, x2, y2]}


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python src/predict_detect_then_classify.py <config_path> <input_path> [top_k] [detector_weights]")
        print("")
        print("Example:")
        print("  python src/predict_detect_then_classify.py configs/eva02_base_resume_windows_3070ti.yaml Test_Input_Image")
        print("  python src/predict_detect_then_classify.py configs/eva02_base_resume_windows_3070ti.yaml Test_Input_Image 5 yolov8x.pt")
        sys.exit(1)

    config_path = sys.argv[1]
    input_path = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    detector_weights = sys.argv[4] if len(sys.argv) >= 5 else "yolov8x.pt"

    config, device, class_names, classifier = build_classifier(config_path)
    detector = YOLO(detector_weights)

    image_files = collect_image_files(input_path)

    print_section_title("Detect -> Crop -> Classify")
    print(f"Classifier config : {config_path}")
    print(f"Input path        : {input_path}")
    print(f"Detector weights  : {detector_weights}")
    print(f"Classifier model  : {config['model']}")
    print(f"Using device      : {device}")
    print(f"Total image files : {len(image_files)}")

    for idx, image_path in enumerate(image_files, start=1):
        print_section_title(f"Prediction {idx}/{len(image_files)}")
        print(f"Test File Name : {image_path.name}")
        print(f"Full File Path : {image_path}")

        try:
            crop_image, detect_info = detect_best_bird_crop(detector, image_path)

            if crop_image is None:
                print("Detection Result: FAILED")
                print("Reason          : No bird object was detected by the pretrained detector.")
                continue

            print("Detection Result: SUCCESS")
            print(f"Bird Confidence : {detect_info['confidence']:.4f}")
            print(f"Bird BBox       : {detect_info['bbox']}")

            top_indices, top_probs = classify_crop(
                model=classifier,
                class_names=class_names,
                config=config,
                device=device,
                crop_image=crop_image,
                top_k=top_k,
            )

            print("-" * 90)
            print(f"{'Rank':<6}{'Predicted Class':<45}{'Probability':>15}")
            print("-" * 90)
            for rank, (class_idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
                class_name = class_names[class_idx]
                print(f"{rank:<6}{class_name:<45}{prob * 100:>13.2f}%")

        except Exception as e:
            print("Detection/Classify Result: FAILED")
            print(f"Reason                  : {e}")


if __name__ == "__main__":
    main()