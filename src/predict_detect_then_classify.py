import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision import datasets

from dataset import build_transforms
from models import build_model
from utils import load_config, get_device


# Reduce extra warnings / cache noise on macOS
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff"}

DEFAULT_INPUT_PATH = "/Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image"
DEFAULT_OUTPUT_PATH = "/Users/wangyiding/ML_Model_CaliforniaBirds/Test_Output_Image"

DEFAULT_CHECKPOINT_CANDIDATES = [
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_best_full.pth",
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_best.pth",
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_last.pth",
]

# Bright green
BRIGHT_GREEN = (0, 255, 120)
PANEL_BG = (242, 255, 246)
TEXT_BLACK = (25, 25, 25)
WHITE = (255, 255, 255)
RED = (200, 40, 40)


def collect_image_files(input_path: str) -> List[Path]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input file type: {input_path.suffix}")
        return [input_path]

    image_files = []
    for file_path in sorted(input_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file_path)

    if not image_files:
        raise ValueError(f"No supported image files found in: {input_path}")

    return image_files


def print_section_title(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def clamp_bbox(bbox: List[int], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def expand_bbox(bbox: List[int], image_size: Tuple[int, int], expand_ratio: float = 0.10) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    dx = int(box_w * expand_ratio)
    dy = int(box_h * expand_ratio)

    new_bbox = [x1 - dx, y1 - dy, x2 + dx, y2 + dy]
    return clamp_bbox(new_bbox, (width, height))


def find_checkpoint_path(config: Dict, checkpoint_path_arg: Optional[str]) -> str:
    candidates = []

    if checkpoint_path_arg:
        candidates.append(checkpoint_path_arg)

    resume_from = config.get("resume_from", None)
    if resume_from:
        candidates.append(resume_from)

    output_dir = config.get("output_dir", "")
    model_name = config.get("model", "")
    if output_dir and model_name:
        candidates.append(os.path.join(output_dir, "checkpoints", f"{model_name}_best_full.pth"))
        candidates.append(os.path.join(output_dir, "checkpoints", f"{model_name}_best.pth"))
        candidates.append(os.path.join(output_dir, "checkpoints", f"{model_name}_last.pth"))

    candidates.extend(DEFAULT_CHECKPOINT_CANDIDATES)

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No checkpoint file found.\n"
        "Checked paths:\n" + "\n".join(candidates)
    )


def clean_state_dict_keys(state_dict: Dict) -> Dict:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        cleaned[clean_key] = value
    return cleaned


def extract_model_state_dict(state) -> Dict:
    if not isinstance(state, dict):
        return state

    if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        return clean_state_dict_keys(state["model_state_dict"])

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return clean_state_dict_keys(state["state_dict"])

    if "model" in state and isinstance(state["model"], dict):
        return clean_state_dict_keys(state["model"])

    if all(torch.is_tensor(v) for v in state.values()):
        return clean_state_dict_keys(state)

    raise ValueError("Unable to extract model weights from checkpoint.")


def robust_load_checkpoint(model, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[Checkpoint] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {unexpected_keys}")

    return model


def build_classifier(config_path: str, checkpoint_path_arg: Optional[str]):
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

    checkpoint_path = find_checkpoint_path(config, checkpoint_path_arg)
    model = robust_load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # IMPORTANT:
    # Use pretrained=False here to avoid repeated HuggingFace downloads during testing.
    _, val_transform = build_transforms(
        model_name=config["model"],
        pretrained=False,
        image_size=int(config["image_size"]),
    )

    return config, device, class_names, model, checkpoint_path, val_transform


@torch.inference_mode()
def classify_crop(model, class_names, val_transform, device, crop_image: Image.Image, top_k: int):
    image_tensor = val_transform(crop_image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    actual_top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=actual_top_k, dim=1)

    top_indices = top_indices.squeeze(0).cpu().tolist()
    top_probs = top_probs.squeeze(0).cpu().tolist()

    top_results = []
    for class_idx, prob in zip(top_indices, top_probs):
        top_results.append({
            "class_name": class_names[class_idx],
            "probability": prob,
        })

    return top_results


def run_yolo_on_array(detector: YOLO, image_array: np.ndarray, imgsz: int, conf: float):
    return detector.predict(
        source=image_array,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=0.45,
        max_det=20,
        device="cpu",
    )[0]


def pick_best_bird_box(result, scale_back: float = 1.0):
    names = result.names
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None

    bird_boxes = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = names[cls_id]
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()

        if cls_name.lower() == "bird":
            mapped_xyxy = [int(v / scale_back) for v in xyxy]
            bird_boxes.append((conf, mapped_xyxy))

    if not bird_boxes:
        return None

    bird_boxes.sort(key=lambda x: x[0], reverse=True)
    best_conf, best_xyxy = bird_boxes[0]
    return {
        "confidence": best_conf,
        "bbox": best_xyxy,
    }


def detect_best_bird(detector: YOLO, image_path: Path):
    original_image = Image.open(image_path).convert("RGB")
    image_array = np.array(original_image)

    # Pass 1: original image, larger imgsz, lower conf for tiny birds
    result_1 = run_yolo_on_array(detector, image_array, imgsz=1536, conf=0.05)
    best_1 = pick_best_bird_box(result_1, scale_back=1.0)

    if best_1 is not None:
        bbox = clamp_bbox(best_1["bbox"], original_image.size)
        bbox = expand_bbox(bbox, original_image.size, expand_ratio=0.08)
        crop = original_image.crop(tuple(bbox))
        return {
            "crop_image": crop,
            "bbox": bbox,
            "confidence": best_1["confidence"],
            "original_image": original_image,
            "detect_mode": "original",
        }

    # Pass 2: upscale image for tiny birds in sky / branches
    up_scale = 2.0
    up_w = int(original_image.size[0] * up_scale)
    up_h = int(original_image.size[1] * up_scale)
    enlarged_image = original_image.resize((up_w, up_h), Image.Resampling.BICUBIC)
    enlarged_array = np.array(enlarged_image)

    result_2 = run_yolo_on_array(detector, enlarged_array, imgsz=2048, conf=0.03)
    best_2 = pick_best_bird_box(result_2, scale_back=up_scale)

    if best_2 is not None:
        bbox = clamp_bbox(best_2["bbox"], original_image.size)
        bbox = expand_bbox(bbox, original_image.size, expand_ratio=0.12)
        crop = original_image.crop(tuple(bbox))
        return {
            "crop_image": crop,
            "bbox": bbox,
            "confidence": best_2["confidence"],
            "original_image": original_image,
            "detect_mode": "upscaled_x2",
        }

    return None


def get_font(size: int):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    return ImageFont.load_default()


def make_result_canvas(
    original_image: Image.Image,
    bbox: Optional[List[int]],
    detection_confidence: Optional[float],
    detect_mode: Optional[str],
    top_results: Optional[List[Dict]],
    output_path: str,
):
    image_w, image_h = original_image.size
    panel_width = max(500, int(image_w * 0.44))
    canvas = Image.new("RGB", (image_w + panel_width, image_h), PANEL_BG)
    canvas.paste(original_image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(28)
    body_font = get_font(22)
    small_font = get_font(18)

    panel_x = image_w + 20
    current_y = 20

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        line_width = max(4, image_w // 250)
        draw.rectangle([(x1, y1), (x2, y2)], outline=BRIGHT_GREEN, width=line_width)

        detection_text = f"Bird detected ({detection_confidence:.3f})"
        text_box = draw.textbbox((x1, max(5, y1 - 34)), detection_text, font=small_font)
        draw.rectangle(text_box, fill=BRIGHT_GREEN)
        draw.text((text_box[0] + 4, text_box[1] + 2), detection_text, fill=TEXT_BLACK, font=small_font)

        draw.text((panel_x, current_y), "Top-5 Bird Predictions", fill=TEXT_BLACK, font=title_font)
        current_y += 48

        draw.text(
            (panel_x, current_y),
            f"Detection confidence: {detection_confidence * 100:.2f}%",
            fill=TEXT_BLACK,
            font=small_font,
        )
        current_y += 34

        draw.text(
            (panel_x, current_y),
            f"Detection mode: {detect_mode}",
            fill=TEXT_BLACK,
            font=small_font,
        )
        current_y += 34

        draw.text(
            (panel_x, current_y),
            f"Bounding box: [{x1}, {y1}, {x2}, {y2}]",
            fill=TEXT_BLACK,
            font=small_font,
        )
        current_y += 42

        if top_results:
            for rank, item in enumerate(top_results, start=1):
                class_name = item["class_name"]
                probability = item["probability"] * 100.0

                rank_text = f"{rank}."
                prob_text = f"{probability:.2f}%"

                draw.text((panel_x, current_y), rank_text, fill=BRIGHT_GREEN, font=body_font)
                draw.text((panel_x + 40, current_y), class_name, fill=TEXT_BLACK, font=body_font)
                current_y += 30
                draw.text((panel_x + 40, current_y), prob_text, fill=TEXT_BLACK, font=small_font)
                current_y += 36
    else:
        draw.text((panel_x, current_y), "Detection Failed", fill=RED, font=title_font)
        current_y += 50
        draw.text(
            (panel_x, current_y),
            "No bird object was detected by YOLO.",
            fill=TEXT_BLACK,
            font=body_font,
        )
        current_y += 34
        draw.text(
            (panel_x, current_y),
            "Typical causes:",
            fill=TEXT_BLACK,
            font=small_font,
        )
        current_y += 30
        draw.text((panel_x, current_y), "- Bird is too small in the sky", fill=TEXT_BLACK, font=small_font)
        current_y += 26
        draw.text((panel_x, current_y), "- Bird is hidden by dense branches", fill=TEXT_BLACK, font=small_font)
        current_y += 26
        draw.text((panel_x, current_y), "- Bird occupies too few pixels", fill=TEXT_BLACK, font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def build_output_path(output_dir: str, image_path: Path) -> str:
    stem = image_path.stem
    ext = image_path.suffix if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} else ".jpg"
    return os.path.join(output_dir, f"{stem}_detected_top5{ext}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/predict_detect_then_classify.py <config_path> [input_path] [output_path] [top_k] [detector_weights] [checkpoint_path]")
        print("")
        print("Examples:")
        print("  python src/predict_detect_then_classify.py configs/eva02_yiding_mac.yaml")
        print("  python src/predict_detect_then_classify.py configs/eva02_yiding_mac.yaml /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Output_Image")
        print("  python src/predict_detect_then_classify.py configs/eva02_yiding_mac.yaml /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Output_Image 5 yolov8x.pt")
        sys.exit(1)

    config_path = sys.argv[1]
    input_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_INPUT_PATH
    output_dir = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_OUTPUT_PATH
    top_k = int(sys.argv[4]) if len(sys.argv) >= 5 else 5
    detector_weights = sys.argv[5] if len(sys.argv) >= 6 else "yolov8x.pt"
    checkpoint_path_arg = sys.argv[6] if len(sys.argv) >= 7 else None

    config, device, class_names, classifier, checkpoint_path, val_transform = build_classifier(config_path, checkpoint_path_arg)
    detector = YOLO(detector_weights)
    image_files = collect_image_files(input_path)

    print_section_title("YOLO Detect -> EVA Classify -> Save Result Images")
    print(f"Classifier config : {config_path}")
    print(f"Input path        : {input_path}")
    print(f"Output path       : {output_dir}")
    print(f"Detector weights  : {detector_weights}")
    print(f"Classifier model  : {config['model']}")
    print(f"Checkpoint path   : {checkpoint_path}")
    print(f"Using device      : {device}")
    print(f"Total image files : {len(image_files)}")

    success_count = 0
    failed_count = 0

    for idx, image_path in enumerate(image_files, start=1):
        print_section_title(f"Prediction {idx}/{len(image_files)}")
        print(f"Test File Name : {image_path.name}")
        print(f"Full File Path : {image_path}")

        output_path = build_output_path(output_dir, image_path)

        try:
            detect_result = detect_best_bird(detector, image_path)

            if detect_result is None:
                original_image = Image.open(image_path).convert("RGB")
                make_result_canvas(
                    original_image=original_image,
                    bbox=None,
                    detection_confidence=None,
                    detect_mode=None,
                    top_results=None,
                    output_path=output_path,
                )
                print("Detection Result : FAILED")
                print("Reason           : No bird object was detected by YOLO.")
                print(f"Saved Output     : {output_path}")
                failed_count += 1
                continue

            crop_image = detect_result["crop_image"]
            bbox = detect_result["bbox"]
            detection_confidence = detect_result["confidence"]
            original_image = detect_result["original_image"]
            detect_mode = detect_result["detect_mode"]

            top_results = classify_crop(
                model=classifier,
                class_names=class_names,
                val_transform=val_transform,
                device=device,
                crop_image=crop_image,
                top_k=top_k,
            )

            make_result_canvas(
                original_image=original_image,
                bbox=bbox,
                detection_confidence=detection_confidence,
                detect_mode=detect_mode,
                top_results=top_results,
                output_path=output_path,
            )

            print("Detection Result : SUCCESS")
            print(f"Bird BBox        : {bbox}")
            print(f"Bird Confidence  : {detection_confidence:.4f}")
            print(f"Detection Mode   : {detect_mode}")
            print("-" * 100)
            print(f"{'Rank':<6}{'Predicted Class':<55}{'Probability':>15}")
            print("-" * 100)
            for rank, item in enumerate(top_results, start=1):
                print(f"{rank:<6}{item['class_name']:<55}{item['probability'] * 100:>13.2f}%")
            print("-" * 100)
            print(f"Saved Output     : {output_path}")
            success_count += 1

        except Exception as e:
            print("Detection/Classify Result : FAILED")
            print(f"Reason                    : {e}")
            failed_count += 1

    print_section_title("Prediction Summary")
    print(f"Total files   : {len(image_files)}")
    print(f"Success count : {success_count}")
    print(f"Failed count  : {failed_count}")
    print(f"Output folder : {output_dir}")

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
