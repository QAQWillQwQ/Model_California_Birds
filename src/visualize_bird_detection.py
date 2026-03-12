import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def detect_best_bird(image_path: str, detector_weights: str = "yolov8x.pt"):
    model = YOLO(detector_weights)
    results = model(image_path, verbose=False)
    result = results[0]

    names = result.names
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    bird_candidates = []
    for idx in range(len(boxes)):
        cls_id = int(boxes.cls[idx].item())
        cls_name = names[cls_id]
        conf = float(boxes.conf[idx].item())
        xyxy = boxes.xyxy[idx].cpu().numpy().tolist()

        if cls_name.lower() == "bird":
            bird_candidates.append((conf, xyxy))

    if not bird_candidates:
        return None, None

    bird_candidates.sort(key=lambda x: x[0], reverse=True)
    best_conf, best_xyxy = bird_candidates[0]
    return best_xyxy, best_conf


def draw_and_save_detection(image_path: str, bbox, confidence: float, save_path: str):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)

        label_text = f"bird {confidence:.3f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 30)), label_text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, max(0, y1 - 30)), label_text, fill="white", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return save_path


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/visualize_bird_detection.py <image_path> [detector_weights]")
        print("")
        print("Example:")
        print("  python src/visualize_bird_detection.py /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image/Test_Real_Unkown.jpg")
        print("  python src/visualize_bird_detection.py /Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image/Test_Real_Unkown.jpg yolov8x.pt")
        sys.exit(1)

    image_path = sys.argv[1]
    detector_weights = sys.argv[2] if len(sys.argv) >= 3 else "yolov8x.pt"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bbox, confidence = detect_best_bird(image_path, detector_weights=detector_weights)

    file_name = Path(image_path).stem
    save_path = f"/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/visualizations/{file_name}_detected.jpg"

    print("\n" + "=" * 90)
    print("Bird Detection Visualization")
    print("=" * 90)
    print(f"Input Image      : {image_path}")
    print(f"Detector Weights : {detector_weights}")
    print(f"Save Output Path : {save_path}")

    if bbox is None:
        print("Detection Result : FAILED")
        print("Reason           : No bird object detected.")
        output_path = draw_and_save_detection(
            image_path=image_path,
            bbox=None,
            confidence=0.0,
            save_path=save_path,
        )
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        print("Detection Result : SUCCESS")
        print(f"Bird Confidence  : {confidence:.4f}")
        print(f"Bird BBox        : [{x1}, {y1}, {x2}, {y2}]")

        output_path = draw_and_save_detection(
            image_path=image_path,
            bbox=bbox,
            confidence=confidence,
            save_path=save_path,
        )

    print(f"Saved Visualization: {output_path}")

    if sys.platform == "darwin":
        os.system(f'open "{output_path}"')


if __name__ == "__main__":
    main()