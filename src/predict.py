import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import datasets

from dataset import build_transforms
from models import build_model
from utils import load_config, get_device, load_checkpoint

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
}

DEFAULT_INPUT_DIR = "/Users/wangyiding/ML_Model_CaliforniaBirds/Test_Input_Image"

DEFAULT_CHECKPOINT_CANDIDATES = [
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_best_full.pth",
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_best.pth",
    "/Users/wangyiding/ML_Model_CaliforniaBirds/outputs/checkpoints/JiHang_EVA_CheckPoints_20260309_233227/eva02_base_patch14_448.mim_in22k_ft_in1k_last.pth",
]


def print_separator():
    print("=" * 90)


def print_section_title(title: str):
    print_separator()
    print(title)
    print_separator()


def is_supported_image_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def collect_image_files(input_path: str) -> List[Path]:
    path_obj = Path(input_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path_obj.is_file():
        if not is_supported_image_file(path_obj):
            raise ValueError(
                f"Unsupported image file format: {path_obj.name}\n"
                f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        return [path_obj]

    image_files = []
    for item in path_obj.rglob("*"):
        if is_supported_image_file(item):
            image_files.append(item)

    image_files.sort(key=lambda p: str(p).lower())
    return image_files


def find_checkpoint_path(config: dict) -> str:
    candidate_paths = []

    resume_from = config.get("resume_from", None)
    if resume_from:
        candidate_paths.append(resume_from)

    candidate_paths.append(
        os.path.join(
            config["output_dir"],
            "checkpoints",
            f'{config["model"]}_best_full.pth'
        )
    )
    candidate_paths.append(
        os.path.join(
            config["output_dir"],
            "checkpoints",
            f'{config["model"]}_best.pth'
        )
    )
    candidate_paths.append(
        os.path.join(
            config["output_dir"],
            "checkpoints",
            f'{config["model"]}_last.pth'
        )
    )

    candidate_paths.extend(DEFAULT_CHECKPOINT_CANDIDATES)

    for checkpoint_path in candidate_paths:
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path

    raise FileNotFoundError(
        "Checkpoint not found.\nChecked paths:\n" + "\n".join(candidate_paths)
    )


@torch.no_grad()
def predict_one_image(
        model,
        image_path: Path,
        class_names: List[str],
        model_name: str,
        image_size: int,
        device,
        top_k: int,
):
    _, val_transform = build_transforms(
        model_name=model_name,
        pretrained=False,
        image_size=image_size,
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    actual_top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=actual_top_k, dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    print(f"Test File Name : {image_path.name}")
    print(f"Full File Path : {image_path}")
    print("-" * 90)
    print("Top Predictions:")
    for rank, (class_idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        class_name = class_names[class_idx]
        print(f"{rank:>2}. {class_name:<40} {prob:.4f} ({prob * 100:.2f}%)")
    print()


def build_inference_model(config_path: str, checkpoint_path_arg: str = None):
    config = load_config(config_path)
    device = get_device()

    dataset = datasets.ImageFolder(root=config["data_root"])
    class_names = dataset.classes
    num_classes = len(class_names)

    model = build_model(
        model_name=config["model"],
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    if checkpoint_path_arg is not None:
        checkpoint_path = checkpoint_path_arg
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    else:
        checkpoint_path = find_checkpoint_path(config)

    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    return config, device, class_names, model, checkpoint_path


def predict_images(config_path: str, input_path: str, top_k: int = 5, checkpoint_path: str = None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config, device, class_names, model, checkpoint_path = build_inference_model(
        config_path=config_path,
        checkpoint_path_arg=checkpoint_path,
    )

    print_section_title("California Birds Prediction")
    print(f"Using device      : {device}")
    print(f"Config path       : {config_path}")
    print(f"Input path        : {input_path}")
    print(f"Model             : {config['model']}")
    print(f"Checkpoint path   : {checkpoint_path}")
    print(f"Number of classes : {len(class_names)}")
    print(f"Top-K             : {top_k}")
    print()

    image_files = collect_image_files(input_path)

    if len(image_files) == 0:
        print_section_title("No Valid Image Found")
        print("No supported image files were found in the given input path.")
        print(f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}")
        return

    print(f"Total image files found: {len(image_files)}")
    print()

    success_count = 0
    failed_count = 0

    for idx, image_path in enumerate(image_files, start=1):
        print_section_title(f"Prediction {idx}/{len(image_files)}")
        try:
            predict_one_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                model_name=config["model"],
                image_size=config["image_size"],
                device=device,
                top_k=top_k,
            )
            success_count += 1
        except UnidentifiedImageError:
            print(f"Test File Name : {image_path.name}")
            print(f"Full File Path : {image_path}")
            print("-" * 90)
            print("Result         : FAILED")
            print("Reason         : The file exists, but Pillow cannot identify it as a valid image.")
            print()
            failed_count += 1
        except Exception as e:
            print(f"Test File Name : {image_path.name}")
            print(f"Full File Path : {image_path}")
            print("-" * 90)
            print("Result         : FAILED")
            print(f"Reason         : {e}")
            print()
            failed_count += 1

    print_section_title("Prediction Summary")
    print(f"Total files   : {len(image_files)}")
    print(f"Success count : {success_count}")
    print(f"Failed count  : {failed_count}")


def main():
    """
    Usage:
      1) Test the default folder:
         python src/predict.py configs/resnet50.yaml

      2) Test one image:
         python src/predict.py configs/resnet50.yaml /path/to/image.jpg

      3) Test one folder:
         python src/predict.py configs/resnet50.yaml /path/to/folder

      4) Specify top_k:
         python src/predict.py configs/resnet50.yaml /path/to/folder 3

      5) Specify checkpoint path:
         python src/predict.py configs/resnet50.yaml /path/to/folder 5 /path/to/best_full.pth
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/predict.py <config_path> [input_path] [top_k]")
        print("")
        print("Examples:")
        print("  python src/predict.py configs/resnet50.yaml")
        print(f"  python src/predict.py configs/resnet50.yaml {DEFAULT_INPUT_DIR}")
        print("  python src/predict.py configs/resnet50.yaml /Users/wangyiding/Desktop/bird.jpg")
        print("  python src/predict.py configs/resnet50.yaml /Users/wangyiding/Desktop 3")
        sys.exit(1)

    config_path = sys.argv[1]

    if len(sys.argv) >= 3:
        input_path = sys.argv[2]
    else:
        input_path = DEFAULT_INPUT_DIR

    if len(sys.argv) >= 4:
        try:
            top_k = int(sys.argv[3])
        except ValueError:
            print("top_k must be an integer.")
            sys.exit(1)
    else:
        top_k = 5

    if len(sys.argv) >= 5:
        checkpoint_path = sys.argv[4]
    else:
        checkpoint_path = None

    try:
        predict_images(
            config_path=config_path,
            input_path=input_path,
            top_k=top_k,
            checkpoint_path=checkpoint_path,
        )
    except Exception as e:
        print_section_title("Prediction Error")
        print(f"Reason: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
