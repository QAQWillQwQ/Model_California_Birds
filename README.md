# California Birds Image Classification

This project trains deep learning models to recognize bird species from images using PyTorch.
The dataset contains hundreds of bird classes, and the training pipeline supports:
	•	ResNet / EfficientNet / ViT
	•	Stratified sampling
	•	Top-1 / Top-5 accuracy
	•	Config-driven experiments
	•	Automatic logging of experiment results

The goal is to build a reproducible training pipeline that can easily be used.

# Project Structure

```text
ML_Model_CaliforniaBirds
│
├── configs
│   ├── efficientnet_b0.yaml
│   ├── local_debug.yaml
│   ├── resnet50.yaml
│   ├── resnet50_small.yaml
│   └── vit_base.yaml
│
├── data
│   ├── species_1
│   ├── species_2
│   └── ...
│
├── notebooks
│   ├── data_check.ipynb
│   └── colab_train_demo.ipynb
│
├── outputs
│   ├── checkpoints
│   │   ├── resnet50_best.pth
│   │   └── resnet50_last.pth
│   │
│   └── logs
│       ├── resnet50_train_log.txt
│       └── experiment_results.csv
│
├── src
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
└── README.md
```

## 1. Environment Setup

Create a Python environment and install dependencies.

Install PyTorch (CPU/MPS version):
```bash
pip install torch torchvision
```

Install other dependencies
```bash
pip install pyyaml pillow matplotlib pandas
```

## 2. Verify Installation

Run these commands to confirm the environment works.
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "from PIL import Image; print('PIL OK')"
python -c "import yaml; print('YAML OK')"
```

Expected output example:
2.x.x
0.x.x
PIL OK
YAML OK

## 3. Prepare Dataset

Place the dataset into the data directory.
Each bird species must be stored in its own folder.

Example layout:
```text
data/
  Species_1/
    image1.jpg
    image2.jpg
  Species_2/
    image1.jpg
    image2.jpg
```

Each folder name becomes the class label.

## 4. Quick Dataset Check

You can visually inspect the dataset using the notebook:
`notebooks/data_check.ipynb`

This notebook verifies:
	•	dataset loading
	•	class count
	•	images per class
	•	random image visualization
	•	transform correctness
	•	dataloader output shapes

Run all cells from top to bottom.

## 5. Configuration System

All experiments are controlled by YAML config files located in the configs folder.

Example config:
`configs/resnet50_small.yaml`

## 6. Run Training

The main training script is:
`src/train.py`

Run training using a config file.

Quick debug run
Small dataset for fast testing.
```bash
python src/train.py configs/local_debug.yaml
```
