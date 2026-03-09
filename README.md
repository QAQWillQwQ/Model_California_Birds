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
│       ├── resnet50_train_log.log
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

Install TensorBoard for live monitoring
```bash
pip install tensorboard
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
Small Training Experiment
```bash
python src/train.py configs/resnet50_small.yaml
```
Full Training
```bash
python src/train.py configs/resnet50.yaml
```

## 7. Live Monitoring

Training now writes TensorBoard event files into each run directory when `tensorboard` is installed.

Start training:
```bash
python src/train.py configs/resnet50.yaml
```

The script prints the run-specific TensorBoard directory, for example:
```text
outputs/output_20260308_120000/tensorboard
```

Launch TensorBoard in another terminal:
```bash
tensorboard --logdir outputs/output_20260308_120000/tensorboard
```

Then open the local URL TensorBoard prints in the terminal, usually:
```text
http://localhost:6006
```

If you want to disable TensorBoard for a run, set this in the YAML config:
```yaml
use_tensorboard: false
```

Or use the launcher script to start training and TensorBoard together, with an automatic local port:
```bash
./scripts/run_with_tensorboard.sh configs/vit_base.yaml
```

You can also launch it in interactive config-selection mode:
```bash
./scripts/run_with_tensorboard.sh -i
```

How the launcher works:
- Pass a YAML path to train with that config directly.
- Pass `-i` to print the available YAML files and choose one by number.
- The script starts `src/train.py`, starts TensorBoard on a free local port, and prints the watch URL.
- Terminal output follows the run's debug log, so you can watch training progress live.

Examples:
```bash
./scripts/run_with_tensorboard.sh configs/resnet50.yaml
./scripts/run_with_tensorboard.sh /home/blue-lobster/PycharmProjects/Model_California_Birds/configs/vit_base.yaml
./scripts/run_with_tensorboard.sh -i
```

The script prints the exact local watch URL, for example:
```text
http://127.0.0.1:6006
```
