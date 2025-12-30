# Comparative Analysis of ResNet-50 and EfficientNet-B0 for Melanoma Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🇺🇸 English](README_EN.md) | [🇧🇷 Português](README.md)

Complete deep learning system for binary classification of skin lesions (benign vs malignant) using the ISIC 2020 dataset, with detailed comparison between ResNet-50 and EfficientNet-B0.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## 🔧 Requirements

- Python 3.8 or higher
- CUDA (optional, but recommended for training)
- 8GB+ RAM
- Disk space: ~5GB (data + models)

## 📦 Installation

### 1. Clone the repository or navigate to the project directory

```bash
# If you cloned the repository:
git clone <repository-url>
cd melanoma-detection

# Or if you already have the project, navigate to the directory:
cd path/to/melanoma-detection
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** If `pip` is not available, use `pip3`:

```bash
pip3 install -r requirements.txt
```

## 📁 Data Preparation

### Directory Structure

The project expects data organized as follows:

```
data/
  isic2020/
    benign/
      image1.jpg
      image2.jpg
      ...
    malignant/
      image1.jpg
      image2.jpg
      ...
```

### Options to Obtain Data

1. **ISIC 2020 Dataset** (official):
   - Visit: https://www.isic-archive.com/
   - Register and download the dataset
   - Organize images into `benign/` and `malignant/` folders

2. **Alternative Kaggle Dataset**:
   - Search for "Skin Cancer: Malignant vs Benign"
   - Download and organize in the structure above

3. **Small Test Dataset**:
   - For quick tests, you can use a smaller subset
   - Maintain the same folder structure

## 🚀 Usage

### 1. Train ResNet-50

```bash
python experiments/train.py --config config/resnet50_config.yaml
```

**What happens:**
- Loads and splits the dataset (70% train, 15% validation, 15% test)
- Trains ResNet-50 model with early stopping
- Saves checkpoints in `./checkpoints/resnet50/`
- Generates TensorBoard logs in `./runs/resnet50/`

### 2. Train EfficientNet-B0

First, create a configuration file for EfficientNet:

```bash
cp config/resnet50_config.yaml config/efficientnet_config.yaml
```

Edit `config/efficientnet_config.yaml` and change:
```yaml
model:
  name: efficientnet_b0
```

Then, train:

```bash
python experiments/train.py --config config/efficientnet_config.yaml
```

### 3. Compare Models

After training both models, run the complete comparison:

```bash
python experiments/compare.py
```

**What is generated:**
- Comparative clinical metrics (console)
- ROC curves (`results/roc_comparison.png`)
- Metrics bar chart (`results/metrics_comparison.png`)
- Statistical analysis (McNemar, confidence intervals)
- Error analysis (false positives/negatives)
- Computational efficiency benchmark

### 4. Explainability Analysis (Grad-CAM)

Generates visual attention maps to compare how models "see" images:

```bash
python experiments/analyze_explainability.py \
    --num_samples 100 \
    --resnet_checkpoint ./checkpoints/resnet50/best_model.pth \
    --effnet_checkpoint ./checkpoints/efficientnet_b0/best_model.pth \
    --save_dir ./results/explainability
```

**Results:**
- Images with Grad-CAM overlay
- Interactive HTML page (`results/explainability/index.html`)

## 📊 Viewing Results

### TensorBoard

To visualize training metrics in real-time:

```bash
tensorboard --logdir ./runs
```

Access: http://localhost:6006

### Generated Plots

Plots are saved in `./results/`:
- `roc_comparison.png` - Comparative ROC curves
- `metrics_comparison.png` - Clinical metrics comparison

### Explainability HTML

Open in browser:
```
./results/explainability/index.html
```

## 📂 Project Structure

```
melanoma-detection/
├── config/                 # YAML configuration files
│   └── resnet50_config.yaml
├── data/                   # Dataset and processing
│   ├── dataset.py          # Dataset loading
│   ├── preprocessing.py    # Transformations and augmentations
│   └── isic2020/          # Data (benign/, malignant/)
├── models/                 # Model architectures
│   ├── resnet.py
│   ├── efficientnet.py
│   └── model_factory.py
├── training/               # Training system
│   └── trainer.py
├── evaluation/             # Metrics and benchmarks
│   ├── metrics.py          # Clinical metrics
│   └── efficiency.py      # Computational benchmark
├── explainability/         # Grad-CAM
│   └── gradcam.py
├── experiments/            # Main scripts
│   ├── train.py            # Training
│   ├── compare.py         # Complete comparison
│   └── analyze_explainability.py
├── utils/                  # Utilities
│   ├── config.py           # Configuration management
│   └── reproducibility.py  # Seed and device
├── checkpoints/            # Trained models (generated)
├── results/                # Results and visualizations (generated)
├── runs/                   # TensorBoard logs (generated)
└── requirements.txt       # Dependencies
```

## 📈 Calculated Metrics

The system calculates the following metrics:

- **Accuracy**: Rate of correct predictions
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Precision**: Rate of correct positive predictions
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Cohen's Kappa**: Agreement between predictions and labels
- **MCC**: Matthews Correlation Coefficient

## 🔬 Statistical Analysis

The comparison includes:

- **Confidence Intervals (95%)**: For differences between models
- **McNemar Test**: Paired model comparison
- **Error Analysis**: False positives/negatives and disagreements
- **Efficiency Benchmark**: FLOPs, latency, memory, size

## ⚙️ Configuration

Edit `config/resnet50_config.yaml` to adjust:

- **Model**: `resnet50` or `efficientnet_b0`
- **Batch size**: Batch size (default: 32)
- **Learning rate**: Learning rate (default: 0.0001)
- **Epochs**: Maximum number of epochs (default: 50)
- **Early stopping**: Patience for early stopping (default: 10)
- **Augmentations**: Rotation, flip, brightness, contrast, zoom

## 🐛 Troubleshooting

### Error: "pip: command not found"

```bash
# Use pip3
pip3 install -r requirements.txt

# Or install Python via Homebrew (macOS)
brew install python3
```

### Error: "CUDA out of memory"

- Reduce `batch_size` in the configuration file
- Use `device: cpu` if you don't have a GPU

### Error: "Dataset not found"

- Check if data is in `./data/isic2020/`
- Confirm structure: `benign/` and `malignant/` inside `isic2020/`

### Checkpoints not found

- Run training first (`experiments/train.py`)
- Checkpoints are automatically saved in `./checkpoints/`

## 📝 Complete Execution Example

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Train ResNet-50
python experiments/train.py --config config/resnet50_config.yaml

# 3. Train EfficientNet-B0 (after creating config)
python experiments/train.py --config config/efficientnet_config.yaml

# 4. Compare models
python experiments/compare.py

# 5. Explainability analysis
python experiments/analyze_explainability.py --num_samples 50

# 6. View in TensorBoard
tensorboard --logdir ./runs
```

## 📄 License

This project is open source and licensed under the [MIT License](LICENSE).

## 👥 Authors

**Lucas Felipe Cassol Seixas** - [@LSeixas98](https://github.com/LSeixas98)

Project developed for comparative analysis of deep learning models in melanoma detection.

## 📚 References

- ISIC 2020 Challenge: https://www.isic-archive.com/
- ResNet: He et al. (2016) - Deep Residual Learning
- EfficientNet: Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
- Grad-CAM: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations

