<div align="center">

# 🩻 DDHSV2

### Swin Transformer V2 for Developmental Dysplasia of the Hip Classification from Ultrasound Images

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Backbone-Swin%20Transformer%20V2-6C63FF?style=for-the-badge" alt="Swin Transformer V2">
</p>

<p>
  <img src="https://img.shields.io/badge/Application-Medical%20Ultrasound-0D5BB9?style=for-the-badge" alt="Medical Ultrasound">
  <img src="https://img.shields.io/badge/Task-DDH%20Classification-27AE60?style=for-the-badge" alt="DDH Classification">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-F39C12?style=for-the-badge" alt="Grad-CAM">
</p>

<p>
  <a href="https://github.com/liwei-bit/DDHSV2/stargazers">
    <img src="https://img.shields.io/github/stars/liwei-bit/DDHSV2?style=for-the-badge&logo=github&color=181717" alt="GitHub Stars">
  </a>
  <a href="https://github.com/liwei-bit/DDHSV2/commits/main">
    <img src="https://img.shields.io/github/last-commit/liwei-bit/DDHSV2?style=for-the-badge&logo=github&color=0D5BB9" alt="Last Commit">
  </a>
</p>

<br>

**DDHSV2** is a Swin Transformer V2-based framework for automated classification of developmental dysplasia of the hip from ultrasound images. It supports binary and multi-class classification, class-imbalance-aware training, comprehensive performance evaluation, and Grad-CAM-based visual interpretation.

</div>

---

## 📖 Overview

Developmental dysplasia of the hip (DDH) assessment from ultrasound images requires accurate characterization of subtle morphological and structural differences. However, automated DDH classification remains challenging because of limited clinical data, class imbalance, inter-class similarity, and variations in ultrasound image quality.

DDHSV2 employs **Swin Transformer V2 Large** to learn hierarchical morphological and texture representations from hip ultrasound images. The framework incorporates extensive image augmentation, weighted sampling, mixed-precision training, differential learning rates, cosine learning-rate scheduling, early stopping, and Grad-CAM visualization.

The overall workflow consists of four stages:

1. Ultrasound images are resized, converted to three channels, normalized, and augmented.
2. Swin Transformer V2 extracts hierarchical morphological and texture features.
3. Global pooled representations are passed to a task-specific classification head.
4. Predictions are evaluated using accuracy, precision, recall, F1-score, AUC, classification reports, and confusion matrices.

---

## ✨ Main Features

- 🩻 **Ultrasound-based DDH classification:** Supports binary and multi-class classification from hip ultrasound images.

- 🧠 **Swin Transformer V2 backbone:** Employs hierarchical window-based self-attention to model local morphology and global structural information.

- ⚖️ **Class-imbalance-aware training:** Uses a weighted random sampler to improve the representation of minority classes during training.

- 🎨 **Comprehensive data augmentation:** Includes random cropping, horizontal and vertical flipping, rotation, color jitter, RandAugment, and random erasing.

- 🚀 **Efficient optimization:** Supports automatic mixed precision, gradient accumulation, differential learning rates, warm-up, cosine decay, and early stopping.

- 📊 **Automatic evaluation:** Saves accuracy, macro F1-score, weighted F1-score, macro precision, macro recall, AUC, classification reports, and confusion matrices.

- 🔥 **Grad-CAM interpretation:** Generates original images, activation heatmaps, overlays, and comparison panels for model interpretation.

---

## 🏗️ Overall Framework

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7f3cd667-15ac-4cc2-b164-79c24c8832e3"
    width="95%"
    alt="Overall framework of DDHSV2"
  />
</p>

<p align="center">
  <em>Overall workflow of the proposed Swin Transformer V2-based DDH classification framework.</em>
</p>

---

## 🎯 Classification Tasks

### Three-Class Classification

The three-class setting distinguishes among the following DDH categories:

| Class | Description |
|---|---|
| `I` | Graf type I |
| `IIabc` | Graf types IIa, IIb, and IIc |
| `D_III_IV` | Graf types D, III, and IV |

### Binary Classification

The default binary setting contains:

| Class | Description |
|---|---|
| `I` | Graf type I |
| `Others` | Other DDH categories |

The training script automatically determines the number and names of classes from the dataset folders. Therefore, other binary grouping strategies can also be used by reorganizing the corresponding class directories.

---

## 🔬 Method

### 1. Image Preprocessing

All ultrasound images are converted to RGB format and processed according to the input configuration of the pretrained Swin Transformer V2 model.

The training augmentation pipeline includes:

- Random resized cropping
- Random horizontal flipping
- Random vertical flipping
- Random rotation
- Brightness, contrast, saturation, and hue adjustment
- RandAugment
- ImageNet normalization
- Random erasing

Validation and test images are resized using bicubic interpolation, center-cropped, converted to tensors, and normalized using ImageNet statistics.

### 2. Feature Extraction

The framework uses the following model from the `timm` library:

```text
swinv2_large_window12to24_192to384.ms_in22k_ft_in1k
```

The network receives images at a resolution of `384 × 384` and extracts hierarchical features through shifted-window self-attention.

### 3. Classification

The original classification head is replaced according to the number of target classes. Global pooled features are used to predict the corresponding DDH category.

### 4. Class-Imbalance Handling

A `WeightedRandomSampler` assigns higher sampling probabilities to minority-class samples. Label smoothing is additionally applied during cross-entropy optimization to improve generalization.

### 5. Model Interpretation

Grad-CAM is applied to the later transformer blocks to identify image regions that contribute most strongly to the predicted DDH category.

---

## 📊 Three-Class Results

The Swin Transformer V2 Large model achieves the following overall performance on the three-class DDH classification task:

| Model | Input Size | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Swin Transformer V2 Large | 384 × 384 | **0.9120** | **0.8925** |

> [!NOTE]
> Performance may vary depending on the dataset split, random seed, model initialization, and training configuration.

---

## 📁 Repository Structure

```text
DDHSV2/
├── train.py
├── test_2cls.py
├── test_3cls.py
├── cam.py
├── config.json
└── README.md
```

| File | Description |
|---|---|
| `train.py` | Model training, validation, testing, and metric saving |
| `test_2cls.py` | Binary DDH prediction and probability export |
| `test_3cls.py` | Three-class DDH prediction and probability export |
| `cam.py` | Grad-CAM generation and visualization |
| `config.json` | Swin Transformer V2 architecture configuration |

---

## ⚙️ Environment

The project is implemented using Python and PyTorch.

### Main Dependencies

| Package | Purpose |
|---|---|
| Python 3.10 | Programming environment |
| PyTorch | Deep-learning framework |
| torchvision | Image transformations and dataset loading |
| timm | Swin Transformer V2 implementation |
| safetensors | Local pretrained-weight loading |
| scikit-learn | Classification metrics |
| NumPy | Numerical computation |
| pandas | Result processing and Excel export |
| Pillow | Image loading |
| tqdm | Progress visualization |
| Matplotlib | Grad-CAM visualization |
| openpyxl | Excel result export |

A CUDA-enabled GPU is recommended for model training and Grad-CAM generation.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/liwei-bit/DDHSV2.git
cd DDHSV2
```

### 2. Create a Conda environment

```bash
conda create -n ddhsv2 python=3.10 -y
conda activate ddhsv2
```

### 3. Install dependencies

```bash
pip install torch torchvision
pip install timm safetensors scikit-learn numpy pandas pillow tqdm matplotlib openpyxl
```

---

## 🗂️ Dataset Preparation

The project uses the standard `ImageFolder` directory structure.

### Three-Class Dataset

```text
DDH_grouped/
└── DDH_3cls/
    ├── train/
    │   ├── D_III_IV/
    │   ├── I/
    │   └── IIabc/
    ├── val/
    │   ├── D_III_IV/
    │   ├── I/
    │   └── IIabc/
    └── test/
        ├── D_III_IV/
        ├── I/
        └── IIabc/
```

### Binary Dataset

```text
DDH_grouped/
└── DDH_2cls/
    ├── train/
    │   ├── I/
    │   └── Others/
    ├── val/
    │   ├── I/
    │   └── Others/
    └── test/
        ├── I/
        └── Others/
```

> [!IMPORTANT]
> The class folder names and class ordering must remain consistent across the training, validation, and test sets.

---

## 🚀 Model Training

Before training, download a compatible Swin Transformer V2 Large checkpoint and provide its local path using `--swin_ckpt`.

### Three-Class Training

```bash
python train.py \
  --data_root ./DDH_grouped/DDH_3cls \
  --output_root ./runs_transformer_DDH_3class \
  --swin_ckpt ./pretrained/swinv2_large/model.safetensors \
  --model_id swinv2_large_window12to24_192to384.ms_in22k_ft_in1k \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 2e-5 \
  --head_lr_mult 10.0 \
  --weight_decay 5e-2 \
  --epochs 30 \
  --patience 15 \
  --num_workers 8 \
  --seed 42
```

### Binary Training

```bash
python train.py \
  --data_root ./DDH_grouped/DDH_2cls \
  --output_root ./runs_transformer_DDH_2class \
  --swin_ckpt ./pretrained/swinv2_large/model.safetensors \
  --model_id swinv2_large_window12to24_192to384.ms_in22k_ft_in1k \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --lr 2e-5 \
  --head_lr_mult 10.0 \
  --weight_decay 5e-2 \
  --epochs 30 \
  --patience 15 \
  --num_workers 8 \
  --seed 42
```

If GPU memory is insufficient, reduce `--batch_size` and increase `--grad_accum_steps`.

---

## 🧪 Model Evaluation

### Three-Class Prediction

```bash
python test_3cls.py \
  --test_root ./DDH_grouped/DDH_3cls/test \
  --train_root ./DDH_grouped/DDH_3cls/train \
  --model_path ./runs_transformer_DDH_3class/swinv2_large/best.pth \
  --output_json ./runs_transformer_DDH_3class/swinv2_large/ddh_3cls_test_predictions.json \
  --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k \
  --img_size 384 \
  --num_classes 3 \
  --class_names D_III_IV I IIabc
```

### Binary Prediction

```bash
python test_2cls.py \
  --test_root ./DDH_grouped/DDH_2cls/test \
  --model_path ./runs_transformer_DDH_2class/swinv2_large/best.pth \
  --output_json ./runs_transformer_DDH_2class/swinv2_large/ddh_2cls_test_predictions.json \
  --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k \
  --img_size 384 \
  --num_classes 2 \
  --class_names I Others
```

The prediction scripts save the predicted category, confidence score, and probability for every class in JSON format.

---

## 🔥 Grad-CAM Visualization

### Three-Class Grad-CAM

The following command generates Grad-CAM results for the predicted category and the `IIabc` category:

```bash
python cam.py \
  --num_classes 3 \
  --test_root ./DDH_grouped/DDH_3cls/test \
  --train_root ./DDH_grouped/DDH_3cls/train \
  --model_path ./runs_transformer_DDH_3class/swinv2_large/best.pth \
  --output_dir ./ddh_3cls_cam \
  --class_names D_III_IV I IIabc \
  --cam_target both \
  --target_class IIabc \
  --target_layer auto \
  --max_per_class 10 \
  --max_images 30
```

### Binary Grad-CAM

```bash
python cam.py \
  --num_classes 2 \
  --test_root ./DDH_grouped/DDH_2cls/test \
  --train_root ./DDH_grouped/DDH_2cls/train \
  --model_path ./runs_transformer_DDH_2class/swinv2_large/best.pth \
  --output_dir ./ddh_2cls_cam \
  --class_names I Others \
  --cam_target both \
  --target_class Others \
  --target_layer auto \
  --max_per_class 10 \
  --max_images 20
```

### List Candidate Target Layers

```bash
python cam.py \
  --num_classes 3 \
  --test_root ./DDH_grouped/DDH_3cls/test \
  --train_root ./DDH_grouped/DDH_3cls/train \
  --model_path ./runs_transformer_DDH_3class/swinv2_large/best.pth \
  --class_names D_III_IV I IIabc \
  --list_layers
```

For each selected image, the Grad-CAM script saves:

- Original image
- Activation heatmap
- Heatmap overlay
- Comparison panel
- Prediction confidence
- Target-class activation score

It also generates `cam_summary.json` and `cam_summary.xlsx`.

---

## 💾 Training Outputs

The training results are organized as follows:

```text
runs_transformer_DDH_3class/
├── all_results.json
├── all_results_sorted.json
├── leaderboard.csv
└── swinv2_large/
    ├── best.pth
    ├── model_config.json
    ├── train_class_count.json
    ├── history.json
    ├── best_val_metrics.json
    ├── test_metrics.json
    ├── test_classification_report.csv
    ├── test_confusion_matrix.csv
    └── summary.json
```

The best checkpoint is selected according to validation accuracy.

---

## 📈 Evaluation Metrics

The training and evaluation pipeline reports:

- 🎯 Accuracy
- ⚖️ Macro F1-score
- 📊 Weighted F1-score
- 🔬 Macro precision
- 📉 Macro recall
- 📈 Binary or one-vs-rest macro AUC
- 🧩 Confusion matrix
- 📋 Class-wise classification report

---

## ♻️ Reproducibility

To improve experimental reproducibility:

- Use the same training, validation, and test split.
- Keep the class folder names consistent.
- Use the same pretrained checkpoint.
- Fix the random seed.
- Report both macro and weighted metrics.
- Evaluate the final model using the checkpoint selected on the validation set.
- Avoid using test-set information during model selection.

> [!NOTE]
> The current implementation uses seed `42` by default and automatically saves the complete training history and evaluation results.

---

## 🔐 Data Availability

Clinical ultrasound images are not included in this repository. Users should organize their authorized data according to the directory structure described above and ensure compliance with applicable institutional, ethical, and privacy requirements.

---

## 📝 Citation

If this repository is useful for your research, please cite the corresponding work. Complete citation information will be added after publication.

---

## 🙏 Acknowledgements

This project uses PyTorch, `timm`, scikit-learn, and other open-source libraries. We thank their developers and maintainers for supporting reproducible medical image analysis research.

---

## 📬 Contact

If you have questions about the code or experimental settings, please open an issue in this repository.

<div align="center">

<br>

⭐ **If you find this project useful, please consider giving it a star.**

<br>

<img src="https://img.shields.io/badge/Research-Medical%20Image%20Analysis-0D5BB9?style=flat-square" alt="Medical Image Analysis">
<img src="https://img.shields.io/badge/Model-Swin%20Transformer%20V2-6C63FF?style=flat-square" alt="Swin Transformer V2">
<img src="https://img.shields.io/badge/Task-DDH%20Classification-27AE60?style=flat-square" alt="DDH Classification">

</div>
