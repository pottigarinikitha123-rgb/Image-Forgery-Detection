# AI-Powered Image Forgery Detection Using Deep Vision Models

Classifying images as **Deepfake**, **Real**, or **Tempered** using two deep vision architectures — EfficientNet-B0 (CNN) and ViT-Base (Vision Transformer) — with Grad-CAM heatmaps for explainability.

As AI-generated and digitally manipulated images become harder to spot, reliable forgery detection matters. This project trains and compares a convolutional model and a transformer model on the same three-class task, then visualizes *where* the network focuses when making a decision.

## Results

| Model | Architecture | Test Accuracy |
|-------|--------------|--------------|
| EfficientNet-B0 | CNN (compound scaling) | 94.22% |
| ViT-Base | Vision Transformer (16×16 patches) | 96.34% |

The Vision Transformer's global self-attention gave it an edge on capturing manipulation cues spread across the image.

## How it works

The pipeline has four parts:

1. **Dataset ingestion & preprocessing** — images resized to 224×224, converted to tensors, and normalized. Data is split 80/20 into training and test sets.
2. **EfficientNet-B0 training** — pretrained CNN with a 3-class head, trained with Adam (lr = 1e-4) and cross-entropy loss.
3. **ViT-Base training** — pretrained Vision Transformer, fine-tuned with Adam (lr = 2e-5).
4. **Grad-CAM explainability** — generates heatmaps over input images to highlight the regions that drove each prediction.

## Dataset

The [CDFFAKE V2 dataset](https://www.kaggle.com/datasets/fari4117/cdffake-v2-dataset) (downloaded via `kagglehub`) with three labeled categories:

- **Deepfake** — artificially generated faces
- **Real** — unaltered, authentic images
- **Tempered** — edited or digitally manipulated photographs

## Tech stack

- **PyTorch** — model training and inference
- **timm** — pretrained EfficientNet-B0 and ViT-Base models
- **pytorch-grad-cam** — explainability heatmaps
- **scikit-learn** — accuracy and classification metrics
- **kagglehub** — dataset download
- **matplotlib, NumPy, Pillow** — visualization and image handling

## Getting started

```bash
# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook DeepL.ipynb
```

Run the cells in order. The notebook downloads the dataset, trains both models, prints classification reports, and displays a Grad-CAM heatmap. Trained weights are saved to `cnn_forgery_model.pth` and `vit_forgery_model.pth`.

> **Note:** A CUDA-enabled GPU is recommended for training. The notebook was originally run in Google Colab.

## Training settings

| Setting | EfficientNet-B0 | ViT-Base |
|---------|-----------------|----------|
| Learning rate | 1e-4 | 2e-5 |
| Epochs | 5 | 3 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Loss | Cross-entropy | Cross-entropy |

## Author

**Nikhitha Pottigari** — Florida Atlantic University, Boca Raton, FL

This project accompanies the research paper *AI-Powered Image Forgery Detection Using Deep Vision Models.*
