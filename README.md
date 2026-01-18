# DINO-AugSeg

**DINO-AugSeg** — *Exploiting DINOv3-Based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation*

📄 **Paper**:  
[Exploiting DINOv3-Based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation](https://www.arxiv.org/pdf/2601.08078)

Hugging Face: https://huggingface.co/papers/2601.08078
Model: https://huggingface.co/appletree1900/DINO-AugSeg/tree/main
Dataset: https://huggingface.co/datasets/appletree1900/ACDC/tree/main
---

## 📌 Overview

This repository provides the official implementation of **DINO-AugSeg**, a few-shot medical image segmentation framework that leverages pretrained **DINOv3** self-supervised visual representations. By integrating feature-level augmentation and contextual fusion, DINO-AugSeg achieves strong performance under limited annotation settings.

The framework introduces two key components:

- **WT-Aug**: A wavelet-based feature augmentation module to enhance feature diversity.
- **CG-Fuse**: A contextual guidance fusion module that effectively integrates semantic and spatial information.

DINO-AugSeg is extensively evaluated on multiple public medical imaging benchmarks spanning **MRI, CT, ultrasound, endoscopy, and dermoscopy**, demonstrating robust cross-modality generalization in few-shot scenarios.

---

## ✨ Features

- Few-shot medical image segmentation with self-supervised **DINOv3** features  
- Feature-level augmentation using **wavelet transforms (WT-Aug)**  
- Multi-scale contextual fusion via **CG-Fuse**  
- Supports multiple imaging modalities and lesion types  
- Modular, readable training and evaluation pipelines  

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/apple1986/DINO-AugSeg.git
cd DINO-AugSeg
```

### 2. Download Required Data and Models
Download the **processed ACDC dataset** and **pretrained models** from:

🔗 https://drive.google.com/file/d/1QKSINiKGvRRzq2dw2q85HSQ6uZDOmh28/view?usp=sharing

- Unzip the downloaded file.
- Place the model file dinov3_convnext_large_pretrain_lvd.pth in checkpoint/dino_ori
- Place the model file acdc_model_last_epoch.pth in checkpoint/checkpoint_acdc/dino_augseg/cross_guide_wt_unet_7
- Place the ACDC dataset in dataset/ACDC
- Please refer the Organize Files below:

### 3. Organize Files
```text
DINO-AugSeg/
├── dataset/
│   └── ACDC/
├── checkpoint/
│   ├── dino_ori/
│   └── checkpoint_acdc/
│       └── dino_augseg/
│           └── cross_guide_wt_unet_7/
```

### 4. Modify Paths
Update the root paths in the following files to match your local environment:

```python
root_path = "/your/absolute/path"
REPO_DIR = os.path.join(root_path, "DINO-AugSeg")
```

### 5. Check the Model
```bash
python model/Dinov3_WTAUG_UNet.py
```

### 6. Testing
```bash
python test_dino_augseg_acdc.py
```

### 7. Training
```bash
python train_dino_augseg_acdc.py
```

---

## 📄 Citation

```bibtex
@article{xu2026dinoaugseg,
  title={Exploiting DINOv3-Based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation},
  author={Xu, Guoping and Udupa, Jayaram K and Lu, Weiguo and Zhang, You},
  year={2026},
  note={arXiv preprint}
}
```

---

## Acknowledgment
This work builds upon recent advances in self-supervised vision models, particularly DINOv3 (https://github.com/facebookresearch/dinov3
), and the growing body of research on foundation models for medical image analysis. We gratefully acknowledge the developers of the open-source codebases and pretrained models, as well as the contributors of the public datasets used in this study.

## 📜 License

This project is released under the **Apache-2.0 License**.
