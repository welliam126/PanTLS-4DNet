# PanTLS-4DNet

### General framework

​      We present PanTLS-4DNet, an innovative deep learning framework employing a dynamic soft-voting ensemble approach for automated prediction of TLS spatial immunophenotypes using contrast-enhanced pancreatic CT imaging . The model architecture comprises four sequentially integrated processing modules designed for comprehensive end-to-end analysis from raw image input to final immunophenotype classification.

The initial segmentation module leverages an optimized nnUNet variant to achieve accurate three-dimensional delineation of pancreatic and tumor volumes through adaptive threshold-based mask generation. Subsequent feature extraction integrates a multimodal CLIP framework that synergistically processes both radiological images and corresponding clinical reports to derive enhanced radiomic signatures via contrastive learning paradigms. Model robustness is ensured through parallel implementation of four heterogeneous convolutional neural networks (FBViT, VGG-GAP, ResNet3D, and GlobalMLP) to capture complementary feature representations. The final classification stage employs an adaptive ensemble strategy where model-specific weights are dynamically optimized based on F1-score performance metrics from validation datasets18. This hierarchical architecture facilitates superior interpretability while achieving enhanced predictive accuracy through synergistic fusion of multidimensional feature spaces.

![image-20250805230921108](C:\Users\52978\AppData\Roaming\Typora\typora-user-images\image-20250805230921108.png)

### segmentation model Description

This repository used a **complete copy of the official nnU-Net v2 code** (original source: https://github.com/MIC-DKFZ/nnUNet). The file structure is explained below:

| File/Directory   | Purpose                                                      |
| :--------------- | :----------------------------------------------------------- |
| `nnunetv2/`      | **Official core code**: Model architecture and training/inference pipeline (Copyright © MIC-DKFZ) |
| `.gitignore`     | Repository configuration: Excludes temporary files (`__pycache__`) and sensitive data |
| `LICENSE`        | **Original license**: Apache 2.0 (as per official repository) |
| `pyproject.toml` | Dependency configuration: Declares required Python environment |
| `setup.py`       | Installation script: Supports deployment via `pip install .` |
| `README.md`      | Project documentation (this file)                            |
| `run_nnunet.sh`  | **Reference script**: Example usage of official commands (non-official component) |
| `tst.sh`         | Test script: Environment verification (non-official component) |

#### Important Disclaimers

1. **Code Ownership**
   All code in the `nnunetv2/` directory is the original work of the [German Cancer Research Center (MIC-DKFZ)](https://www.dkfz.de/en/mic/index.php). This repository only hosts an unmodified copy.
2. **Modification Record**
   No changes were made to the core code. Only peripheral configurations (e.g., sample scripts, `.gitignore`) were adjusted.
3. **Official Documentation**
   Full documentation: [nnU-Net Official Docs](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation.md)

#### Classification model Description

This repository used a **complete copy of the official CT_CLIP code** (original source:https://github.com/ibrahimethemhamamci/CT-CLIP/tree/main/CT_CLIP

### Requirements

PyTorch: 2.1.1
CUDA Toolkit: 11.8
cuDNN: 8.6.0
transformers: 4.30.0
scikit-learn: 1.0.0
torchvision: 0.16.1
Pillow: 8.0.0
opencv-python: 4.5.0
torch==2.1.1+cu118
torchvision==0.16.1+cu118
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.64.0
nibabel>=4.0.0
