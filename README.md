# Self-Supervised Representation Learning for Survival Prediction from Multichannel Microscopy Images of Cancer Tissue 

> 📄 The full thesis can be read [here](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-593914).

## Overview

This project investigates whether multichannel multiplexed immunofluorescence (mIF) microscopy images can be used to predict survival outcomes in non-small cell lung cancer (NSCLC) patients. The focus is on capturing patterns within the tumor microenvironment (TME) — not only at the single-cell level, but also across the broader spatial context of the tissue.

Rather than relying primarily on cell segmentation and handcrafted single-cell features, this project evaluates state-of-the-art self-supervised learning (SSL) approaches and foundation models capable of learning meaningful representations from multichannel images with minimal annotation.

http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-593914

## Abstract

Multiplexed immunofluorescence (mIF) imaging captures complex interactions between tumour and immune cells that can provide valuable information about cancer progression and patient outcome. However, extracting prognostic information from multichannel mIF images remains challenging due to the high dimensionality of the data, limited availability of annotations, and image quality variations.

This thesis investigated whether self-supervised learning (SSL) can be used to learn meaningful image representations from multichannel mIF images of NSCLC tissue for downstream survival prediction. We evaluated two SSL approaches — a hierarchical Masked Autoencoder (MAE) + Distillation of knowledge with NO labels (DINO) framework, and KRONOS, a foundation model for spatial proteomics — and compared them against a supervised ResNet18 baseline.

To improve downstream performance, several preprocessing and representation refinement strategies were explored, including:
- Autofluorescence (AF) correction
- A hierarchical feature extraction framework for KRONOS representations
- Principal component analysis (PCA)

Learned image representations were combined with a Cox proportional hazards model and evaluated using the concordance index (C-index) and time-dependent AUC.

**Key findings:** The evaluated methods achieved similar performance overall, with mean test C-index values ranging from 0.50–0.57 and time-dependent AUC values up to 0.58. The AF-corrected MAE+DINO representations with PCA applied achieved the highest average scores (C-index: 0.564, AUC: 0.577), though differences between methods were small and should be interpreted with caution. The results suggest limited usefulness of the evaluated SSL methods for multichannel mIF images under the studied configuration, while also highlighting challenges and opportunities for future representation learning approaches in multichannel mIF imaging.

**Keywords:** Multiplexed immunofluorescence, Self-supervised learning, Non-small cell lung cancer, Survival prediction, Foundation models, Tumor microenvironment

## Methods

Three main approaches were evaluated for learning image representations from mIF data:

| Method | Description |
|---|---|
| **Supervised baseline** | ResNet18 trained in a supervised manner as a performance reference point |
| **Hierarchical MAE + DINO** | A hierarchical self-supervised framework combining Masked Autoencoders with DINO-based knowledge distillation, adapted for multichannel mIF images |
| **KRONOS** | A pretrained foundation model for spatial proteomics, evaluated with a hierarchical feature extraction framework built on top of its representations |

Representations from each method were passed through a **Cox proportional hazards model** to predict patient survival, and performance was assessed using:
- **Concordance index (C-index)**
- **Time-dependent AUC**

Preprocessing and refinement strategies explored alongside these methods included autofluorescence correction and PCA-based dimensionality reduction of the learned representations.

## Repository Structure

```
.
├── baseline_resnet18/         # Supervised ResNet18 baseline model and training scripts
├── hierarchical_ssl_method/   # Hierarchical MAE + DINO self-supervised framework
├── kronos/                    # KRONOS-based feature extraction and hierarchical aggregation
├── external/                  # Third-party code used with minimal or no modification
└── README.md
```


## References

1. G. Atarsaikhan, I. Mogollon, K. Välimäki, T. Mirtti, T. Pellinen, and L. Paavolainen, "Self-supervised learning enables unbiased patient characterization from multiplexed cancer tissue microscopy images," *bioRxiv*, 2025. [https://doi.org/10.1101/2025.03.05.640729](https://www.biorxiv.org/content/10.1101/2025.03.05.640729v1)

2. M. Shaban, Y. Chang, et al., "A Foundation Model for Spatial Proteomics," *arXiv preprint arXiv:2506.03373*, 2025. Code: [https://github.com/mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS)

## Acknowledgements & Licensing

This project builds on and adapts publicly available research code. Code taken from external sources is kept in the `external/` folder with original licensing and attribution preserved.

- The hierarchical MAE + DINO framework is adapted from Atarsaikhan et al. (2025), [SSL-Multiplexed-Imaging](https://github.com/bioimage-profiling/SSL-Multiplexed-Imaging) (MIT License).
- KRONOS code and pretrained weights are from Shaban et al. (2025), [mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS), released for **non-commercial academic research use only**. Any commercial use or derivative model training requires prior approval from the original authors.

## Contact

For questions about this project, please open an issue in this repository.
