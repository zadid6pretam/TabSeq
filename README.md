# TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Sequencing](https://img.shields.io/badge/Sequencing-Feature%20Ordering-blueviolet)
![Backbone](https://img.shields.io/badge/Backbone-Transformer--Autoencoder-orange)
![Model](https://img.shields.io/badge/Model-TabSeq-skyblue)
![Conference](https://img.shields.io/badge/Conference-ICPR%202024-blue)
[![Citation](https://img.shields.io/badge/Cite%20Us-Springer--ICPR--2024-red)](https://doi.org/10.1007/978-3-031-78128-5_27)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

TabSeq is a cutting-edge framework designed to bridge the gap in applying deep learning to tabular datasets, which often has feature heterogeneous and sequential characteristics. By leveraging feature ordering, TabSeq organizes features to maximize their relevance and interactions, significantly improving the model's ability to learn from tabular data.

The framework incorporates:

- Clustering to group features with similar characteristics in feature ordering.
- Multi-Head Attention (MHA) to prioritize essential feature interactions.
- Denoising Autoencoder (DAE) to reduce redundancy and reconstruct noisy inputs.

TabSeq has demonstrated remarkable performance across various real-world datasets, outperforming traditional methods. Its modular design and adaptability make it a powerful tool for both binary and multi-class classification tasks, addressing challenges in health informatics, financial modeling, and more.

Explore the potential of TabSeq and see how it transforms deep learning on tabular data.

## Files
- **TabSeq_arxiv.pdf**: Research paper (pre-print) describing the framework.
- **binary.py**: Implementation for binary classification tasks.
- **multiclass.py**: Implementation for multi-class classification tasks.

## Requirements
- Python 3.8+
- numpy, pandas, scikit-learn, tensorflow, networkx

## Citation

Al Zadid Sultan Bin Habib, Kesheng Wang, Mary-Anne Hartley, Gianfranco Doretto, and Donald A. Adjeroh. "TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering." In International Conference on Pattern Recognition (ICPR), 2024, pp. 418–434. Springer.


BibTeX:
```bash
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
```

## Installation

You can install **TabSeq** in multiple ways depending on your use case:

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/TabSeq.git
cd TabSeq
pip install -r requirements.txt
pip install -e .
```

---

### Option 2: Install via pip from GitHub (No Cloning Needed)

```bash
pip install git+https://github.com/zadid6pretam/TabSeq.git
```

---

### Option 3: Install in a Virtual Environment

```bash
python -m venv tabseq-env
source tabseq-env/bin/activate  # On Windows: tabseq-env\Scripts\activate
git clone https://github.com/zadid6pretam/TabSeq.git
cd TabSeq
pip install -r requirements.txt
pip install -e .
```

---

### Option 4: Manual Install Using setup.py

```bash
git clone https://github.com/zadid6pretam/TabSeq.git
cd TabSeq
pip install .
```

---

### Option 5: Install from PyPI

```bash
pip install TabSeq
```

### Example Usage

```bash
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabseq.binary import train_binary_model
from tabseq.multiclass import train_multiclass_model

# Generate synthetic dataset
X = np.random.rand(40, 80)                   # 40 samples, 80 features
y_binary = np.random.randint(0, 2, 40)       # Binary labels (0 or 1)
y_multiclass = np.random.randint(0, 3, 40)   # Multiclass labels (0, 1, 2)

# Scale features
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))

# Split into train, valid, test
X_train, X_temp, y_train_b, y_temp_b = train_test_split(X_scaled, y_binary, test_size=0.4, stratify=y_binary)
X_valid, X_test, y_valid_b, y_test_b = train_test_split(X_temp, y_temp_b, test_size=0.5, stratify=y_temp_b)

_, X_temp, y_train_m, y_temp_m = train_test_split(X_scaled, y_multiclass, test_size=0.4, stratify=y_multiclass)
X_valid_m, X_test_m, y_valid_m, y_test_m = train_test_split(X_temp, y_temp_m, test_size=0.5, stratify=y_temp_m)

# Run TabSeq for Binary Classification
train_binary_model(X_train, X_valid, X_test, y_train_b, y_valid_b, y_test_b)

# Run TabSeq for Multi-Class Classification
train_multiclass_model(X_train, X_valid, X_test, y_train_m, y_valid_m, y_test_m, num_classes=3)
```

### Default Parameter Values for Binary Classification

```bash
# =======================================================
# TabSeq Default Configuration Parameters (Binary Version)
# =======================================================
# Feature Ordering:
# - num_clusters: 5 (KMeans clustering is applied to transpose of feature matrix)
# - Intra-cluster ordering: Features sorted in descending order of variance
# - Global ordering: Integrated from local orderings using variance-based random weights

# Autoencoder (Denoising with Attention):
# - Noise: Gaussian noise with std = 0.1 added before training, clipped to [0, 1]
# - Attention Heads: 4
# - Attention Head Dimension (dk): 64
# - Dropout Rate in Attention: 0.1
# - Epochs: 50
# - Batch Size: 32
# - Loss Function: Mean Squared Error
# - Optimizer: Adam
# - EarlyStopping: patience = 5, monitor = 'val_loss', restore_best_weights = True

# Classifier:
# - Architecture: [Dense(128, relu) → BN → Dropout(0.5) → Dense(64, relu) → BN → Dropout(0.5) → Dense(1, sigmoid)]
# - Epochs: 50
# - Batch Size: 32
# - Loss Function: Binary Crossentropy
# - Metric: Accuracy
# - EarlyStopping: patience = 5, monitor = 'val_loss', restore_best_weights = True
```

### Default Parameter Values for Multiclass Classification

```bash
# ===============================================
# TabSeq Default Configuration (Multiclass Version)
# ===============================================

# Feature Ordering:
# - num_clusters: 5 (KMeans clustering on transposed feature matrix)
# - Intra-cluster ordering: Features sorted by descending variance
# - Global ordering: Weighted integration of local orderings based on random-scaled variances

# Denoising Autoencoder with Multihead Attention:
# - Noise: Gaussian noise with std = 0.1, clipped between [0, 1]
# - Attention Heads: 4
# - Head Dimension (dk): 64
# - Dropout Rate in Attention: 0.1
# - Encoder: Dense(128 → 64), BatchNorm, Dropout(0.2)
# - Decoder: Dense(input_dim, sigmoid)
# - Epochs: 50
# - Batch Size: 32
# - Loss Function: Mean Squared Error
# - Optimizer: Adam
# - EarlyStopping: patience = 5, monitor = 'val_loss'

# Classifier:
# - Architecture: [Dense(128, relu) → BN → Dropout(0.5) → Dense(64, relu) → BN → Dropout(0.5) → Dense(num_classes, softmax)]
# - Loss Function: Categorical Crossentropy
# - Metric: Accuracy
# - Epochs: 50
# - Batch Size: 32
# - EarlyStopping: patience = 5, monitor = 'val_loss'

# Evaluation:
# - AUC: macro-average, using one-vs-rest (ovr)
# - Classification report: includes precision, recall, F1 for each class
```

# Our Related Works

### GOTabPFN (ICML 2026)

Our recent ICML 2026 Regular main conference paper on feature ordering and compression for tabular foundation models for high-dimensional low-sample-size tabular data:
- **GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data**

- GitHub: https://github.com/zadid6pretam/GOTabPFN
- - **Find it on ICML portal:** https://icml.cc/virtual/2026/poster/62523
- **Project Webpage:** https://www.zadidhabib.com/gotabpfn.html
- **OpenReview:** https://openreview.net/forum?id=fpqfV3lCIB
- **Hugging Face Space:** [ZeroGPU Live Demo](https://zadid6pretam-GOTabPFN.hf.space) *(recommended; faster GPU-backed testing)* | [CPU Backup Demo](https://zadid6pretam-GOTabPFN-CPU.hf.space) *(use if ZeroGPU is unavailable)* | [ZeroGPU Space Repository](https://huggingface.co/spaces/zadid6pretam/GOTabPFN) | [CPU Backup Space Repository](https://huggingface.co/spaces/zadid6pretam/GOTabPFN_CPU)

```bibtex
@inproceedings{habib2026gotabpfn,
  title     = {GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

### iSyncTab (ECCV 2026)

Our neural synchrony-based cross-modal feature sequencing framework for multimodal learning with image and tabular data. iSyncTab addresses the image–tabular integration problem by aligning and sequencing cross-modal feature groups before structured multimodal representation learning.

- **iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony**  
- Accepted at the European Conference on Computer Vision (ECCV 2026)
- GitHub: https://github.com/zadid6pretam/iSyncTab (will be made public soon)
- Project Page: https://www.zadidhabib.com/isynctab.html (will be made public soon)

```bibtex
@inproceedings{habib2026isynctab,
  title     = {iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2026}
}
```
- If you are interested in cross-modal feature sequencing, neural synchrony-guided image–tabular integration, and order-aware multimodal representation learning, please refer to the iSyncTab repository, project page, and paper.

### BSTabDiff (ICLR 2026 DeLTa Workshop)

Our generative modeling framework for high-dimensional low-sample-size tabular data:
- BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation
GitHub: https://github.com/zadid6pretam/BSTabDiff

```bibtex
@inproceedings{habib2026bstabdiff,
  title     = {BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {ICLR 2026 2nd Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)},
  year      = {2026}
}
```
- If you are interested in high-dimensional tabular synthesis, block-subunit generation, and diffusion/flow priors for HDLSS tabular data, please also refer to the BSTabDiff repository and paper.

### iStructTab (ICPR 2026)

Our structured feature sequencing framework for multimodal learning with image and tabular data. This work is part of my PhD research on feature sequencing or ordering for multimodal image-tabular representation learning.

- **iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data**  
  GitHub: https://github.com/zadid6pretam/iStructTab

```bibtex
@inproceedings{habib2026istructtab,
  title     = {iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- If you are interested in structured feature sequencing, multimodal fusion of image and tabular data (the integration problem), and feature order-aware tabular representation learning, please also refer to the iStructTab repository and paper.

## DynaTab (AAAI 2026 NeuroAI Workshop)

Our more recent work on learned feature ordering for high-dimensional tabular data:

- **DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data**
GitHub: https://github.com/zadid6pretam/DynaTab

```bibtex
@inproceedings{habib2026dynatab,
  title     = {{DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data}},
  author    = {Habib, Al Zadid Sultan Bin and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the AAAI 2026 First International Workshop on Neuro for AI \& AI for Neuro: Towards Multi-Modal Natural Intelligence (NeuroAI)},
  year      = {2026},
  series    = {PMLR}
}
```
- If you are interested in learned feature ordering, neural rewiring for high-dimensional tabular data, and sequential backbone design for HDLSS settings, please also refer to the DynaTab repository and paper.
- DynaTab has completed camera-ready submission, and the public proceedings version is expected to appear online later.


### TabSeq (ICPR 2024)

Our earlier work on sequential modeling for tabular data:

- **TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**  
  GitHub: https://github.com/zadid6pretam/TabSeq  
  Springer ICPR 2024 proceedings: https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27

```bibtex
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
```
- If you are interested in sequential ordering for tabular data, deep sequential backbones, and early feature-ordering-based tabular modeling, please also refer to the TabSeq repository and paper.


----------------------------------------------------------------------------------------------------------------------------------------------------------


### ZAYAN (ICPR 2026)

This repository corresponds to our separate collaborative work on tabular remote sensing and environmental data:
- ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data
GitHub: https://github.com/zadid6pretam/ZAYAN

```bibtex
@inproceedings{habib2026zayan,
  title     = {ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data},
  author    = {Habib, Al Zadid Sultan Bin and Tasnim, Tanpia and Islam, Md. Ekramul and Tabasum, Muntasir},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- ZAYAN focuses on feature-level contrastive learning and Transformer-based classification for tabular remote sensing and environmental datasets.
- Unlike my PhD dissertation projects on high-dimensional tabular learning and HDLSS modeling, ZAYAN was developed as a separate collaboration.

## Contact

For any questions, issues, or suggestions related to this repository, please feel free to contact us or open an issue on GitHub.
