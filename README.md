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
