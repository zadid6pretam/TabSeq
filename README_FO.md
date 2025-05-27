# TabSeq Feature Ordering

[![PyPI version](https://img.shields.io/pypi/v/tabseq-feature-ordering.svg)](https://pypi.org/project/tabseq-feature-ordering/)
[![Python Versions](https://img.shields.io/pypi/pyversions/tabseq-feature-ordering.svg)](https://pypi.org/project/tabseq-feature-ordering/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICPR 2024 Paper](https://img.shields.io/badge/Paper-ICPR%202024-blue)](https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27)
[![GitHub Stars](https://img.shields.io/github/stars/zadid6pretam/TabSeq?style=social)](https://github.com/zadid6pretam/TabSeq)

---

This module extracts and packages the feature ordering algorithm used in **TabSeq (ICPR 2024)** as a standalone utility, enabling integration into any tabular deep learning pipeline.



## Key Features

-  Variance-based intra-cluster ordering  
-  KMeans clustering for feature grouping  
-  Weighted global ordering from local cluster orders  
-  Minimal dependencies, flexible integration  

## Installation

```bash
pip install tabseq-feature-ordering
```

## Usage

```python
from tabseq_feature_ordering import reorder_features

# Inputs
X_train = ...  # pandas DataFrame of shape (n_samples, n_features)
cluster_size = 5
sort_order = 'descending'  # or 'ascending'

# Output
global_ordering, X_train_reordered = reorder_features(X_train, cluster_size, sort_order)
```

## Parameters

- `X_train`: Tabular training data as `pd.DataFrame`
- `cluster_size`: Number of clusters (e.g., 5)
- `sort_order`: Intra-cluster sorting order by variance (`'ascending'` or `'descending'`)

## Output

- `global_ordering`: List of column names in reordered order
- `X_train_reordered`: DataFrame with reordered columns

## Example

```python
import pandas as pd
import numpy as np
from tabseq_feature_ordering import reorder_features

# Example input
X = pd.DataFrame(np.random.rand(40, 80), columns=[f"F{i}" for i in range(80)])

# Run feature ordering
order, X_reordered = reorder_features(X, cluster_size=5, sort_order='descending')

print(order[:10])  # First 10 features in the new order
```

## License

MIT License © 2024 Zadid Habib

## Citation

If you use this module, please cite our paper:

Habib, Al Zadid Sultan Bin, Kesheng Wang, Mary-Anne Hartley, Gianfranco Doretto, and Donald A. Adjeroh. "TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering." In International Conference on Pattern Recognition, pp. 418-434. Cham: Springer Nature Switzerland, 2024.

---

### Bibtex

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
