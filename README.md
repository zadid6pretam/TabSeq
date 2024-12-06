# TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering

TabSeq is a cutting-edge framework designed to bridge the gap in applying deep learning to tabular datasets, which often has feature heterogeneous and sequential characteristics. By leveraging feature ordering, TabSeq organizes features to maximize their relevance and interactions, significantly improving the model's ability to learn from tabular data.

The framework incorporates:

- Clustering to group features with similar characteristics.
- Multi-Head Attention (MHA) to prioritize essential feature interactions.
- Denoising Autoencoder (DAE) to reduce redundancy and reconstruct noisy inputs.

TabSeq has demonstrated remarkable performance across various real-world datasets, outperforming traditional methods. Its modular design and adaptability make it a powerful tool for both binary and multi-class classification tasks, addressing challenges in health informatics, financial modeling, and more.

Explore the potential of TabSeq and see how it transforms deep learning on tabular data.

## Files
- **TabSeq_arxiv.pdf**: Research paper (pre-print) describing the framework.
- **TabSeq_BinaryClassification.py**: Implementation for binary classification tasks.
- **TabSeq_MultiClassClassification.py**: Implementation for multi-class classification tasks.

## Requirements
- Python 3.8+
- numpy, pandas, scikit-learn, tensorflow, networkx

## Citation
Al Zadid Sultan Bin Habib, Kesheng Wang, Mary-Anne Hartley, Gianfranco Doretto, and Donald A. Adjeroh. "TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering." In International Conference on Pattern Recognition (ICPR), 2024, pp. 418–434. Springer.

BibTeX:
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
