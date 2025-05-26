"""
TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering
"""

# Version of the TabSeq package (bump this for each release)
__version__ = "0.1.4"

# Expose the main training functions at package level
from .binary import train_binary_model
from .multiclass import train_multiclass_model
