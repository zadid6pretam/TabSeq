"""
TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering

This script implements the TabSeq framework, which includes feature ordering, a denoising autoencoder, 
and a classification model for multi-class classification. Users can tune parameters to optimize 
performance for their datasets.

Environment Requirements:
- Python 3.8.18
- numpy == 1.24.4
- pandas == 2.0.3
- scikit-learn == 1.3.2
- tensorflow == 2.13.0
- networkx == 3.1

File Name:
TabSeq_MultiClassClassification.py

Instructions:
- Modify parameters like `num_clusters`, `dropout_rate`, `learning_rate`, and clustering algorithm to suit your dataset.
- Ensure your datasets are preprocessed (e.g., scaled) before using this script.
- Set `num_classes` to the number of unique classes in your dataset.
- One-hot encode target labels (`y_train`, `y_valid`, `y_test`) before training.

Acknowledgment and Citation:
- If you use this script in your research, please cite our original paper:
  Al Zadid Sultan Bin Habib, Kesheng Wang, Mary-Anne Hartley, Gianfranco Doretto, and Donald A. Adjeroh. 
  "TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering." 
  In International Conference on Pattern Recognition (ICPR), 2024, pp. 418–434. Springer.

BibTeX:
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}

Author: Zadid Habib
"""

# ============================
# Instructions for Parameter Tuning
# ============================

# Users should consider tuning the following parameters to optimize the model for their datasets:
# 
# 1. **Number of Clusters (`num_clusters`)**:
#    - Adjust the number of clusters in the feature ordering step to match the nature of your dataset.
# 
# 2. **Clustering Algorithm**:
#    - The current implementation uses KMeans for clustering features.
#    - Consider experimenting with other clustering algorithms like Agglomerative Clustering, DBSCAN, or Spectral Clustering depending on the dataset's feature relationships.
#    - Choose the clustering algorithm based on your dataset's structure and computational constraints.
# 
# 3. **Ascending or Descending Order for Feature Ordering**:
#    - The current feature ordering minimizes dispersion in descending order (highest variance first).
#    - Try ascending order (lowest variance first) or alternative ordering strategies (e.g., correlation-based) depending on how features impact your model's performance.
# 
# 4. **Dropout Rates (`dropout_rate`)**:
#    - Modify dropout rates in the Multi-Head Attention and Classifier layers to control regularization.
#    - Use higher dropout rates (e.g., 0.4–0.6) to reduce overfitting or lower rates (e.g., 0.1–0.3) if underfitting occurs.
# 
# 5. **Learning Rate (`learning_rate`)**:
#    - Adjust the learning rate for the classifier (default: 0.001).
#    - Lower values (e.g., 0.0001) can help if the model is unstable or not converging.
# 
# 6. **Batch Size (`batch_size`)**:
#    - Experiment with different batch sizes (e.g., 16, 32, 64) to balance training speed and gradient stability.
# 
# 7. **Number of Attention Heads (`num_heads`) and Dimensionality (`dk`)**:
#    - Tune the number of attention heads and their dimensions based on the dataset's feature complexity.
# 
# 8. **Early Stopping (`patience`)**:
#    - Modify the patience parameter in early stopping callbacks to allow more epochs for convergence.
# 
# 9. **Feature Ordering Strategy**:
#    - Experiment with the feature ordering methods (e.g., variance minimization, random weights) to find the best global ordering for your dataset.
#    - Try ascending or descending order for feature prioritization to match the dataset's nature.
# 
# 10. **Model Architecture**:
#     - Add or remove layers in the classifier or adjust the number of neurons per layer for better learning capacity.
#     - Ensure the architecture aligns with the dataset's complexity and size.
# 
# 11. **Noise Factor (`noise_factor`)**:
#     - Adjust the noise level when training the denoising autoencoder for better feature extraction.
# 
# 12. **Loss Function and Metrics**:
#     - For multi-class classification, ensure the loss function is `categorical_crossentropy`.
#     - Use metrics like accuracy and AUC (macro-average) for evaluation.

# Note:
# - Always validate your changes using a validation dataset to avoid overfitting.
# - Start with the provided default settings and iterate based on your dataset's performance and characteristics.
# - Use grid search or Bayesian optimization tools for systematic parameter tuning.

# ===================
# Key Multi-Class Changes
# ===================
# - Set `num_classes` to match your dataset's unique class count.
# - Update the classifier output layer to use `softmax` for multi-class classification.
# - Change loss function to `categorical_crossentropy`.
# - Ensure labels are one-hot encoded using `tensorflow.keras.utils.to_categorical`.
# - Use AUC (macro) for evaluation metrics.



import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import networkx as nx
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Softmax, LayerNormalization, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, classification_report

# ==========================
# Feature Ordering Functions
# ==========================

def construct_graph_for_cluster(X_cluster):
    G = nx.Graph()
    for feature in X_cluster.columns:
        G.add_node(feature)
    # Edges can be added based on specific conditions (omitted for simplicity)
    return G

def minimize_dispersion(X_cluster):
    # Calculate variance for each feature across samples
    variances = X_cluster.var(axis=0)
    # Order features by variance in descending order
    ordered_features = variances.sort_values(ascending=False).index.tolist()
    return ordered_features

def calculate_cluster_variances(X_train, cluster_labels):
    num_clusters = len(np.unique(cluster_labels))
    cluster_variances = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        feature_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_features = X_train.iloc[:, feature_indices]
        # Sum of variances across features in the cluster
        cluster_variances[cluster_id] = cluster_features.var().sum()
    return cluster_variances

def assign_weights_with_randomness(cluster_variances):
    # Normalize variances to get base weights
    base_weights = cluster_variances / cluster_variances.sum()
    # Introduce randomness
    random_factors = np.random.rand(len(base_weights))  # Random factors between 0 and 1
    weights = base_weights * random_factors  # Adjust base weights by random factors
    # Normalize weights to ensure they sum to 1
    normalized_weights = weights / weights.sum()
    return normalized_weights

def integrate_local_orderings_with_weights(local_orderings, weights, feature_names):
    feature_positions = {feature: 0 for feature in feature_names}
    total_weight = sum(weights)
    for ordering, weight in zip(local_orderings, weights):
        for position, feature_name in enumerate(ordering):
            feature_positions[feature_name] += (position * weight)
    averaged_positions = {feature: pos / total_weight for feature, pos in feature_positions.items()}
    global_ordering = sorted(averaged_positions, key=averaged_positions.get)
    return global_ordering

# ======================
# Feature Ordering Steps
# ======================

# Assume X_train_scaled_df5, X_valid_scaled_df5, X_test_scaled_df5, y_train5, y_valid5, y_test_5 are already defined and scaled

# Copy datasets to ensure original data is not modified
X_train = X_train_scaled_df5.copy()
X_valid = X_valid_scaled_df5.copy()
X_test = X_test_scaled_df5.copy()

# Number of clusters (adjust as needed)
num_clusters = 5

# Transpose data to have features as rows for clustering
X_features = X_train.transpose()

# Clustering model (cluster features)
clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = clustering_model.fit_predict(X_features)

# Group features by clusters
clusters = []
for cluster_id in range(num_clusters):
    features_in_cluster = X_features.index[cluster_labels == cluster_id]
    X_cluster = X_train[features_in_cluster]
    clusters.append(X_cluster)

# Perform local ordering within each cluster
local_orderings = [minimize_dispersion(X_cluster) for X_cluster in clusters]

# Calculate cluster variances
cluster_variances = calculate_cluster_variances(X_train, cluster_labels)

# Assign weights with randomness
weights = assign_weights_with_randomness(cluster_variances)

# Feature names
feature_names = X_train.columns.tolist()

# Integrate local orderings into a global ordering
global_ordering = integrate_local_orderings_with_weights(local_orderings, weights, feature_names)

# Reorder datasets based on the global ordering
X_train_reordered = X_train[global_ordering]
X_valid_reordered = X_valid[global_ordering]
X_test_reordered = X_test[global_ordering]

# =======================
# TabSeq Model Components
# =======================

def multihead_attention_layer(inputs, num_heads, dk, dropout_rate=0.1):
    # Reshape inputs to (batch_size, sequence_length, feature_dim)
    x = tf.expand_dims(inputs, axis=1)  # Treat features as a sequence
    weighted_values_list = []
    for _ in range(num_heads):
        # Linear projections for Q, K, V
        Q = Dense(dk)(x)
        K = Dense(dk)(x)
        V = Dense(dk)(x)
        # Scaled dot-product attention calculation
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        depth = tf.cast(tf.shape(K)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        weights = Softmax(axis=-1)(logits)
        # Weighting value vectors by the attention weights
        weighted_values = tf.matmul(weights, V)
        weighted_values_list.append(weighted_values)
    # Concatenate the weighted values from all heads
    multihead_output = Concatenate(axis=-1)(weighted_values_list)
    # Flatten the output
    multihead_output = tf.squeeze(multihead_output, axis=1)
    # Apply layer normalization and dropout
    multihead_output = LayerNormalization(epsilon=1e-6)(multihead_output)
    multihead_output = Dropout(dropout_rate)(multihead_output)
    # Combine heads' outputs
    combined_output = Dense(dk, activation='relu')(multihead_output)
    return combined_output

# ===================
# Build the TabSeq Model
# ===================

# Parameters
input_dim = X_train_reordered.shape[1]
num_heads = 4
dk = 64  # Dimensionality for attention heads
dropout_rate = 0.1

# Input Layer
input_layer = Input(shape=(input_dim,))

# Multi-Head Attention
attention_output = multihead_attention_layer(input_layer, num_heads, dk, dropout_rate)

# Denoising Autoencoder
encoded = Dense(128, activation='relu')(attention_output)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)

decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stopping_ae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Add noise to inputs for denoising autoencoder
noise_factor = 0.1
X_train_noisy = X_train_reordered + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_reordered.shape)
X_valid_noisy = X_valid_reordered + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_valid_reordered.shape)

# Ensure data remains within valid range
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_valid_noisy = np.clip(X_valid_noisy, 0., 1.)

# Training the Autoencoder
autoencoder.fit(
    X_train_noisy, X_train_reordered,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid_noisy, X_valid_reordered),
    callbacks=[early_stopping_ae]
)

# Encoder Model for Feature Extraction
encoder = Model(inputs=autoencoder.input, outputs=encoded)

# Extract encoded features
encoded_train = encoder.predict(X_train_reordered)
encoded_valid = encoder.predict(X_valid_reordered)
encoded_test = encoder.predict(X_test_reordered)

# Preprocess labels to one-hot encode them
num_classes = 3  # Example: Change to the actual number of classes
y_train5 = to_categorical(y_train5, num_classes)
y_valid5 = to_categorical(y_valid5, num_classes)
y_test_5 = to_categorical(y_test_5, num_classes)

# Classifier Model
classifier = Sequential([
    Dense(128, activation='relu', input_dim=encoded_train.shape[1]),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Multi-class classification
])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
classifier.fit(
    encoded_train, y_train5,
    epochs=50,
    batch_size=32,
    validation_data=(encoded_valid, y_valid5),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Evaluation
test_loss, test_accuracy = classifier.evaluate(encoded_test, y_test_5)
print(f'Test Accuracy: {test_accuracy:.4f}')

# AUC and Classification Report
y_test_pred_proba = classifier.predict(encoded_test)
test_auc = roc_auc_score(y_test_5, y_test_pred_proba, average='macro', multi_class='ovr')
print(f'Test AUC (macro): {test_auc:.4f}')

y_test_pred_classes = np.argmax(y_test_pred_proba, axis=1)
y_test_true_classes = np.argmax(y_test_5, axis=1)
print(classification_report(y_test_true_classes, y_test_pred_classes))