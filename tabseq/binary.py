"""
TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering

This script implements the TabSeq framework, which includes feature ordering,
a denoising autoencoder, and a classification model. Users can tune parameters
to optimize performance for their datasets.

Author: Zadid Habib
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Softmax,
    LayerNormalization, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
import networkx as nx

# ================================
# Feature Ordering Helper Methods
# ================================

def construct_graph_for_cluster(X_cluster):
    """Creates a feature graph from cluster columns."""
    G = nx.Graph()
    for feature in X_cluster.columns:
        G.add_node(feature)
    return G  # Edge construction can be added here if needed

def minimize_dispersion(X_cluster):
    """Orders features within a cluster by descending variance."""
    variances = X_cluster.var(axis=0)
    return variances.sort_values(ascending=False).index.tolist()

def calculate_cluster_variances(X_train, cluster_labels):
    """Calculates total variance within each cluster of features."""
    num_clusters = len(np.unique(cluster_labels))
    cluster_variances = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        feature_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_features = X_train.iloc[:, feature_indices]
        cluster_variances[cluster_id] = cluster_features.var().sum()
    return cluster_variances

def assign_weights_with_randomness(cluster_variances):
    """Assigns weights to clusters based on variance with added randomness."""
    base_weights = cluster_variances / cluster_variances.sum()
    random_factors = np.random.rand(len(base_weights))
    weights = base_weights * random_factors
    return weights / weights.sum()

def integrate_local_orderings_with_weights(local_orderings, weights, feature_names):
    """Integrates local orderings across clusters using weighted positions."""
    feature_positions = {feature: 0 for feature in feature_names}
    total_weight = sum(weights)
    for ordering, weight in zip(local_orderings, weights):
        for position, feature_name in enumerate(ordering):
            feature_positions[feature_name] += (position * weight)
    averaged_positions = {f: pos / total_weight for f, pos in feature_positions.items()}
    return sorted(averaged_positions, key=averaged_positions.get)

# =======================
# Multihead Attention Layer
# =======================

def multihead_attention_layer(inputs, num_heads, dk, dropout_rate=0.1):
    """Implements a multi-head self-attention mechanism over tabular features."""
    x = tf.expand_dims(inputs, axis=1)
    weighted_values_list = []
    for _ in range(num_heads):
        Q = Dense(dk)(x)
        K = Dense(dk)(x)
        V = Dense(dk)(x)
        logits = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        weights = Softmax(axis=-1)(logits)
        weighted_values = tf.matmul(weights, V)
        weighted_values_list.append(weighted_values)
    concat_output = Concatenate(axis=-1)(weighted_values_list)
    flattened = tf.squeeze(concat_output, axis=1)
    normed = LayerNormalization(epsilon=1e-6)(flattened)
    dropped = Dropout(dropout_rate)(normed)
    return Dense(dk, activation='relu')(dropped)

# ===========================
# Binary Training Function
# ===========================

def train_binary_model(X_train_input, X_valid_input, X_test_input, y_train, y_valid, y_test):
    """
    Trains a binary classification model using TabSeq with feature ordering and a denoising autoencoder.
    
    Args:
        X_train_input (pd.DataFrame): Scaled training features
        X_valid_input (pd.DataFrame): Scaled validation features
        X_test_input (pd.DataFrame): Scaled test features
        y_train (array-like): Training labels (binary)
        y_valid (array-like): Validation labels (binary)
        y_test (array-like): Test labels (binary)
    """

    # ======================
    # Data Preparation
    # ======================
    X_train = X_train_input.copy()
    X_valid = X_valid_input.copy()
    X_test = X_test_input.copy()

    # Step 1: Clustering features
    num_clusters = 5
    X_features = X_train.transpose()
    cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X_features)

    # Step 2: Cluster-wise ordering
    clusters = [X_train[X_features.index[cluster_labels == i]] for i in range(num_clusters)]
    local_orderings = [minimize_dispersion(X_cluster) for X_cluster in clusters]
    cluster_variances = calculate_cluster_variances(X_train, cluster_labels)
    weights = assign_weights_with_randomness(cluster_variances)
    global_ordering = integrate_local_orderings_with_weights(local_orderings, weights, X_train.columns.tolist())

    # Step 3: Reorder feature columns
    X_train_reordered = X_train[global_ordering]
    X_valid_reordered = X_valid[global_ordering]
    X_test_reordered = X_test[global_ordering]

    # ======================
    # Autoencoder with Attention
    # ======================
    input_dim = X_train_reordered.shape[1]
    input_layer = Input(shape=(input_dim,))
    attention_output = multihead_attention_layer(input_layer, num_heads=4, dk=64, dropout_rate=0.1)

    encoded = Dense(128, activation='relu')(attention_output)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(
        np.clip(X_train_reordered + 0.1 * np.random.normal(size=X_train_reordered.shape), 0., 1.),
        X_train_reordered,
        epochs=50,
        batch_size=32,
        validation_data=(np.clip(X_valid_reordered + 0.1 * np.random.normal(size=X_valid_reordered.shape), 0., 1.),
                         X_valid_reordered),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    encoder = Model(inputs=autoencoder.input, outputs=encoded)
    encoded_train = encoder.predict(X_train_reordered, verbose=0)
    encoded_valid = encoder.predict(X_valid_reordered, verbose=0)
    encoded_test = encoder.predict(X_test_reordered, verbose=0)

    # ======================
    # Classifier
    # ======================
    classifier = Sequential([
        Dense(128, activation='relu', input_dim=encoded_train.shape[1]),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(
        encoded_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(encoded_valid, y_valid),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    # ======================
    # Evaluation
    # ======================
    test_loss, test_accuracy = classifier.evaluate(encoded_test, y_test, verbose=0)
    print(f" Test Accuracy: {test_accuracy:.4f}")

    y_test_pred_proba = classifier.predict(encoded_test, verbose=0)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    print(f" Test AUC: {test_auc:.4f}")
