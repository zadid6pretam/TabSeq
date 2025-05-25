# TabSeq Multiclass Classification Model (Refactored for Importable Use)
# Author: Zadid Habib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, classification_report
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Softmax, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import networkx as nx

# -------------------------------
# Feature Graph Construction and Ordering Utilities
# -------------------------------

def construct_graph_for_cluster(X_cluster):
    G = nx.Graph()
    for feature in X_cluster.columns:
        G.add_node(feature)
    return G

def minimize_dispersion(X_cluster):
    variances = X_cluster.var(axis=0)
    return variances.sort_values(ascending=False).index.tolist()

def calculate_cluster_variances(X_train, cluster_labels):
    num_clusters = len(np.unique(cluster_labels))
    cluster_variances = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        feature_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_features = X_train.iloc[:, feature_indices]
        cluster_variances[cluster_id] = cluster_features.var().sum()
    return cluster_variances

def assign_weights_with_randomness(cluster_variances):
    base_weights = cluster_variances / cluster_variances.sum()
    random_factors = np.random.rand(len(base_weights))
    weights = base_weights * random_factors
    return weights / weights.sum()

def integrate_local_orderings_with_weights(local_orderings, weights, feature_names):
    feature_positions = {feature: 0 for feature in feature_names}
    total_weight = sum(weights)
    for ordering, weight in zip(local_orderings, weights):
        for position, feature_name in enumerate(ordering):
            feature_positions[feature_name] += position * weight
    averaged_positions = {f: p / total_weight for f, p in feature_positions.items()}
    return sorted(averaged_positions, key=averaged_positions.get)

# -------------------------------
# Multi-head Attention Layer (1D)
# -------------------------------

def multihead_attention_layer(inputs, num_heads, dk, dropout_rate=0.1):
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
    multihead_output = Concatenate(axis=-1)(weighted_values_list)
    multihead_output = tf.squeeze(multihead_output, axis=1)
    multihead_output = LayerNormalization(epsilon=1e-6)(multihead_output)
    multihead_output = Dropout(dropout_rate)(multihead_output)
    return Dense(dk, activation='relu')(multihead_output)

# -------------------------------
# Main Training Function
# -------------------------------

def train_multiclass_model(X_train_input, X_valid_input, X_test_input, y_train, y_valid, y_test, num_classes=3):
    # Copy inputs
    X_train = X_train_input.copy()
    X_valid = X_valid_input.copy()
    X_test = X_test_input.copy()

    # Cluster and reorder features
    num_clusters = 5
    X_features = X_train.transpose()
    cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X_features)
    clusters = [X_train[X_features.index[cluster_labels == cid]] for cid in range(num_clusters)]
    local_orderings = [minimize_dispersion(cluster) for cluster in clusters]
    cluster_variances = calculate_cluster_variances(X_train, cluster_labels)
    weights = assign_weights_with_randomness(cluster_variances)
    global_ordering = integrate_local_orderings_with_weights(local_orderings, weights, X_train.columns.tolist())

    X_train = X_train[global_ordering]
    X_valid = X_valid[global_ordering]
    X_test = X_test[global_ordering]

    # Define TabSeq model structure
    input_dim = X_train.shape[1]
    num_heads = 4
    dk = 64
    dropout_rate = 0.1

    input_layer = Input(shape=(input_dim,))
    attention_output = multihead_attention_layer(input_layer, num_heads, dk, dropout_rate)

    # Autoencoder structure
    encoded = Dense(128, activation='relu')(attention_output)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping_ae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Add noise and train autoencoder
    noise_factor = 0.1
    X_train_noisy = np.clip(X_train + noise_factor * np.random.normal(0.0, 1.0, X_train.shape), 0., 1.)
    X_valid_noisy = np.clip(X_valid + noise_factor * np.random.normal(0.0, 1.0, X_valid.shape), 0., 1.)

    autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=32,
                    validation_data=(X_valid_noisy, X_valid), callbacks=[early_stopping_ae])

    # Extract latent representations
    encoder = Model(inputs=autoencoder.input, outputs=encoded)
    encoded_train = encoder.predict(X_train)
    encoded_valid = encoder.predict(X_valid)
    encoded_test = encoder.predict(X_test)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes)
    y_valid = to_categorical(y_valid, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Classifier structure
    classifier = Sequential([
        Dense(128, activation='relu', input_dim=encoded_train.shape[1]),
        BatchNormalization(), Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping_clf = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    classifier.fit(encoded_train, y_train, epochs=50, batch_size=32,
                   validation_data=(encoded_valid, y_valid), callbacks=[early_stopping_clf])

    # Evaluation
    test_loss, test_accuracy = classifier.evaluate(encoded_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    y_test_pred_proba = classifier.predict(encoded_test)
    test_auc = roc_auc_score(y_test, y_test_pred_proba, average='macro', multi_class='ovr')
    print(f'Test AUC (macro): {test_auc:.4f}')

    y_pred_classes = np.argmax(y_test_pred_proba, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print(classification_report(y_true_classes, y_pred_classes))
