"""
TabSeq Feature Ordering Module

This module provides feature ordering functionality for tabular data using
variance-based clustering and weighted integration. It can be used to reorder
feature columns for deep learning models.

Author: Zadid Habib
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import networkx as nx

def construct_graph_for_cluster(X_cluster):
    """Creates a feature graph from cluster columns."""
    G = nx.Graph()
    for feature in X_cluster.columns:
        G.add_node(feature)
    return G  # Edge construction can be added here if needed

def minimize_dispersion(X_cluster, ascending=False):
    """Orders features within a cluster by variance."""
    variances = X_cluster.var(axis=0)
    return variances.sort_values(ascending=ascending).index.tolist()

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

def reorder_features(X_train_input, cluster_size=5, ascending=False):
    """
    Perform feature ordering on a training dataset.

    Args:
        X_train_input (pd.DataFrame): The input training DataFrame.
        cluster_size (int): Number of feature clusters to form.
        ascending (bool): Sort order for variance. If True, low variance first.

    Returns:
        tuple: (global_ordering, reordered_dataframe)
    """
    X_train = X_train_input.copy()
    X_features = X_train.transpose()
    cluster_labels = KMeans(n_clusters=cluster_size, random_state=42).fit_predict(X_features)

    clusters = [X_train[X_features.index[cluster_labels == i]] for i in range(cluster_size)]
    local_orderings = [minimize_dispersion(X_cluster, ascending=ascending) for X_cluster in clusters]
    cluster_variances = calculate_cluster_variances(X_train, cluster_labels)
    weights = assign_weights_with_randomness(cluster_variances)
    global_ordering = integrate_local_orderings_with_weights(local_orderings, weights, X_train.columns.tolist())
    reordered_df = X_train[global_ordering]

    return global_ordering, reordered_df