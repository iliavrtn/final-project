# -*- coding: utf-8 -*-
"""
dbscan_clustering.py

Contains the DBSCAN clustering logic:
vectorization, clustering with varying ε, MinPts, distance metrics,
PCA for visualization, and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def run_dbscan_clustering(segmented_data, eps=0.5, min_samples=5, distance_metric='cosine'):
    """
    Runs DBSCAN clustering on the given segmented DataFrame,
    visualizes the results via PCA (2D), and computes evaluation metrics.

    Parameters:
    - segmented_data: DataFrame containing 'text' and optionally 'label_id'
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    - distance_metric: The metric to use when calculating distance between instances.

    Returns:
    - cluster_labels: Array of cluster assignments for each sample.
    """
    texts = segmented_data['text'].tolist()
    true_labels = segmented_data.get('label_id', None)

    # Step 1: Vectorize the text data with TF-IDF
    print("Vectorizing text data for DBSCAN...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    print("Vectorization complete. Shape of TF-IDF matrix:", X.shape)

    # Normalize features for cosine metric if needed
    if distance_metric == 'cosine':
        # When using cosine distance, it's often beneficial to normalize vectors
        X = normalize(X)

    # Step 2: DBSCAN Clustering
    print(f"Clustering with DBSCAN using eps={eps}, min_samples={min_samples}, metric={distance_metric}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(X)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"DBSCAN found {n_clusters} clusters with {n_noise} noise points.")

    # Step 3: PCA for Visualization
    print("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    # Convert sparse matrix to dense for PCA if necessary
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    X_pca = pca.fit_transform(X_dense)
    print("PCA complete. Plotting DBSCAN clusters...")

    # Plot Predicted Clusters
    plt.figure(figsize=(12, 8))
    unique_labels = set(cluster_labels)
    # Use a color palette with enough colors
    palette = sns.color_palette("hls", len(unique_labels))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=cluster_labels,
        palette=palette,
        legend='full',
        alpha=0.7
    )
    plt.title(f"DBSCAN Clusters (eps={eps}, min_samples={min_samples}, metric={distance_metric})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster', loc='best')
    plt.show()

    # Step 4: Compute Clustering Evaluation Metrics (if ground truth available)
    if true_labels is not None and len(set(cluster_labels)) > 1:
        print("\nComputing clustering evaluation metrics...")
        silhouette = silhouette_score(X_dense, cluster_labels, metric=distance_metric)
        calinski = calinski_harabasz_score(X_dense, cluster_labels)
        davies = davies_bouldin_score(X_dense, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Index: {calinski:.4f}")
        print(f"Davies-Bouldin Index: {davies:.4f}")
    else:
        print("Skipping evaluation metrics due to lack of true labels or single cluster scenario.")

    return cluster_labels
