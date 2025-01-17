# -*- coding: utf-8 -*-
"""
pam_clustering_optimized.py

Optimized PAM clustering module using dimensionality reduction (TruncatedSVD)
and sampling for large-scale text data. It includes:
- BUILD algorithm for initial medoid selection
- SWAP algorithm for medoid refinement
- Cluster assignment
- PCA for visualization
- Evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.metrics import pairwise_distances


# Core PAM functions (without modifications)
def build_medoid_initialization(X, k, distance_metric):
    print("Starting BUILD algorithm for initial medoid selection...")
    n_samples = X.shape[0]
    print("Computing pairwise distance matrix...")
    D = pairwise_distances(X, metric=distance_metric)
    medoids = []
    for medoid_idx in range(k):
        print(f"Selecting medoid {medoid_idx + 1}/{k}...")
        best_candidate = None
        best_cost = np.inf
        for candidate in range(n_samples):
            if candidate in medoids:
                continue
            current_medoids = medoids + [candidate]
            distances_to_medoids = np.min(D[:, current_medoids], axis=1)
            cost = distances_to_medoids.sum()
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate
        medoids.append(best_candidate)
        print(f"Medoid {medoid_idx + 1} selected: index {best_candidate} with cost {best_cost:.4f}")
    print("BUILD algorithm completed.")
    return medoids, D


def swap_optimization(medoids, D, max_iter):
    print("Starting SWAP algorithm for medoid optimization...")
    k = len(medoids)
    n_samples = D.shape[0]
    current_medoids = medoids.copy()
    distances_to_medoids = np.min(D[:, current_medoids], axis=1)
    current_cost = distances_to_medoids.sum()
    print(f"Initial total cost: {current_cost:.4f}")

    for iteration in range(max_iter):
        print(f"SWAP iteration {iteration + 1}")
        improvement = False
        # Iterate over indices of medoids to allow swapping without searching for index
        for medoid_idx in range(k):
            m = current_medoids[medoid_idx]
            for o in range(n_samples):
                if o in current_medoids:
                    continue
                candidate_medoids = current_medoids.copy()
                candidate_medoids[medoid_idx] = o  # swap at specific position
                new_distances = np.min(D[:, candidate_medoids], axis=1)
                new_cost = new_distances.sum()
                if new_cost < current_cost:
                    print(f"  Found improvement: swapping medoid {m} with candidate {o}, new cost {new_cost:.4f}")
                    current_cost = new_cost
                    current_medoids = candidate_medoids
                    improvement = True
        if not improvement:
            print("No further improvements found. Ending SWAP algorithm.")
            break
    print("SWAP algorithm completed.")
    print(f"Optimized medoids: {current_medoids}, final cost: {current_cost:.4f}")
    return current_medoids


def assign_clusters(X, medoids, distance_metric):
    print("Assigning clusters based on optimized medoids...")
    D = pairwise_distances(X, X[medoids], metric=distance_metric)
    cluster_labels = np.argmin(D, axis=1)
    print("Cluster assignment complete.")
    return cluster_labels


def run_pam_clustering_with_sampling(segmented_data, k=14, distance_metric='euclidean',
                                     max_iter=300, n_components=100, sample_fraction=0.1):
    """
    Runs PAM clustering using dimensionality reduction and sampling.
    """
    texts = segmented_data['text'].tolist()
    true_labels = segmented_data.get('label_id', None)

    # Step 1: Vectorize text data with TF-IDF
    print("Vectorizing text data for PAM with sampling...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    print("Vectorization complete. Shape of TF-IDF matrix:", X.shape)

    # Step 2: Dimensionality Reduction
    print(f"Reducing dimensionality to {n_components} components using TruncatedSVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    print("Dimensionality reduction complete. New shape:", X_reduced.shape)

    X_dense = X_reduced  # Use reduced data for clustering

    # Step 3: Sampling for initial medoid determination
    n_samples = X_dense.shape[0]
    sample_size = int(n_samples * sample_fraction)
    print(f"Sampling {sample_size} out of {n_samples} data points for medoid initialization...")
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    X_sample = X_dense[sample_indices]

    # Run PAM on the sampled data
    print("Running PAM on sampled data...")
    initial_medoids_sample, D_sample = build_medoid_initialization(X_sample, k, distance_metric)
    optimized_medoids_sample = swap_optimization(initial_medoids_sample, D_sample, max_iter)

    # Map sample medoids back to original indices
    optimized_medoids = [sample_indices[idx] for idx in optimized_medoids_sample]

    # Step 4: Assign clusters for the full dataset using optimized medoids
    print("Assigning clusters to the full dataset using determined medoids...")
    cluster_labels = assign_clusters(X_dense, optimized_medoids, distance_metric)

    # Step 5: PCA for Visualization
    print("Performing PCA for dimensionality reduction (visualization)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    print("PCA dimensionality reduction complete. Plotting clusters...")

    # Plot Predicted Clusters
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hls", k)
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=cluster_labels,
        palette=palette,
        legend='full',
        alpha=0.7
    )
    plt.title("PAM Clusters of Text Data (PCA 2D Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster', loc='best')
    plt.show()

    # Plot Actual Labels if available
    if true_labels is not None:
        plt.figure(figsize=(12, 8))
        unique_labels = np.unique(true_labels)
        palette_labels = sns.color_palette("hls", len(unique_labels))
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=true_labels,
            palette=palette_labels,
            legend='full',
            alpha=0.7
        )
        plt.title("Actual Labels of Text Data (PCA 2D Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title='True Label', loc='best')
        plt.show()

    # Step 6: Evaluation Metrics
    if true_labels is not None:
        print("\nComputing clustering evaluation metrics...")
        silhouette = silhouette_score(X_dense, cluster_labels, metric=distance_metric)
        calinski = calinski_harabasz_score(X_dense, cluster_labels)
        davies = davies_bouldin_score(X_dense, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Index: {calinski:.4f}")
        print(f"Davies-Bouldin Index: {davies:.4f}")
    else:
        print("True labels not provided; skipping evaluation metrics based on ground truth.")

    return cluster_labels
