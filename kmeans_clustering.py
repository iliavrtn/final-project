# -*- coding: utf-8 -*-
"""
kmeans_clustering.py

Contains the K-Means clustering logic: vectorization, clustering,
PCA for visualization, and evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import (
    adjusted_rand_score, adjusted_mutual_info_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)


def run_kmeans_clustering(segmented_data, k=14):
    """
    Runs K-Means clustering on the given segmented DataFrame
    and visualizes the results via PCA (2D).

    Returns:
        cluster_labels (ndarray): Cluster assignments for each segment
    """
    texts = segmented_data['text'].tolist()
    true_labels = segmented_data['label_id'].tolist()

    # Step 1: Vectorize the text data with TF-IDF
    print("Vectorizing text data for K-Means...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    print("Vectorization complete. Shape of TF-IDF matrix:", X.shape)

    # Step 2: K-Means Clustering
    print(f"Clustering into {k} clusters using K-Means...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    print("K-Means clustering complete.")

    # Print top terms per cluster
    print("\nTop terms per cluster:")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Cluster {i}: {', '.join(top_terms)}")

    # Step 3: PCA for Visualization
    print("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    print("Dimensionality reduction complete. Plotting K-Means clusters...")

    # Plot Clusters (Predicted)
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
    plt.title("K-Means Clusters of Text Data (PCA 2D Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster', loc='best')
    plt.show()

    # Plot Actual Labels
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

    # Step 4: Compute Clustering Evaluation Metrics
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    hom = homogeneity_score(true_labels, cluster_labels)
    com = completeness_score(true_labels, cluster_labels)
    v = v_measure_score(true_labels, cluster_labels)

    print("\nClustering Evaluation Metrics:")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Homogeneity: {hom:.4f}")
    print(f"Completeness: {com:.4f}")
    print(f"V-measure: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Class_{c}" for c in sorted(set(true_labels))],
        columns=[f"Cluster_{c}" for c in range(len(set(cluster_labels)))]
    )
    print("\nConfusion Matrix (True Class vs. Cluster):")
    print(cm_df)

    return cluster_labels
