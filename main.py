# -*- coding: utf-8 -*-
"""
main.py

Main orchestration script: loads data, decides which clustering method
to use, runs the pipeline end-to-end, and trains XLNet.
"""

import os
from text_preprocess import preprocess_data
from kmeans_clustering import run_kmeans_clustering
from pam_clustering import run_pam_clustering_with_sampling
from dbscan_clustering import run_dbscan_clustering
from xlnet import run_xlnet_classification

# Global config (customize as needed)
BASE_DIR = 'G:\\.shortcut-targets-by-id\\1NUrx2vjprp18cd2ydeFPggWKmMmvP-Qf\\Capstone Project 2024-2025\\TXT_BOOKS'
VERSION = "v1.2_no_clean"


def main():
    # 1. Preprocessing
    segmented_data = preprocess_data(BASE_DIR, VERSION)

    # 2. Clustering
    # In the future, you can add more strategies, e.g. "dbscan", "hierarchical", etc.
    chosen_clustering_method = "kmeans"

    if chosen_clustering_method == "kmeans":
        cluster_labels = run_kmeans_clustering(segmented_data, k=14)
        # If you like, do something with cluster_labels here.
    elif chosen_clustering_method == "pam":
        # You can adjust k, distance_metric, and max_iter as needed.
        cluster_labels = run_pam_clustering_with_sampling(
            segmented_data,
            k=14,
            distance_metric='manhattan',  # or another preferred metric
            max_iter=300,
            n_components=100,
            sample_fraction=0.1
        )
        # If you like, do something with cluster_labels here.
    elif chosen_clustering_method == "dbscan":
        cluster_labels = run_dbscan_clustering(
            segmented_data,
            eps=0.5,
            min_samples=5,
            distance_metric='euclidean'  # or another metric such as 'euclidean'
        )

    # 3. XLNet Classification
    #run_xlnet_classification(segmented_data)


if __name__ == '__main__':
    main()
