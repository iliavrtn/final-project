# -*- coding: utf-8 -*-
"""
main.py

Main orchestration script: loads data, decides which clustering method
to use, runs the pipeline end-to-end, and trains XLNet.
"""

import os
from text_preprocess import preprocess_data
from kmeans_clustering import run_kmeans_clustering
from xlnet import run_xlnet_classification

# Global config (customize as needed)
BASE_DIR = 'G:\\.shortcut-targets-by-id\\1NUrx2vjprp18cd2ydeFPggWKmMmvP-Qf\\Capstone Project 2024-2025\\TXT_BOOKS'
VERSION = "v1.1_oxford_voc_cleaner_math_cs_updated"


def main():
    # 1. Preprocessing
    segmented_data = preprocess_data(BASE_DIR, VERSION)

    # 2. Clustering
    # In the future, you can add more strategies, e.g. "dbscan", "hierarchical", etc.
    chosen_clustering_method = "kmeans"

    if chosen_clustering_method == "kmeans":
        cluster_labels = run_kmeans_clustering(segmented_data, k=14)
        # If you like, do something with cluster_labels here.

    # 3. XLNet Classification
    #run_xlnet_classification(segmented_data)


if __name__ == '__main__':
    main()
