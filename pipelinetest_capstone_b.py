# -*- coding: utf-8 -*-
"""
PipelineTEST - Capstone B

This script processes text data for domain classification, using XLNet for text classification.
Now includes functionality for saving and loading preprocessed files based on a specified version.
"""

import os
import nltk
import re
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import confusion_matrix

######## TEXT PREPROCESSING START ##########################################################################################
# Base directory for files (provide path with duplicate backslashes '\\')
BASE_DIR = 'G:\\.shortcut-targets-by-id\\1NUrx2vjprp18cd2ydeFPggWKmMmvP-Qf\\Capstone Project 2024-2025\\TXT_BOOKS'

# Versioning
VERSION = "v1.1_oxford_voc_cleaner_math_cs_updated"  # You can change this version number as needed
PROCESSED_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), f"TXT_BOOKS_{VERSION}")

# Paths to domain dictionaries
CS_DICT_PATH = os.path.join(BASE_DIR, 'cs.txt')
MATH_DICT_PATH = os.path.join(BASE_DIR, 'math.txt')
OXFORD_DICT_PATH = os.path.join(BASE_DIR, 'oxford.txt')

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
print("NLTK resources downloaded.")

lemmatizer = WordNetLemmatizer()


def main():
    # Helper Functions
    def get_wordnet_pos(treebank_tag):
        """Map POS tag to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def preprocess_text(file_path, domain_terms, oxford_words):
        """Process a text file to extract lemmatized terms."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: '{file_path}' not found.")
            return None

        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        text = ' '.join(word for word in text.split() if len(word) > 1)
        text = re.sub(r'\s+', ' ', text).strip()

        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        lemmatized_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
        ]

        lemmatized_text = ' '.join(lemmatized_words)

        onegram = list(ngrams(lemmatized_words, 1))
        bigrams = list(ngrams(lemmatized_words, 2))
        trigrams = list(ngrams(lemmatized_words, 3))

        onegram_oxford_strings_to_delete = {
            " ".join(onegram) for onegram in onegram if " ".join(onegram) not in oxford_words
        }

        trigram_domain_strings_to_delete = {
            " ".join(trigram) for trigram in trigrams if " ".join(trigram) in domain_terms
        }
        bigram_domain_strings_to_delete = {
            " ".join(bigram) for bigram in bigrams if " ".join(bigram) in domain_terms
        }
        onegram_domain_strings_to_delete = {
            " ".join(onegram) for onegram in onegram if " ".join(onegram) in domain_terms
        }

        all_domain_phrases = trigram_domain_strings_to_delete | bigram_domain_strings_to_delete | onegram_domain_strings_to_delete



        def replace_phrases(text, phrases, replacer):
            phrases = sorted(phrases, key=lambda x: len(x.split()), reverse=True)
            pattern = r'\b(' + r'|'.join(re.escape(phrase) for phrase in phrases) + r')\b'
            return re.sub(pattern, replacer, text)

        lemmatized_text = re.sub(r'\s+', ' ', replace_phrases(lemmatized_text, all_domain_phrases, '<TERM>')).strip()

        return re.sub(r'\s+', ' ', replace_phrases(lemmatized_text, onegram_oxford_strings_to_delete, '')).strip()

    # Read and process dictionaries
    print("Processing dictionaries...")
    with open(CS_DICT_PATH, 'r', encoding='utf-8') as f:
        cs_words = [term.lower() for term in f.read().splitlines()]

    with open(MATH_DICT_PATH, 'r', encoding='utf-8') as f:
        math_words = [term.lower() for term in f.read().splitlines()]

    with open(OXFORD_DICT_PATH, 'r', encoding='utf-8') as f:
        oxford_words = [term.lower() for term in f.read().splitlines()]

    lemmatized_cs_words = {
        " ".join(
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tag(word_tokenize(cs_word))
        )
        for cs_word in cs_words
    }
    print("Processed CS dictionary.")

    lemmatized_math_words = {
        " ".join(
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tag(word_tokenize(math_word))
        )
        for math_word in math_words
    }
    print("Processed MATH dictionary.")

    lemmatized_oxford_words = {
        " ".join(
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tag(word_tokenize(oxford_word))
        )
        for oxford_word in oxford_words
    }
    print("Processed OXFORD dictionary.")

    # Process categories
    print("Processing text categories...")
    CATEGORIES = {
        "CS": ["AUTOMATA", "COMPILER", "OOP", "OS", "IMPPROG", "FPROG", "DSA"],
        "MATH": ["LINALG", "ABSALG", "CALC", "COMBI", "PROB", "LOGIC", "SETTHEORY"]
    }

    all_texts, all_labels = [], []

    # If the processed version folder exists, we use preprocessed files from there
    use_preprocessed = os.path.exists(PROCESSED_BASE_DIR)

    if use_preprocessed:
        print(f"Version '{VERSION}' found. Using preprocessed files...")
    else:
        print(f"Version '{VERSION}' not found. Creating and preprocessing files...")
        os.makedirs(PROCESSED_BASE_DIR, exist_ok=True)

    for category, fields in CATEGORIES.items():
        for field in fields:
            original_folder_path = os.path.join(BASE_DIR, category, field)
            processed_folder_path = os.path.join(PROCESSED_BASE_DIR, category, field)

            # Create processed folder if not exist
            if not use_preprocessed:
                os.makedirs(processed_folder_path, exist_ok=True)

            text_files = glob.glob(os.path.join(original_folder_path, "*.txt"))
            domain_terms = lemmatized_cs_words if category == "CS" else lemmatized_math_words

            for file_path in text_files:
                processed_file_path = os.path.join(processed_folder_path, os.path.basename(file_path))

                if use_preprocessed:
                    # Load from preprocessed file
                    if not os.path.exists(processed_file_path):
                        print(f"Preprocessed file {processed_file_path} not found, skipping.")
                        continue
                    with open(processed_file_path, 'r', encoding='utf-8') as pf:
                        processed_text = pf.read().strip()
                else:
                    # Preprocess and save
                    print(f"Processing file: {file_path}")
                    processed_text = preprocess_text(file_path, domain_terms, lemmatized_oxford_words)
                    if processed_text:
                        with open(processed_file_path, 'w', encoding='utf-8') as pf:
                            pf.write(processed_text)

                if processed_text:
                    all_texts.append(processed_text)
                    all_labels.append(f"{category}_{field}")

    print("Creating DataFrame...")
    data = pd.DataFrame({'text': all_texts, 'label': all_labels})
    data['label_id'] = LabelEncoder().fit_transform(data['label'])
    print(f"DataFrame created with {len(data)} samples.")

    # Segment texts
    print("Segmenting texts...")

    def chunk_text(text, max_length):
        words = text.split()
        return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

    segmented_texts, segmented_labels = [], []
    for idx, row in data.iterrows():
        segments = chunk_text(row['text'], max_length=510)
        segmented_texts.extend(segments)
        segmented_labels.extend([row['label_id']] * len(segments))
    print(f"Segmented {len(segmented_texts)} text samples.")

    segmented_data = pd.DataFrame({'text': segmented_texts, 'label_id': segmented_labels})

    ######## TEXT PREPROCESSING END ##########################################################################################

    # ===== Start Clustering Code Here =====
    texts = segmented_data['text'].tolist()
    true_labels = segmented_data['label_id'].tolist()

    # Step 1: Vectorize the text data with TF-IDF
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # consider unigrams and bigrams
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    print("Vectorization complete. Shape of TF-IDF matrix:", X.shape)

    # Step 2: Apply K-Means Clustering
    k = 14  # We have 14 fields in total
    print(f"Clustering into {k} clusters using K-Means...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    print("Clustering complete.")

    # Print top terms per cluster to understand them better
    print("\nTop terms per cluster:")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Cluster {i}: {', '.join(top_terms)}")

    # Step 3: Dimensionality Reduction for Visualization
    print("Performing dimensionality reduction for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    print("Dimensionality reduction complete. Plotting the results...")

    # Step 4: Plot the clusters
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

    # Create a confusion matrix (counts how many samples of each true class fall into each cluster)
    cm = confusion_matrix(true_labels, cluster_labels)
    cm_df = pd.DataFrame(cm, index=[f"Class_{c}" for c in sorted(set(true_labels))], columns=[f"Cluster_{c}" for c in range(len(set(cluster_labels)))])

    print("\nConfusion Matrix (True Class vs. Cluster):")
    print(cm_df)

    # ===== End Clustering Code =====

    # ===== Start XLNet text classification =====

    # Train/Validation/Test Split
    print("Splitting data into training, validation, and test sets...")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        segmented_data['text'], segmented_data['label_id'], test_size=0.2, random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # XLNet Preparation
    print("Preparing XLNet model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',
                                                           num_labels=len(data['label_id'].unique()))
    model.to(device)
    print("XLNet model loaded and moved to device.")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length', truncation=True, max_length=512
        )

    datasets = DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'validation': Dataset.from_dict({'text': val_texts, 'label': val_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels}),
    }).map(tokenize_function, batched=True)

    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print("Datasets tokenized and formatted.")

    # Trainer Setup
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    print("Training arguments configured.")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized.")

    # Training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Evaluation
    print("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    # Test Prediction
    print("Predicting on test set...")
    predictions = trainer.predict(datasets['test'])
    test_pred = np.argmax(predictions.predictions, axis=-1)

    # Test Metrics
    print("Calculating test metrics...")
    test_acc = accuracy_score(test_labels, test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_pred, average='weighted'
    )

    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1 Score: {test_f1}")

    # ===== End XLNet text classification =====


if __name__ == "__main__":
    main()
