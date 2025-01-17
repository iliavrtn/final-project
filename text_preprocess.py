# -*- coding: utf-8 -*-
"""
text_preprocessing.py

This module handles all text data processing: reading text files,
cleaning, lemmatizing, segmenting, and preparing a final DataFrame
suitable for clustering and classification.
"""

import os
import re
import glob
import nltk
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    Map POS tag to WordNet POS tags.
    """
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
    """
    Process a text file to extract lemmatized terms.
    Returns processed text or None if the file doesn't exist.
    """
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
        " ".join(g) for g in onegram if " ".join(g) not in oxford_words
    }

    trigram_domain_strings_to_delete = {
        " ".join(t) for t in trigrams if " ".join(t) in domain_terms
    }
    bigram_domain_strings_to_delete = {
        " ".join(b) for b in bigrams if " ".join(b) in domain_terms
    }
    onegram_domain_strings_to_delete = {
        " ".join(g) for g in onegram if " ".join(g) in domain_terms
    }

    all_domain_phrases = (trigram_domain_strings_to_delete
                          | bigram_domain_strings_to_delete
                          | onegram_domain_strings_to_delete)

    def replace_phrases(txt, phrases, replacer):
        phrases = sorted(phrases, key=lambda x: len(x.split()), reverse=True)
        pattern = r'\b(' + r'|'.join(re.escape(phrase) for phrase in phrases) + r')\b'
        return re.sub(pattern, replacer, txt)

    lemmatized_text = re.sub(
        r'\s+',
        ' ',
        replace_phrases(lemmatized_text, all_domain_phrases, '<TERM>')
    ).strip()

    final_text = re.sub(
        r'\s+',
        ' ',
        replace_phrases(lemmatized_text, onegram_oxford_strings_to_delete, '')
    ).strip()

    return final_text


def chunk_text(text, max_length=510):
    """
    Splits text into smaller chunks of length = max_length tokens.
    """
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]


def preprocess_data(BASE_DIR, VERSION):
    """
    Main function to orchestrate reading dictionaries, processing text,
    and returning the final segmented DataFrame.
    """
    # Derived path for processed data
    PROCESSED_BASE_DIR = os.path.join(
        os.path.dirname(BASE_DIR),
        f"TXT_BOOKS_{VERSION}"
    )

    # Dictionaries paths
    CS_DICT_PATH = os.path.join(BASE_DIR, 'cs.txt')
    MATH_DICT_PATH = os.path.join(BASE_DIR, 'math.txt')
    OXFORD_DICT_PATH = os.path.join(BASE_DIR, 'oxford.txt')

    print("Processing dictionaries...")
    with open(CS_DICT_PATH, 'r', encoding='utf-8') as f:
        cs_words = [term.lower() for term in f.read().splitlines()]

    with open(MATH_DICT_PATH, 'r', encoding='utf-8') as f:
        math_words = [term.lower() for term in f.read().splitlines()]

    with open(OXFORD_DICT_PATH, 'r', encoding='utf-8') as f:
        oxford_words = [term.lower() for term in f.read().splitlines()]

    # Lemmatize dictionary words
    print("Lemmatizing CS dictionary...")
    lemmatized_cs_words = {
        " ".join(
            lemmatizer.lemmatize(w, get_wordnet_pos(p))
            for w, p in pos_tag(word_tokenize(cs_word))
        )
        for cs_word in cs_words
    }

    print("Lemmatizing MATH dictionary...")
    lemmatized_math_words = {
        " ".join(
            lemmatizer.lemmatize(w, get_wordnet_pos(p))
            for w, p in pos_tag(word_tokenize(math_word))
        )
        for math_word in math_words
    }

    print("Lemmatizing OXFORD dictionary...")
    lemmatized_oxford_words = {
        " ".join(
            lemmatizer.lemmatize(w, get_wordnet_pos(p))
            for w, p in pos_tag(word_tokenize(oxford_word))
        )
        for oxford_word in oxford_words
    }

    # Categories
    CATEGORIES = {
        "CS": ["AUTOMATA", "COMPILER", "OOP", "OS", "IMPPROG", "FPROG", "DSA"],
        "MATH": ["LINALG", "ABSALG", "CALC", "COMBI", "PROB", "LOGIC", "SETTHEORY"]
    }

    all_texts, all_labels = [], []
    # Check if we can use preprocessed files
    use_preprocessed = os.path.exists(PROCESSED_BASE_DIR)

    if use_preprocessed:
        print(f"Version '{VERSION}' found. Using preprocessed files...")
    else:
        print(f"Version '{VERSION}' not found. Creating and preprocessing files...")
        os.makedirs(PROCESSED_BASE_DIR, exist_ok=True)

    # Process each category and field
    for category, fields in CATEGORIES.items():
        for field in fields:
            original_folder_path = os.path.join(BASE_DIR, category, field)
            processed_folder_path = os.path.join(PROCESSED_BASE_DIR, category, field)
            if not use_preprocessed:
                os.makedirs(processed_folder_path, exist_ok=True)

            text_files = glob.glob(os.path.join(original_folder_path, "*.txt"))
            domain_terms = (
                lemmatized_cs_words if category == "CS" else lemmatized_math_words
            )

            for file_path in text_files:
                processed_file_path = os.path.join(processed_folder_path,
                                                   os.path.basename(file_path))

                if use_preprocessed:
                    # Load preprocessed file
                    if not os.path.exists(processed_file_path):
                        print(f"Preprocessed file {processed_file_path} not found, skipping.")
                        continue
                    with open(processed_file_path, 'r', encoding='utf-8') as pf:
                        processed_text = pf.read().strip()
                else:
                    # Preprocess file
                    print(f"Processing file: {file_path}")
                    processed_text = preprocess_text(
                        file_path,
                        domain_terms,
                        lemmatized_oxford_words
                    )
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
    segmented_texts, segmented_labels = [], []
    for _, row in data.iterrows():
        segments = chunk_text(row['text'], max_length=256)
        segmented_texts.extend(segments)
        segmented_labels.extend([row['label_id']] * len(segments))

    print(f"Segmented {len(segmented_texts)} text samples.")
    segmented_data = pd.DataFrame({'text': segmented_texts, 'label_id': segmented_labels})

    return segmented_data
