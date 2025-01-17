# -*- coding: utf-8 -*-
"""
xlnet.py

Module containing functions for XLNet-based text classification.
It sets up train/val/test splits, tokenizes data, initializes Trainer,
and evaluates the model on validation and test sets.
"""

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def run_xlnet_classification(segmented_data):
    """
    Splits data into train/val/test sets, trains XLNet model, and evaluates performance.
    """
    print("Splitting data into training, validation, and test sets...")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        segmented_data['text'],
        segmented_data['label_id'],
        test_size=0.2,
        random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # XLNet Preparation
    print("Preparing XLNet model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSequenceClassification.from_pretrained(
        'xlnet-base-cased',
        num_labels=len(segmented_data['label_id'].unique())
    )
    model.to(device)
    print("XLNet model loaded and moved to device.")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    datasets = DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'validation': Dataset.from_dict({'text': val_texts, 'label': val_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels}),
    }).map(tokenize_function, batched=True)

    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    print("Configuring training arguments...")
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
    )

    # Training
    print("Starting XLNet training...")
    trainer.train()
    print("Training completed.")

    # Validation
    print("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    # Test
    print("Predicting on test set...")
    predictions = trainer.predict(datasets['test'])
    test_pred = np.argmax(predictions.predictions, axis=-1)

    # Test Metrics
    print("Calculating test metrics...")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    test_acc = accuracy_score(datasets['test']['label'], test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        datasets['test']['label'], test_pred, average='weighted'
    )

    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1 Score: {test_f1}")
