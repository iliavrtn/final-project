# Metalanguage as an Interdisciplinary Classifier for Mathematics and Computer Science

This repository contains all materials and code related to our capstone project, which investigates whether **math and computer science** texts can still be differentiated after removing domain-specific vocabulary. We aim to discover if a “metalanguage” remains that is strong enough for classification and clustering across various subfields.

---

## Repository Structure

```plaintext
.
├── PhaseA
│   ├── PhaseA_Book.docx
│   └── PhaseA_Presentation.pptx
│   
├── PhaseB
│   ├── PhaseB_Book.docx
│   ├── PhaseB_Presentation.pptx
│   ├── User_Guide.docx
│   └── Maintenance_Guide.docx
│   
├── PDF_to_TXT_pipeline.ipynb
│   
└── classification_pipeline.ipynb
│   
└── README.md
```

### **PhaseA**
- **Book** and **Presentation** describing our initial project proposal, literature review, and the scope of expected work.

### **PhaseB**
- **Book** and **Presentation** detailing our final methodology, experiments, and findings.
- **User Guide** explaining how to run the pipelines and replicate our experiments.
- **Maintenance Guide** offering instructions for updating dependencies, troubleshooting, and extending the system.

### **Code**  
1. **PDF_to_TXT_pipeline.ipynb**  
   - Scripts for **converting PDFs to raw text** using libraries such as PyMuPDF and Tesseract (for scanned PDFs).

2. **classification_pipeline.ipynb**  
   - **Preprocessing** (lemmatization, domain-term removal, chunking).  
   - **Classification** (XLNet fine-tuning on the preprocessed text).  
   - **Clustering** (Doc2Vec embeddings fed into K-Means, PAM, DBSCAN, and Gaussian Mixture).

---

## Project Goal

Many students find mathematics and computer science challenging, especially because each discipline has its own specialized terminology. In this project, we **strip away domain-specific words** (e.g., “matrix,” “stack,” “derivative”) to see if the remaining general language—“metalanguage”—still contains enough clues to classify the text by field or subfield. Our main goals are:

1. **Evaluate classification** performance using **XLNet** on text that lacks obvious “giveaway” terms.  
2. **Apply clustering algorithms** to see if unsupervised methods replicate domain or subfield distinctions without key terminology.  
3. **Investigate** whether mathematics and computer science exhibit a shared “metalanguage,” or if subfields (e.g., *Set Theory* vs. *Abstract Algebra*) retain enough unique style and structure to remain distinct once domain-specific words are removed.

---

## Pipeline Overview

1. **PDF Conversion**  
   - Extract text from both digital and scanned PDFs.
2. **Text Preprocessing**  
   - Remove irrelevant pages (TOC, references), convert to lowercase, lemmatize, filter out domain-specific words.
3. **Segmentation**  
   - Split the cleaned text into chunks.
4. **Classification with XLNet**  
   - Fine-tune a transformer model on these chunks to see how well it distinguishes math vs. CS, as well as subfields.
5. **Clustering**  
   - Convert chunks to **Doc2Vec** embeddings.  
   - Run **K-Means**, **PAM**, **DBSCAN**, and **GMM** to see how unsupervised clusters align with known labels.

---

## Main Findings

- **Classification**: XLNet achieved ~**83% accuracy**, showing that even after domain-specific terms are removed, each discipline still has distinctive linguistic cues.  
- **Clustering**: Unsupervised methods tend to split the data into **broad Math vs. CS** clusters (e.g., K=2), but subfields often overlap, indicating shared language structures within each domain.  
- **Conclusion**: A persistent “metalanguage” remains, yet distinguishing closely related subfields (like Calculus vs. Probability) is more challenging without specialized terms.

---

For detailed instructions, see the **User Guide** in the `PhaseB` folder.

---
