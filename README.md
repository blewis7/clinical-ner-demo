# Clinical Document Understanding (Synthetic)

Section Classification + Named Entity Recognition

This project demonstrates an end-to-end NLP pipeline for extracting structured information from unstructured clinical notes using Transformer-based models.

The system performs two tasks:

1. Line-level section classification
   Classifies each line of a clinical note into one of:

   - HPI
   - MEDS
   - ASSESSMENT
   - PLAN

2. Token-level Named Entity Recognition (NER)
   Extracts structured clinical entities using BIO tagging:
   - MED
   - DOSE
   - ROUTE
   - FREQ
   - CONDITION
   - PROCEDURE

All data used in this project is synthetic and intentionally clean to focus on pipeline design, modeling mechanics, and system clarity.

## PROJECT ARCHITECTURE

High-level pipeline:

Synthetic Data Generation  
↓  
Preprocessing & Label Alignment  
↓  
Model Training  
↓  
Inference Pipeline  
↓  
Visualization (Streamlit)

Each stage is implemented as a separate, reusable component.

## PIPELINE BREAKDOWN

### 1. Data Generation Pipeline

Purpose:  
Create realistic, labeled clinical notes without using real patient data.

Script:  
src/data_prep/make_synthetic_notes.py

Output:

- JSONL files containing synthetic clinical notes
- Each note includes section labels, tokenized text, and BIO-formatted NER tags

Why this step exists:

- Avoids PHI concerns
- Enables fast iteration
- Keeps focus on ML pipeline correctness

### 2. Preprocessing Pipeline

Purpose:  
Normalize raw synthetic notes into a format suitable for model training.

Script:  
src/data_prep/preprocess_with_pandas.py

Responsibilities:

- Load raw JSONL notes
- Split notes into individual lines
- Preserve section labels
- Align token-level BIO tags
- Convert data into Hugging Face Dataset format

Output structure:  
data/processed/  
├── ner/  
└── sections/

### 3. Training Pipelines

#### A. NER Training Pipeline

Script:  
src/models/train_ner.py

Steps:  
Load processed dataset  
→ Tokenize with BERT tokenizer  
→ Align labels with subword tokens  
→ Fine-tune BERT for token classification
→ Evaluate (F1 score)
→ Save trained model

Model type:
AutoModelForTokenClassification

#### B. Section Classification Training Pipeline

Script:
src/models/train_sections.py

Steps:  
Load processed dataset  
→ Tokenize line text  
→ Fine-tune BERT for sequence classification  
→ Evaluate (accuracy / macro-F1)  
→ Save trained model

Model type:  
AutoModelForSequenceClassification

### 4. Inference Pipeline

Script:  
src/inference/infer.py

Steps (per note):  
Raw clinical text  
→ Split into lines  
→ Section classifier (per line)  
→ NER model (per line)  
→ Post-processing (merge subword spans)  
→ Structured output

### 5. Visualization Pipeline (Streamlit App)

Script:  
app/app.py

Purpose:  
Interactive demo of the inference pipeline with human-readable outputs.

Features:

- Color-coded entity highlights by type
- Section predictions with confidence scores
- Entity legend for interpretability
- Configurable confidence thresholds

## SETUP

python -m venv .venv  
source .venv/bin/activate (Windows: .venv\Scripts\activate)  
pip install -r requirements.txt

## RUN ORDER

python src/data_prep/make_synthetic_notes.py  
python src/data_prep/preprocess_with_pandas.py  
python src/models/train_ner.py  
python src/models/train_sections.py  
streamlit run app/app.py

## DESIGN NOTES

- This project focuses on pipeline correctness, not clinical reasoning
- Models perform pattern recognition, not diagnosis or treatment decisions
- Subword tokenization artifacts are expected and handled via post-processing
- The architecture mirrors real-world NLP systems used in enterprise settings
