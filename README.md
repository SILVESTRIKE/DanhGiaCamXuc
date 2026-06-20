# DuDoanCamXuc - Vietnamese Sentiment Analysis Web Application

A machine learning web application designed to predict and analyze customer sentiment from Vietnamese text reviews (such as restaurant feedback). It integrates pre-trained deep learning Transformers models with underthesea for Vietnamese text preprocessing, wrapped in a Streamlit web portal.

---

## Technical Features

- Web Interface: Streamlit web dashboard.
- Natural Language Processing: underthesea library for Vietnamese word segmentation and text tokenization.
- Deep Learning Backend: PyTorch (torch) and Hugging Face Transformers.
- Visualizations: Matplotlib and Seaborn for plotting sentiment metrics, confidence ratings, and review analytics.
- Data Processing: Pandas, NumPy, and Scikit-Learn for data cleaning and evaluation metrics.

---

## File Organization

```
DuDoanCamXuc/
├── app.py                # Main Streamlit web application entrypoint
├── config.py             # Inference configurations and threshold parameters
├── main_testing.py       # Evaluation scripts and metric scoring
├── restaurants_testing.py# Script for processing restaurant dataset batches
├── utils.py              # Data helpers, tokenizers, and preprocessing pipelines
├── requirements.txt      # Project library dependencies
└── README.md             # This document
```

---

## Workflow

1. Input Review: The user enters a Vietnamese review in the Streamlit text box (or uploads a batch of reviews).
2. NLP Preprocessing: The text is normalized and segmented into proper tokens using the `underthesea` library.
3. VLM/Transformer Inference: The tokenized vector is processed by a deep learning classifier running on PyTorch.
4. Visualization: The predicted label (Positive, Neutral, Negative) and confidence scores are rendered in the dashboard using Matplotlib charts.

---

## Setup & Running

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (optional, falls back to CPU)

### 1. Installation
1. Navigate to the project directory:
   ```bash
   cd D:\DEEPLEARNING\DuDoanCamXuc
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate # Linux/macOS
   pip install -r requirements.txt
   ```

### 2. Execution
Start the Streamlit portal:
```bash
streamlit run app.py
```
The application will launch on your local host (default: http://localhost:8501).
