# CvScanner

CvScanner is a Streamlit-based CV analysis and career guidance project.  
It analyzes CV text and predicts suitable job categories using multiple ML/NLP approaches, then shows insights and skill-gap suggestions in an interactive dashboard.

## Project summary

The app provides:
- CV upload and text extraction (TXT/PDF)
- NLP-based category similarity scoring
- Deep learning classification for career path prediction
- Traditional ML-based prediction from skills text
- EDA visualizations for category/skills distribution

Main app entry point: `/home/runner/work/CvScanner/CvScanner/app.py`

## Models used

The project currently combines three model families:

1. **Deep Learning (Transformers + PyTorch)**
   - Custom classifier head on top of Hugging Face `AutoModel`
   - Default base model in training: `bert-base-uncased`
   - Quick-train helper uses: `distilbert-base-uncased`
   - Saved checkpoint used by app: `models/best_cv_classifier.pth`

2. **NLP Similarity Model (TF-IDF + Cosine Similarity)**
   - `TfidfVectorizer` with n-grams and stopword filtering
   - Category ranking through cosine similarity
   - Used for top matching categories and missing keyword suggestions

3. **Classical ML Model (Scikit-learn)**
   - `TfidfVectorizer` + `LinearSVC` pipeline
   - Predicts career category from provided skills/text

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/AbdelrahmanMohmmed/CvScanner.git
cd CvScanner
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the app
```bash
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`).

## Notes

- The app auto-creates the `models/` directory if it does not exist.
- If `models/best_cv_classifier.pth` is missing, it is downloaded automatically from Google Drive on first run.
