# Hospital Readmission Risk Predictor

**Author:** Eileen Erkan — 20956743  

Predicts 30-day hospital readmission risk for diabetic patients using the UCI Diabetes 130-US Hospitals dataset (100k+ patient encounters). Includes a full ML pipeline and an interactive Streamlit web app for clinical use.

---

## Project Structure

```
├── hospital_readmission_project.ipynb   # Full ML pipeline (EDA → model → evaluation)
├── app.py                               # Streamlit web app (patient risk scorer)
├── diabetic_data.csv                    # UCI Diabetes 130 dataset
├── model.joblib                         # Trained Random Forest model
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Clone the repo and install dependencies

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Run the web app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> The app loads `model.joblib` automatically. No need to run the notebook first.

### 3. (Optional) Re-run the notebook

To retrain the model from scratch:

```bash
jupyter notebook hospital_readmission_project.ipynb
```

Run all cells top-to-bottom. The data is read from `diabetic_data.csv` in the project root.

---

## App Features

Enter patient discharge information to get:
- **Readmission probability** (0–100%)
- **Risk level** — Low / Medium / High
- **Clinical recommendation** — tailored intervention based on risk tier
- **Key risk drivers** — top factors contributing to this patient's score

---

## ML Pipeline (Notebook)

| Step | Description |
|------|-------------|
| Load & Clean | 100k+ rows, handle missing values, remove deaths/hospice discharges |
| EDA | Readmission rates by age, diagnosis count, medication use |
| Feature Engineering | Drug change count, A1C flags, age band encoding |
| Modelling | Random Forest classifier with cross-validation |
| Evaluation | ROC-AUC, precision-recall, feature importance |

---

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
The `diabetic_data.csv` file is included in this repo for reproducibility.
