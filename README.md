# Hospital Readmission Risk Predictor

**Author:** Eileen Erkan — 20956743  
**Courses:** MSE 261 · MSE 331 · MSE 446 · MSE 434

Predicts 30-day hospital readmission risk for diabetic patients using the UCI Diabetes 130-US Hospitals dataset (100k+ patient encounters). Includes a full ML pipeline and an interactive Streamlit web app for clinical use.

---

## Project Structure

```
├── hospital_readmission_project.ipynb   # Full ML pipeline (EDA → model → evaluation)
├── app.py                               # Streamlit web app (patient risk scorer)
├── diabetic_data.csv                    # UCI Diabetes 130 dataset
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 2. Run the notebook

Open `hospital_readmission_project.ipynb` in Jupyter or VS Code and run all cells.  
The notebook reads `diabetic_data.csv` from the project root automatically.

```bash
jupyter notebook hospital_readmission_project.ipynb
```

### 3. Run the web app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

The app works in two modes:
- **With `model.joblib`** — uses the trained Random Forest from the notebook (run the notebook first to generate this file)
- **Without `model.joblib`** — falls back to a logistic scoring approximation, still fully functional

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
| EDA | Fare distribution, demand curves, competition effects |
| Feature Engineering | Drug change count, A1C flags, age encoding |
| Modelling | Random Forest classifier with cross-validation |
| Evaluation | ROC-AUC, precision-recall, feature importance |

---

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
The `diabetic_data.csv` file is included in this repo for reproducibility.
