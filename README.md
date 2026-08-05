# Credit Risk Assessment — Fuzzy Logic vs. Machine Learning

An end-to-end pipeline for credit scoring and default risk prediction using the
**Statlog German Credit Data** benchmark dataset.

This project experimentally compares two fundamentally different decision-making paradigms:
traditional supervised learning models (**data-driven AI**) and an interpretable
**Fuzzy Logic System** (**rule-based AI**). Additionally, it applies **Explainable AI (XAI)**
via **SHAP** to address the interpretability gap of complex ensemble models — demonstrating
that ML can meet the transparency requirements of financial institutions.

---

## Key Features

- **Automated Data Processing (`prepare_data.py`):** Translates German attribute names to
  English, applies one-hot encoding and `StandardScaler` feature scaling, and handles the
  70:30 class imbalance using **SMOTE** on the training set only.
- **Supervised Machine Learning (`ml_models.py`, `tune.py`):** Four classifiers — Random
  Forest, Logistic Regression, Decision Tree, and thresholded Linear Regression. Random
  Forest is automatically tuned via `GridSearchCV` with `f1_weighted` scoring and 5-fold
  cross-validation.
- **Fuzzy Logic Decision System (`fuzzy_logic_system.py`):** A Mamdani fuzzy inference
  system built with `scikit-fuzzy`. Evaluates credit risk across 6 configurations combining
  2–3 input attributes, 3/5/7 linguistic intervals, and `trimf` / `gaussmf` / `trapmf`
  membership function shapes.
- **Explainable AI (`xai.py`):** SHAP `TreeExplainer` generates a global summary plot for
  the tuned Random Forest, identifying `checking_status`, `savings_status`, and `duration`
  as the three most influential credit risk drivers across all 20 features.
- **Benchmark & Visualizations (`evaluate.py`, `plots.py`):** Compares all models on
  Accuracy, F1-Score, and inference time. Exports confusion matrices and bar charts to
  `data/processed/`.

---

## Benchmark Results

| Methodology | Model / Configuration | Accuracy | F1-Score | Inference Time |
| :--- | :--- | :---: | :---: | :---: |
| **Machine Learning** | **Random Forest (tuned)** | **75.0%** | **0.749** | ~0.0033 s |
| Machine Learning | Logistic Regression | 72.0% | 0.731 | ~0.0019 s |
| Machine Learning | Decision Tree | 71.5% | 0.718 | ~0.0001 s |
| Machine Learning | Linear Regression (adapted) | 71.5% | 0.726 | ~0.0020 s |
| Fuzzy Logic | 3 attributes, 3× trimf | 71.5% | 0.664 | ~1.0798 s |
| Fuzzy Logic | 2 attributes, 3× trimf (Base) | 62.0% | 0.633 | ~0.3412 s |
| Fuzzy Logic | 2 attributes, 3× gaussmf | 62.0% | 0.633 | ~0.2816 s |
| Fuzzy Logic | 2 attributes, 3× trapmf | 62.0% | 0.633 | ~0.2504 s |
| Fuzzy Logic | 2 attributes, 5× trimf | 59.5% | 0.612 | ~0.9107 s |
| Fuzzy Logic | 2 attributes, 7× trimf | 62.0% | 0.633 | ~2.2599 s |

> **Key Insight:** Increasing membership functions from 3 to 7 caused a 6.6× increase in
> execution time with zero accuracy gain — a direct consequence of exponential rule-base
> growth (k^n rules for n attributes and k membership functions each).

---

## Project Structure

```text
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv    # Original dataset
│   └── processed/                    # Generated plots, scaled data, saved model
│
├── src/
│   ├── prepare_data.py               # Loading, encoding, scaling, SMOTE
│   ├── ml_models.py                  # ML classifier definitions
│   ├── tune.py                       # GridSearchCV tuning for Random Forest
│   ├── fuzzy_logic_system.py         # Mamdani fuzzy controller with rule generator
│   ├── evaluate.py                   # Accuracy, F1-score, and timing logic
│   ├── plots.py                      # Benchmark charts and confusion matrices
│   └── xai.py                        # SHAP summary plot generation
│   
├── config.py                         # Paths, seeds, selected features
├── main.py                           # Experiment entry point
├── requirements.txt
└── raport_MSiD.pdf                   # Full academic report (Polish)
```

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data & Preprocessing** | pandas, NumPy, scikit-learn (StandardScaler), imbalanced-learn (SMOTE) |
| **ML Models** | scikit-learn (RandomForest, LogisticRegression, DecisionTree, LinearRegression) |
| **Fuzzy Logic** | scikit-fuzzy (Mamdani architecture) |
| **Explainability** | shap (TreeExplainer) |
| **Visualizations** | matplotlib, seaborn |
| **Model Persistence** | joblib |

---

## Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This will:

1. Load and preprocess the dataset (StandardScaler + SMOTE)
2. Train and evaluate all 4 ML models (Random Forest with GridSearchCV tuning)
3. Run all 6 Fuzzy Logic configurations
4. Generate a SHAP summary plot for the best ML model
5. Print a full benchmark table to stdout
6. Save all outputs to `data/processed/`

### Output Files

| File | Description |
| :--- | :--- |
| `benchmark_results.png` | Accuracy and inference time comparison charts |
| `confusion_matrices.png` | Best ML vs. best Fuzzy Logic model side-by-side |
| `shap_summary.png` | SHAP feature importance for the tuned Random Forest |
| `best_rf_model.pkl` | Saved tuned Random Forest model |

---

## Dataset

**Statlog German Credit Data** — 1,000 observations, 20 attributes covering credit history,
financial status, and customer demographics. Originally published by Prof. Hans Hofmann
(University of Hamburg).
Available via [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).