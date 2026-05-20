# CI/CD Pipeline Failure Prediction

A machine-learning project that predicts whether a **CI/CD build will pass or fail** based on repository metadata, git history, and Travis CI build information.

---

## Problem Statement

Continuous Integration pipelines run thousands of builds daily. Many of those builds fail, wasting developer time and compute resources. This project trains a classifier on historical build data from the **TravisTorrent** dataset to predict the build outcome (`passed` / `failed` / `errored`) *before* the build finishes — enabling teams to prioritise, cancel, or fast-track builds intelligently.

---

## Project Structure

```
CI-CD-Pipeline-Failure-Prediction/
│
├── data/                   # Raw & processed datasets (git-ignored)
│   ├── data.csv            # Original TravisTorrent dataset (~3.5 GB)
│   └── processed_cleaned.csv
│
├── notebooks/              # Jupyter notebooks
│   └── 01_Comprehensive_EDA.ipynb   # Full exploratory data analysis
│
├── src/                    # Source code modules
│   ├── pipeline/           # End-to-end preprocessing pipeline runner
│   ├── preprocessing/      # Cleaning, feature engineering & encoding scripts
│   ├── training/           # Model training & evaluation code
│   └── visualization/      # Plotting utilities
│
├── models/                 # Saved model artifacts (git-ignored)
├── app/                    # (Future) Deployment / API code
├── doc/                    # Documentation & reports
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

| Item | Detail |
|------|--------|
| **Source** | [TravisTorrent](https://travistorrent.testroots.org/) |
| **Rows** | ~3.8 million (job-level records) |
| **Columns** | 66 features covering GitHub repo info, git diffs, Travis CI logs |
| **Target** | `tr_status` — `passed`, `failed`, or `errored` |

> A 250 000-row sample is used during EDA to avoid OOM issues while retaining statistical significance.

---

## Key Features Used

| Category | Examples |
|----------|----------|
| **Repository** | `gh_team_size`, `gh_sloc`, `gh_repo_age`, `gh_lang` |
| **Git Diff** | `git_diff_src_churn`, `gh_diff_files_modified`, `gh_diff_tests_added` |
| **Code Quality** | `gh_test_lines_per_kloc`, `gh_test_cases_per_kloc` |
| **Pre-build Signals** | `gh_is_pr`, `gh_lang`, `git_prev_commit_resolution_status`, grouped `git_branch` |

> Build-log and duration fields such as `tr_duration`, `tr_log_num_tests_run`, and `tr_log_num_tests_failed` are useful for post-build analysis, but are excluded from the model-ready pre-build prediction dataset to avoid leakage.

See the **Column Data Dictionary** inside `01_Comprehensive_EDA.ipynb` for a full description of every column.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CI-CD-Pipeline-Failure-Prediction.git
cd CI-CD-Pipeline-Failure-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `data.csv` from [TravisTorrent](https://travistorrent.testroots.org/) and place it in the `data/` folder.

### 5. Run the EDA notebook

```bash
jupyter notebook notebooks/01_Comprehensive_EDA.ipynb
```

---

## ML Preprocessing Pipeline

## Interactive Frontend

Run the futuristic Flask dashboard:

```bash
python3 run_frontend.py
```

Then open:

```text
http://127.0.0.1:5000
```

Frontend files:

```text
app/app.py                 # Flask routes and API endpoints
app/model_service.py       # Loads the trained model, metrics, and prediction service
app/templates/index.html   # Dashboard layout
app/static/css/styles.css  # Futuristic responsive UI
app/static/js/app.js       # Interactive controls, API calls, live step trace
run_frontend.py            # Simple server launcher
tests/test_frontend_app.py # API and page smoke tests
```

The dashboard is linked to `models/best_raw_model.joblib` and displays the saved test accuracy, F1 score, ROC-AUC, validation model comparison, feature importance, prediction confidence, and each processing step used during inference.

Run the frontend tests:

```bash
python3 -m unittest tests/test_frontend_app.py
```

For a clean large-file training run directly from `data/data.csv`, use:

```bash
python src/training/train_from_raw.py --input data/data.csv --output-dir models --max-rows 100000
```

This reads the 3.5 GB CSV in chunks, keeps a stratified sample, uses only pre-build features, trains Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost when installed, then saves `models/best_raw_model.joblib`, `models/raw_model_metrics.csv`, `models/raw_model_report.json`, `models/raw_confusion_matrix.png`, and `models/raw_roc_curve.png`.

Run the full data engineering and model-training pipeline:

```bash
python main.py
```

To resume from a later stage:

```bash
python main.py --from-step scale_balance
```

Train only from the already processed model-ready files:

```bash
python main.py --only-step train
```

For a fast experiment while developing, use stratified samples from the large processed CSVs:

```bash
python main.py --only-step train --max-train-rows 10000 --max-eval-rows 5000
```

To run the proposal's RandomizedSearchCV tuning on the top two validation models:

```bash
python main.py --only-step train --tune --tune-iter 12
```

Training reads `data/scaled_balanced/balanced_train_data.csv` and evaluates on the untouched scaled validation/test files. It writes `models/best_model.joblib`, `models/model_metrics.csv`, `models/model_metrics.json`, ROC/confusion-matrix plots, and feature-importance artifacts.

The same steps can also be run individually:

```bash
python src/preprocessing/clean_data.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/encode.py
python src/preprocessing/split.py
python src/preprocessing/scale_balance.py
python src/training/train_model.py
python src/training/train_from_raw.py
```

The project objective is prediction before build completion, so the model-ready scaling/balancing step excludes build-log and duration fields that would leak post-build information. Categorical decisions are made from the training split only: selected pre-build categorical columns are analyzed, low-cardinality safe features are one-hot encoded, the scaler is fitted on train only, and SMOTE is applied only to the scaled training set.

---

## EDA Notebook Overview

The `01_Comprehensive_EDA.ipynb` notebook covers:

| Section | Description |
|---------|-------------|
| **1. Data Loading** | Loads a 250K-row sample, shows `info()` and `describe()` |
| **2. Column Data Dictionary** | Explains what every column represents |
| **3. Unique Values Analysis** | Cardinality, dtype, null %, sample values per column |
| **4. Missing Values** | Visualises missing data with a 90% drop threshold |
| **5. Drop Recommendations** | Auto-flags columns to remove (too sparse, constant, ID-like) |
| **6. Target Distribution** | Class balance analysis for `tr_status` |
| **7. Numerical Distributions** | Histograms + box-plots for key features |
| **8. Feature Correlations** | Heatmap + auto-detection of highly correlated pairs |
| **9. Recommendations** | Actionable next steps for cleaning, engineering & modelling |

---

## Tech Stack

- **Python 3.10+**
- **pandas** / **NumPy** – data wrangling
- **Matplotlib** / **Seaborn** – visualisation
- **scikit-learn** – machine learning
- **imbalanced-learn** – SMOTE for class imbalance
- **Jupyter** – interactive notebooks

---

## License

This project is for academic / educational purposes.
