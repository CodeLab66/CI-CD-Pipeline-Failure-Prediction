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
│   ├── preprocessing/      # Cleaning, feature engineering & encoding scripts
│   ├── models/             # Model training & evaluation
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
| **Build / Test** | `tr_duration`, `tr_log_num_tests_run`, `tr_log_num_tests_failed` |

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
