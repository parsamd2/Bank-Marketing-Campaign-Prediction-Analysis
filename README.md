# Bank Marketing Campaign Prediction Analysis

A machine-learning study on direct telephone-marketing data from a Portuguese bank, exploring which clients are most likely to subscribe to a term deposit product. The project builds, compares, and interprets several classification models and concludes with customer-segment profiling and actionable campaign recommendations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Methodology](#methodology)
   - [Data Exploration](#1-data-exploration)
   - [Preprocessing](#2-preprocessing)
   - [Baseline Logistic Regression](#3-baseline-logistic-regression)
   - [Hyperparameter Optimisation](#4-hyperparameter-optimisation)
   - [Regularised Models (L1 / L2)](#5-regularised-models-l1--l2)
   - [Model Comparison](#6-model-comparison)
   - [Decision Threshold Tuning](#7-decision-threshold-tuning)
   - [K-Nearest Neighbours](#8-k-nearest-neighbours)
   - [Client Segmentation](#9-client-segmentation)
5. [Key Findings](#key-findings)
6. [Business Recommendations](#business-recommendations)
7. [How to Run](#how-to-run)
8. [Dependencies](#dependencies)

---

## Project Overview

Phone-based marketing campaigns are resource-intensive; calling the wrong clients wastes agent time and annoys customers. This project trains predictive models on historical campaign data to:

- Score each client's likelihood of subscribing to a term deposit
- Identify which client characteristics drive subscriptions
- Recommend an optimal operating threshold to balance precision and recall according to the campaign's business goal

---

## Dataset

| Property | Value |
|---|---|
| **File** | `bank-additional-full.csv` |
| **Source** | UCI Machine Learning Repository – Bank Marketing dataset |
| **Observations** | 41,188 |
| **Features** | 20 input features + 1 target (`y`) |
| **Target** | `y` – whether the client subscribed to a term deposit (`yes` / `no`) |
| **Class balance** | ~11 % positive (`yes`), ~89 % negative (`no`) |
| **Separator** | Semicolon (`;`) |

**Feature groups:**

- *Client demographics* – `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- *Last contact information* – `contact`, `month`, `day_of_week`, `duration`
- *Campaign statistics* – `campaign`, `pdays`, `previous`, `poutcome`
- *Social / economic context* – `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

---

## Repository Structure

```
├── bank-additional-full.csv          # Raw dataset
├── bank_marketing_analysis.ipynb     # Main analysis notebook (all models and visuals)
└── README.md                         # This file
```

---

## Methodology

### 1. Data Exploration

Initial inspection covers:

- Dataset shape, column types, and missing-value audit (the dataset uses `"unknown"` strings rather than `NaN`; these are handled during preprocessing)
- Class-balance visualisation – the strong imbalance (~11 % positive) justifies using F1-score and ROC-AUC as primary metrics rather than raw accuracy
- Subscription rate by age group and job category to motivate segment-level analysis

### 2. Preprocessing

A `scikit-learn` `Pipeline` with a `ColumnTransformer` is applied consistently to every model:

| Step | Numeric features | Categorical features |
|---|---|---|
| Imputation | Median | Most-frequent value |
| Encoding | `StandardScaler` | `OneHotEncoder` (unknown categories ignored) |

An 80 / 20 stratified train / test split (random state 42) is used throughout, ensuring the class distribution is preserved in both subsets.

### 3. Baseline Logistic Regression

A default `LogisticRegression` (L2 penalty, `lbfgs` solver, `max_iter=1000`) is trained as a reference point.

Evaluation metrics reported:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix
- Full classification report

### 4. Hyperparameter Optimisation

`GridSearchCV` (5-fold stratified CV, optimising F1) searches over:

| Parameter | Values searched |
|---|---|
| `C` | 0.01, 0.1, 1, 10 |
| `solver` | `liblinear`, `lbfgs` |
| `max_iter` | 500, 1000 |
| `class_weight` | `None`, `balanced` |

A visualisation plots mean CV-F1 at each ranked optimisation step, making it easy to see how quickly performance converges. The best configuration is compared to the baseline to quantify improvement.

### 5. Regularised Models (L1 / L2)

Two dedicated models are trained under identical evaluation conditions:

| Model | Penalty | Solver | C |
|---|---|---|---|
| Lasso (L1) | `l1` | `liblinear` | 1.0 |
| Ridge (L2) | `l2` | `lbfgs` | 1.0 |

L1 regularisation produces sparser coefficients (some driven to exactly zero), which can improve interpretability when the feature space is large after one-hot encoding. L2 regularisation distributes the penalty more evenly, which often yields smoother generalisation when many correlated features are present.

### 6. Model Comparison

All logistic models (baseline, optimised, L1, L2) are compared side-by-side:

- Tabular summary of all metrics
- Overlaid ROC curves (AUC annotated)
- Three-panel confusion-matrix grid

The comparison highlights trade-offs between accuracy-focused and recall-focused models, and identifies which configuration best suits a marketing campaign where missing a true subscriber (false negative) is often more costly than making an unnecessary call (false positive).

### 7. Decision Threshold Tuning

Rather than fixing the decision boundary at 0.5, the chosen logistic model is evaluated at:

`[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`

A multi-line plot shows how Accuracy, Precision, Recall, and F1 change with the threshold, making the precision–recall trade-off visually clear. The optimal threshold is selected to maximise F1, with a discussion of how the threshold choice should reflect the specific campaign budget and call-centre capacity.

### 8. K-Nearest Neighbours

KNN is trained and compared as an alternative to logistic regression:

1. **K tuning** – K is swept from 3 to 50 using 5-fold stratified CV; a plot of mean F1 vs. K identifies the optimal neighbourhood size
2. **Test-set evaluation** – the best-K model is evaluated with the same metric suite
3. **Head-to-head comparison** across three dimensions:

| Dimension | Logistic Regression | KNN |
|---|---|---|
| Stored parameters | Coefficient vector (one weight per feature + intercept) | All *N* training points × feature dimension |
| Training time | Fast (iterative optimisation) | Effectively instant (no fitting step) |
| Prediction time | O(1) | O(N × D) per query |
| Interpretability | High (coefficients) | Low (instance-based, no explicit model) |

The analysis explains why logistic regression typically generalises better on high-cardinality one-hot encoded data and scales more comfortably to production volumes.

### 9. Client Segmentation

Using the best logistic model's learned coefficients:

- The top-20 most influential features (positive and negative) are extracted and visualised
- Clients are segmented by age group × job category; each segment's subscription rate is computed and ranked
- The top segments by predicted likelihood are profiled to build actionable customer personas

---

## Key Findings

| Finding | Detail |
|---|---|
| **Class imbalance matters** | With only ~11 % positives, accuracy is a misleading metric. F1 and ROC-AUC are used throughout. |
| **Optimisation gains are modest** | Grid search improved F1 marginally over the baseline because the default regularisation is already well-suited to this data. |
| **L1 vs L2** | Both regularised variants perform comparably; L1 yields sparser coefficients (useful for feature selection), L2 generalises slightly more stably across CV folds. |
| **Threshold matters** | Lowering the decision threshold (e.g., to 0.3) substantially increases recall at the cost of precision — the right choice depends on campaign budget. |
| **KNN is competitive but less scalable** | KNN reaches comparable F1 at its optimal K but requires O(N) memory and O(N × D) prediction time, making it impractical for large real-time scoring. |
| **High-value segments** | Retired clients and students tend to show higher subscription rates; economic context variables (`euribor3m`, `emp.var.rate`) are among the strongest model predictors. |

---

## Business Recommendations

1. **Prioritise by score** – Rank all clients by predicted subscription probability and focus outreach on the top deciles to maximise ROI per call.

2. **Choose the threshold strategically** – If the call centre has capacity, use a lower threshold (e.g., 0.3) to capture more true subscribers. If calls are costly or agents are limited, raise the threshold (e.g., 0.5–0.6) for higher precision.

3. **Target identified segments** – Retired and student segments, as well as clients previously contacted with successful outcomes (`poutcome = success`), show disproportionately high subscription rates and should be prioritised.

4. **Monitor economic indicators** – Variables such as `euribor3m` and `emp.var.rate` are strong predictors. Scheduling campaigns when rates are favourable can improve the overall positive rate and make models more effective.

5. **Re-train periodically** – Economic conditions and client behaviour shift over time. Retrain the model at least quarterly and monitor live performance metrics (precision, recall) against the deployed threshold.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/parsamd2/Bank-Marketing-Campaign-Prediction-Analysis.git
cd Bank-Marketing-Campaign-Prediction-Analysis

# 2. Install dependencies
pip install -r requirements.txt   # or see the Dependencies section below

# 3. Launch the notebook
jupyter notebook bank_marketing_analysis.ipynb
```

Run all cells from top to bottom. The notebook is self-contained: it reads `bank-additional-full.csv` from the current directory, preprocesses the data, trains every model, and renders all plots inline.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Modelling, preprocessing, grid search, metrics |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `jupyter` / `nbformat` | Notebook environment |

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

*Analysis performed with scikit-learn, using a fixed random state (42) throughout for full reproducibility.*
