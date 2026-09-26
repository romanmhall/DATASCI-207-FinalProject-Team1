# DATASCI 207 Final Project: Team Tasklist

**Project Title**: Behavioral Dynamics and Alternative Data for Thin-File Credit Default Prediction  
**Repository**: [DATASCI-207-FinalProject-Team1](https://github.com/romanmhall/DATASCI-207-FinalProject-Team1)  
**Next Deliverable**: Project Milestone Due Oct 12, 2026

---

## 1. Team Roster & Notebook Ownership Claim Sheet

Each team member will own a core notebook track and corresponding pipeline responsibilities. Team members can claim tracks by filling in their name below:

### Team Member Claim Roster
| Team Member | Email | Claimed Track / Role | Completed Tasks |
|---|---|---|---|
| Advik Goel | `advik_goel@berkeley.edu` | `[x]` | `[ ]` |
| Faisal | `faisal_k@berkeley.edu` | `[ ]` | `[ ]` |
| Roman Hall | `romanhall@berkeley.edu` | `[x]` | `[ ]` |
| Tung La | `tung_la@berkeley.edu` | `[ ]` | `[ ]` |

### Primary Notebook Track Directory
| Primary Notebook Track | Scope & Responsibilities | Claimed Owner |
|---|---|---|
| `notebooks/01_eda_home_credit.ipynb` | Static Application EDA, Missingness Profiling & Outlier Imputation | `[x]` |
| `notebooks/02_eda_relational_tables.ipynb` | Multi-Table Joins, Dynamic Aggregations & Colab Parquet Export | `[ ]` |
| `notebooks/03_alternative_features.ipynb` | Feature Cohorts, Baseline Scorecards & LightGBM Tuning | `[ ]` |
| `notebooks/04_model_experiments.ipynb` | TensorFlow Deep Neural Net, Subgroup Analysis & Metrics | `[x]` |

---

## 2. Task Breakdown & Detailed Checklist

### Task 1: Data Ingestion, Cleaning & Static EDA
**Primary Notebook**: `notebooks/01_eda_home_credit.ipynb`  
**Lead Owner**: `[x]`  
**Supporting Member(s)**: `[ ]`

- [ ] **Data Verification & Setup** (`data/home_credit_default_risk/`)
  - [ ] Verify `application_train.csv` (307,511 rows, 122 columns) integrity and sha256 checksums. **Owner: [ ]**
  - [ ] Confirm target distribution imbalance (8.07% default rate, label `TARGET` $\in \{0, 1\}$). **Owner: [ ]**
- [ ] **Missing Value Profiling & Data Cleaning**
  - [ ] Generate missingness heatmaps/barplots for external risk scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). **Owner: [ ]**
  - [ ] Implement median/mode imputation strategy; document handling of anomalous values (e.g., `DAYS_EMPLOYED = 365243`). **Owner: [ ]**
- [ ] **Demographic & Static Visualizations (Rubric Requirement)**
  - [ ] Produce labeled histograms and boxplots for income (`AMT_INCOME_TOTAL`), credit amount (`AMT_CREDIT`), and annuity (`AMT_ANNUITY`). **Owner: [ ]**
  - [ ] Plot default rates across age brackets (`DAYS_BIRTH`), education, and housing types. **Owner: [ ]**
  - [ ] Export 2–3 high-resolution milestone figures to `reports/milestone/figures/`. **Owner: [ ]**

#### Notes
-

---

### Task 2: Multi-Table Relational Aggregations & Colab Export
**Primary Notebook**: `notebooks/02_eda_relational_tables.ipynb`  
**Pipeline Code**: `src/data/preprocess.py`  
**Lead Owner**: `[ ]`  
**Supporting Member(s)**: `[ ]`

- [ ] **Dynamic Behavioral Aggregations**
  - [ ] Aggregate `installments_payments.csv` (13.6M rows): compute installment delay gap mean/max and underpayment frequency per `SK_ID_CURR`. **Owner: [ ]**
  - [ ] Aggregate `POS_CASH_balance.csv` (10.0M rows): compute monthly balance trajectories, past due days, and contract status counts. **Owner: [ ]**
  - [ ] Aggregate `previous_application.csv` (1.6M rows): compute historical refusal count, approval rate, and mean down payment percentage. **Owner: [ ]**
  - [ ] Aggregate `bureau.csv` (1.7M rows): compute active bureau lines, total overdue debt, and max days past due. **Owner: [ ]**
- [ ] **Data Challenge: Colab & Memory Optimization**
  - [ ] Execute one-time pre-aggregation join back to primary `SK_ID_CURR` index. **Owner: [ ]**
  - [ ] Export consolidated feature matrix to compressed Parquet (`data/processed/processed_features.parquet`) for fast Colab execution. **Owner: [ ]**
  - [ ] Document data size reduction metrics (before vs. after memory footprint in MB) for the milestone report. **Owner: [ ]**

#### Notes
-

---

### Task 3: Feature Cohorts, Baseline Scorecards & Tree Ensembles
**Primary Notebook**: `notebooks/03_alternative_features.ipynb`  
**Pipeline Code**: `src/features/build_features.py`, `src/models/baseline.py`, `src/models/tree_models.py`  
**Lead Owner**: `[ ]`  
**Supporting Member(s)**: `[ ]`

- [ ] **Feature Cohort Partitioning**
  - [ ] Cohort 1 (Traditional): Static demographics + Credit bureau features (`bureau.csv`). **Owner: [ ]**
  - [ ] Cohort 2 (Alternative): Behavioral trajectories (`installments_payments`, `previous_application`, `POS_CASH`) + `EXT_SOURCE`. **Owner: [ ]**
  - [ ] Cohort 3 (Hybrid): Concatenated feature space + interaction domain ratios (`PAYMENT_RATE`, `INCOME_CREDIT_RATIO`). **Owner: [ ]**
- [ ] **Baseline Linear Scorecard (Model 1)**
  - [ ] Train Logistic Regression (ElasticNet / L2) and Decision Tree on 80/20 train/validation split. **Owner: [ ]**
  - [ ] Compute baseline benchmarks: ROC-AUC, PR-AUC, and F1 score at calibrated threshold. **Owner: [ ]**
- [ ] **Gradient Boosted Decision Trees (Model 2 - Improvement 1)**
  - [ ] Train LightGBM / XGBoost with `scale_pos_weight` handling 8.07% class imbalance. **Owner: [ ]**
  - [ ] Execute 5-fold Stratified Cross-Validation on training set. **Owner: [ ]**
  - [ ] Generate SHAP feature importance summary plot comparing traditional vs. alternative predictors. **Owner: [ ]**

#### Notes
-

---

### Task 4: TensorFlow Tabular Neural Network & Subgroup Analysis
**Primary Notebook**: `notebooks/04_model_experiments.ipynb`  
**Pipeline Code**: `src/models/tf_neural_net.py`, `src/evaluation/metrics.py`  
**Lead Owner**: `[Advik ]`  
**Supporting Member(s)**: `[ ]`

- [ ] **Deep Tabular Neural Network (Model 3 - Improvement 2 - Mandatory Rubric Requirement)**
  - [ ] Build `tf.keras` Tabular Deep Neural Network with Entity Embeddings for categorical features. **Owner: [ ]**
  - [ ] Implement Batch Normalization, Dropout (0.2–0.3), and residual skip connections for numerical features. **Owner: [ ]**
  - [ ] Optimize using Adam with early stopping on validation ROC-AUC; track loss and learning curves. **Owner: [ ]**
- [ ] **Rigorous Imbalance & Subgroup Evaluation (Testing Hypothesis 2)**
  - [ ] Calculate comprehensive evaluation metrics: ROC-AUC, PR-AUC, Brier score, Expected Loss. **Owner: [ ]**
  - [ ] Subgroup segmentation: evaluate and report metrics separately for:
    - Group A: Applicants *with* credit bureau records (`bureau.csv` present). **Owner: [ ]**
    - Group B: Thin-file applicants *without* credit bureau records (zero bureau history). **Owner: [ ]**
  - [ ] Statistically test whether the alternative/hybrid lift ($\Delta\text{AUC}$) is significantly larger for Group B. **Owner: [ ]**

#### Notes
-

---

### Task 5: Project Milestone Report ($\le 3$-Page PDF) & Delivery
**Report Directory**: `reports/milestone/`  
**Target File**: `reports/milestone/milestone_report.md` $\rightarrow$ `reports/milestone/milestone_report.pdf`  
**Lead Owner**: `[Advik ]`  
**Section Writers**:
  - Section 1 (Team Info & Motivation): `[ ]`
  - Section 2 (Data Description, Sizes & Preprocessing): `[ ]`
  - Section 3 (EDA Visualizations & Interpretations): `[ ]`
  - Section 4 (Data Challenges & Scaling Solutions): `[ ]`
  - Section 5 (Planned Models, Evaluation & Subgroup Strategy): `[Advik ]`
  - Section 6 (Team Contributions & Notebook Ownership Log): `[ ]`

- [ ] **Milestone Quality Gate Checklist**
  - [ ] Strict page limit verified: $\le 3$ pages total (excluding references if permitted, or including per rubric). **Owner: [ ]**
  - [ ] All 4 team members' full names and `@berkeley.edu` emails included in header. **Owner: [ ]**
  - [ ] Preprocessing documented: missing values, outlier treatment, feature scaling, dataset sizes before and after. **Owner: [ ]**
  - [ ] Minimum 2–3 clearly labeled visual plots with descriptive captions and substantive analytical interpretations. **Owner: [ ]**
  - [ ] Discussion of data challenges: handling 13.6M row scale, memory constraints, and lookahead leakage prevention. **Owner: [ ]**
  - [ ] Detailed description of Baseline + 2 Improvements (including explicit mention of TensorFlow). **Owner: [ ]**
  - [ ] Individual contributions and notebook ownership clearly logged per member. **Owner: [ ]**
  - [ ] Gradescope submission: export to PDF, verify layout, and ensure all 4 members are tagged on submission. **Owner: [ ]**

#### Notes
-

---

## 3. Rubric Score Alignment (Milestone: 10% / 100 Points)

| Rubric Criteria | Points | Responsible Deliverable | Verification Command / Check |
|---|---|---|---|
| **Team Info** | 5 pts | Header of `milestone_report.pdf` | 4 names + 4 `@berkeley.edu` emails |
| **Motivation** | 5 pts | Section 1 of report | Clear thin-file problem + 2 hypotheses |
| **Data Description** | 10 pts | Section 2 of report | Tables, sizes before/after, inputs, target |
| **Data Preprocessing** | 25 pts | Section 2 & `src/data/preprocess.py` | Imputation, outliers, joins, parquet exports |
| **EDA & Visualizations** | 25 pts | Section 3 & `notebooks/01_eda_*.ipynb` | $\ge 2$ labeled plots with interpretations |
| **Data Challenges** | 10 pts | Section 4 of report | 13.6M row aggregation, imbalance, Colab RAM |
| **Methods & Experiments** | 10 pts | Section 5 & `notebooks/03_*.ipynb`, `04_*.ipynb` | Baseline + GBDT + TensorFlow NN, CV, metrics |
| **Team Contributions** | 5 pts | Section 6 of report | Table of notebook ownership & specific tasks |
| **Submission Format** | 5 pts | Final PDF on Gradescope | $\le 3$ pages, clean layout, all members tagged |
