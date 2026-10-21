# DATASCI 207 Final Project: Team Tasklist

**Project Title**: Behavioral Dynamics and Alternative Data for Thin-File Credit Default Prediction  
**Repository**: [DATASCI-207-FinalProject-Team1](https://github.com/romanmhall/DATASCI-207-FinalProject-Team1)  
**Next Deliverables**: Final Project Report (20%) & Live Presentation (5%) - Module 13

---

## 1. Team Roster & Notebook Ownership Claim Sheet

Each team member owns a core notebook track with sequential cell numbering starting at 1. In accordance with the course rubric, **each team member must contribute to the Methods and Experiments section**.

### Team Member Claim Roster
| Team Member | Email | Claimed Track / Role | Completed Tasks |
|---|---|---|---|
| Advik Goel | `advik_goel@berkeley.edu` | `[x]` | `[ ]` |
| Faisal Khasawneh | `faisal_k@berkeley.edu` | `[x]` | `[ ]` |
| Roman Hall | `romanhall@berkeley.edu` | `[x]` | `[x]` |
| Tung La | `tung_la@berkeley.edu` | `[x]` | `[x]` |

### Primary Notebook Track Directory
| Primary Notebook Track | Scope & Responsibilities | Claimed Owner |
|---|---|---|
| `notebooks/01_eda_home_credit.ipynb` | Static Application EDA, Missingness Profiling, Outlier Imputation & Baseline Scorecard | `[Roman Hall]` |
| `notebooks/02_eda_relational_tables.ipynb` | Multi-Table Joins, Dynamic Aggregations, Parquet Export & Dimensionality Reduction | `[Tung La]` |
| `notebooks/03_alternative_features.ipynb` | Feature Cohorts, TensorFlow Logistic Baseline & LightGBM Tuning | `[Faisal Khasawneh]` |
| `notebooks/04_model_experiments.ipynb` | TensorFlow Tabular Deep Neural Net, Subgroup Analysis & Metrics | `[Advik Goel]` |

---

## 2. Task Breakdown & Detailed Checklist

### Task 1: Data Ingestion, Cleaning, Static EDA & Baseline Scorecard
**Primary Notebook**: `notebooks/01_eda_home_credit.ipynb`  
**Lead Owner**: `[Roman Hall]`  
**Supporting Member(s)**: `[ ]`

- [x] **Data Verification & Setup** (`data/home_credit_default_risk/`)
  - [x] Verify `application_train.csv` (307,511 rows, 122 columns) integrity and sha256 checksums. **Owner: [Roman Hall]**
  - [x] Confirm target distribution imbalance (8.07% default rate, label `TARGET` $\in \{0, 1\}$). **Owner: [Roman Hall]**
- [x] **Missing Value Profiling & Data Cleaning**
  - [x] Generate missingness heatmaps/barplots for external risk scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). **Owner: [Roman Hall]**
  - [x] Implement median/mode imputation strategy; document handling of anomalous values (e.g., `DAYS_EMPLOYED = 365243`). **Owner: [Roman Hall]**
- [x] **Demographic & Static Visualizations**
  - [x] Produce labeled histograms and boxplots for income (`AMT_INCOME_TOTAL`), credit amount (`AMT_CREDIT`), and annuity (`AMT_ANNUITY`). **Owner: [Roman Hall]**
  - [x] Plot default rates across age brackets (`DAYS_BIRTH`), education, and housing types. **Owner: [Roman Hall]**
  - [x] Export 2-3 high-resolution milestone figures to `reports/milestone/figures/`. **Owner: [Roman Hall]**
- [ ] **Methods & Experiment Contribution (Mandatory Rubric Rule)**
  - [ ] Benchmark a baseline logistic scorecard on static application features to measure standalone discriminatory power prior to relational joins. **Owner: [Roman Hall]**
  - [ ] Ensure notebook cell numbers run sequentially starting at 1 with all execution outputs embedded. **Owner: [Roman Hall]**

#### Notes
- Static EDA complete with high-res figures embedded in milestone report. Modeling benchmark in progress to satisfy individual Methods/Experiments rubric credit.

---

### Task 2: Multi-Table Relational Aggregations & Colab Export
**Primary Notebook**: `notebooks/02_eda_relational_tables.ipynb`  
**Pipeline Code**: `src/data/preprocess.py`  
**Lead Owner**: `[Tung La]`  
**Supporting Member(s)**: `[ ]`

- [x] **Dynamic Behavioral Aggregations**
  - [x] Aggregate `installments_payments.csv` (13.6M rows): compute installment delay gap mean/max and underpayment frequency per `SK_ID_CURR`. **Owner: [Tung La]**
  - [x] Aggregate `POS_CASH_balance.csv` (10.0M rows): compute monthly balance trajectories, past due days, and contract status counts. **Owner: [Tung La]**
  - [x] Aggregate `previous_application.csv` (1.6M rows): compute historical refusal count, approval rate, and mean down payment percentage. **Owner: [Tung La]**
  - [x] Aggregate `bureau.csv` (1.7M rows): compute active bureau lines, total overdue debt, and max days past due. **Owner: [Tung La]**
- [x] **Data Challenge: Colab & Memory Optimization**
  - [x] Execute one-time pre-aggregation join back to primary `SK_ID_CURR` index. **Owner: [Tung La]**
  - [x] Export consolidated feature matrix to compressed Parquet (`data/processed/processed_features.parquet`) for fast Colab execution. **Owner: [Tung La]**
  - [x] Document data size reduction metrics (before vs. after memory footprint in MB) for the milestone report. **Owner: [Tung La]**
- [ ] **Methods & Experiment Contribution (Mandatory Rubric Rule)**
  - [ ] Evaluate feature selection / dimensionality reduction experiments (e.g., VarianceThreshold or correlation filtering) on baseline metrics. **Owner: [Tung La]**
  - [ ] Ensure notebook cell numbers run sequentially starting at 1 with all execution outputs embedded. **Owner: [Tung La]**

#### Notes
- Multi-table joins complete; `processed_features.parquet` committed to repo. Feature selection evaluation in progress for individual rubric credit.

---

### Task 3: Feature Cohorts, Baseline Scorecards & Tree Ensembles
**Primary Notebook**: `notebooks/03_alternative_features.ipynb`  
**Pipeline Code**: `src/features/build_features.py`, `src/models/baseline.py`, `src/models/tree_models.py`  
**Lead Owner**: `[Faisal Khasawneh]`  
**Supporting Member(s)**: `[ ]`

- [x] **Feature Cohort Partitioning**
  - [x] Cohort 1 (Traditional): Static demographics + Credit bureau features (`bureau.csv`). **Owner: [Faisal Khasawneh]**
  - [x] Cohort 2 (Alternative): Behavioral trajectories (`installments_payments`, `previous_application`, `POS_CASH`) + `EXT_SOURCE`. **Owner: [Faisal Khasawneh]**
  - [x] Cohort 3 (Hybrid): Concatenated feature space + interaction domain ratios (`PAYMENT_RATE`, `INCOME_CREDIT_RATIO`). **Owner: [Faisal Khasawneh]**
- [ ] **Baseline Linear Scorecard (Model 1 - Mandatory TensorFlow Rule)**
  - [ ] **CRITICAL RUBRIC REQUIREMENT**: Implement Logistic Regression in TensorFlow (`tf.keras.Sequential([Dense(1, activation='sigmoid')])`). **Owner: [Faisal Khasawneh]**
  - [x] Compute baseline benchmarks: ROC-AUC, PR-AUC, and F1 score at calibrated threshold across cohorts. **Owner: [Faisal Khasawneh]**
- [x] **Gradient Boosted Decision Trees (Model 2 - Improvement 1)**
  - [x] Train LightGBM with `scale_pos_weight` handling 8.07% class imbalance. **Owner: [Faisal Khasawneh]**
  - [x] Execute 5-fold Stratified Cross-Validation on training set. **Owner: [Faisal Khasawneh]**
  - [x] Generate SHAP feature importance summary plot comparing traditional vs. alternative predictors. **Owner: [Faisal Khasawneh]**
- [ ] **Notebook Execution & Cell Numbering**
  - [ ] Ensure notebook cell numbers run sequentially starting at 1 with all execution outputs embedded. **Owner: [Faisal Khasawneh]**

#### Notes
- Tree ensemble training and SHAP analysis complete; baseline model needs conversion to TensorFlow to satisfy course rubric.

---

### Task 4: TensorFlow Tabular Neural Network & Subgroup Analysis
**Primary Notebook**: `notebooks/04_model_experiments.ipynb`  
**Pipeline Code**: `src/models/tf_neural_net.py`, `src/evaluation/metrics.py`  
**Lead Owner**: `[Advik Goel]`  
**Supporting Member(s)**: `[ ]`

- [ ] **Deep Tabular Neural Network (Model 3 - Improvement 2 - Mandatory TensorFlow Rubric Rule)**
  - [ ] Build `tf.keras` Tabular Deep Neural Network with Entity Embeddings for categorical features. **Owner: [Advik Goel]**
  - [ ] Implement Batch Normalization, Dropout (0.2-0.3), and residual skip connections for numerical features. **Owner: [Advik Goel]**
  - [ ] Optimize using Adam with early stopping on validation ROC-AUC; track loss and learning curves. **Owner: [Advik Goel]**
- [ ] **Rigorous Imbalance & Subgroup Evaluation (Mandatory Rubric Rule)**
  - [ ] Calculate comprehensive evaluation metrics: ROC-AUC, PR-AUC, Brier score, Expected Loss. **Owner: [Advik Goel]**
  - [ ] Subgroup segmentation: evaluate and report metrics separately for:
    - [ ] Group A: Applicants *with* credit bureau records (`bureau.csv` present). **Owner: [Advik Goel]**
    - [ ] Group B: Thin-file applicants *without* credit bureau records (`HAS_BUREAU_HISTORY = 0`). **Owner: [Advik Goel]**
    - [ ] Group C: Gender demographic parity (`CODE_GENDER` Male vs. Female). **Owner: [Advik Goel]**
  - [ ] Statistically test whether the alternative/hybrid lift ($\Delta\text{AUC}$) is significantly larger for thin-file applicants (Hypothesis 2). **Owner: [Advik Goel]**
- [ ] **Notebook Execution & Cell Numbering**
  - [ ] Ensure notebook cell numbers run sequentially starting at 1 with all execution outputs embedded. **Owner: [Advik Goel]**

#### Notes
- Advik leading deep tabular neural net architecture, embeddings, and required subgroup evaluations.

---

### Task 5: Project Milestone Deliverable ($\le 3$-Page PDF) [COMPLETED]
**Report Directory**: `reports/milestone/`  
**Deliverable**: `reports/milestone/Final Project Milestone Submission.pdf`  
**Lead Owner**: `[Advik Goel / Team]`

- [x] **Milestone Quality Gate Checklist**
  - [x] Strict page limit verified: exactly 3 pages on Gradescope submission. **Owner: [Team]**
  - [x] All 4 team members' full names and `@berkeley.edu` emails included in header. **Owner: [Advik Goel]**
  - [x] Preprocessing documented: missing values, outlier treatment, feature scaling, dataset sizes before and after. **Owner: [Tung La]**
  - [x] Minimum 2-3 clearly labeled visual plots with descriptive captions and substantive analytical interpretations. **Owner: [Roman Hall]**
  - [x] Discussion of data challenges: handling 13.6M row scale, memory constraints, and lookahead leakage prevention. **Owner: [Tung La]**
  - [x] Detailed description of Baseline + 2 Improvements (including explicit mention of TensorFlow). **Owner: [Advik Goel / Faisal Khasawneh]**
  - [x] Individual contributions and notebook ownership clearly logged per member. **Owner: [Roman Hall]**
  - [x] Gradescope submission completed with all 4 members tagged. **Owner: [Team]**

#### Notes
- Milestone deliverable completed, verified at 3 pages, and pushed to repo.

---

### Task 6: Final Project Report ($\le 5$-Page PDF) & Delivery
**Report Directory**: `reports/final_report/`  
**Target File**: `reports/final_report/Final_Project_Report.pdf`  
**Lead Owner**: `[Team]`  
**Section Writers**:
  - Abstract & Introduction: `[Advik Goel]`
  - Related Work: `[Faisal Khasawneh]`
  - Dataset (preprocessing, EDA & implications): `[Roman Hall / Tung La]`
  - Methods (Baseline + 2 Improvements, TensorFlow justification): `[Faisal Khasawneh / Advik Goel]`
  - Experiments, Results & Discussion (tuning, metrics, subgroups, overfitting): `[All Team Members]`
  - Conclusions: `[Roman Hall]`
  - Contributions (notebook mapping, workload breakdown, GitHub link): `[Roman Hall]`
  - References & Appendix (excluded from 5-page limit): `[Team]`

- [ ] **Final Report Quality Gate Checklist**
  - [ ] **Strict Page Limit**: $\le 5$ pages total for main body (Header through Contributions). References and Appendix placed on page 6+ (excluded from page limit). **Owner: [Team]**
  - [ ] **Team Info (5 pts)**: Full names and `@berkeley.edu` emails for all 4 members. **Owner: [Advik Goel]**
  - [ ] **Abstract (5 pts)**: Concise 1-paragraph summary of motivation, ML methodology, and key findings. **Owner: [Advik Goel]**
  - [ ] **Introduction (5 pts)**: Clearly explains problem importance and includes exact mandatory sentence: *"The input to our algorithm is {tabular applicant demographic and historical credit transaction records}, and we use {TensorFlow Logistic Regression, LightGBM, and a TensorFlow Tabular Deep Neural Network} to predict {the binary probability of loan default within the target horizon}."* **Owner: [Advik Goel]**
  - [ ] **Related Work (5 pts)**: Reviews $\ge 2$ relevant papers/products, grouped by approach, discussing strengths/weaknesses and SOTA comparison with consistent citations. **Owner: [Faisal Khasawneh]**
  - [ ] **Dataset (5 pts)**: Data source, sizes before/after, train/validation/test splits, preprocessing steps, and thorough discussion of EDA visualizations and modeling implications. **Owner: [Roman Hall / Tung La]**
  - [ ] **Methods (30 pts)**: At least 3 algorithms (Baseline + 2 Improvements), justification, paragraph on how each works, and **TensorFlow strictly used for logistic regression and neural network**. **Owner: [Faisal Khasawneh / Advik Goel]**
  - [ ] **Experiments & Results (30 pts)**: Hyperparameter tuning, validation strategy, evaluation metrics, **subgroup performance evaluations** (thin-file vs bureau-backed, gender), overfitting analysis & mitigations, model comparison and selection justification. **Owner: [Advik Goel / Faisal Khasawneh]**
  - [ ] **Conclusions (5 pts)**: Wrap-up of key findings, best algorithms & why they succeeded, limitations, and future improvements. **Owner: [Roman Hall]**
  - [ ] **Contributions (5 pts)**: Detailed breakdown per member, notebook ownership (1 contributor per notebook, sequential cells starting at 1), Methods/Experiments contribution per member, GitHub repo link. **Owner: [Roman Hall]**
  - [ ] **Submission Format (5 pts)**: $\le 5$-page PDF, Letter (8.5x11 in), 10-12 pt font, margins $\ge 0.5$ in, submitted to Gradescope with all 4 members tagged. **Owner: [Team]**

#### Notes
- Final Report worth 20% of project grade. Reference format: MLE style or consistent APA/IEEE.

---

### Task 7: Final Project Presentation (12 Minutes Live)
**Presentation Directory**: `reports/presentation/`  
**Deliverable**: `reports/presentation/final_project_presentation.pdf` (or `.pptx`)  
**Lead Owner**: `[Team]`  
**Target Duration**: 12 minutes total (~3 minutes per team member)

- [ ] **Presentation Slide Deck Structure (10-12 Slides)**
  - [ ] Slide 1: Title, Team Members, Problem Statement & Motivation
  - [ ] Slide 2: Related Work & Industry Benchmark Limitations
  - [ ] Slide 3: Dataset Scale (26.9M rows) & Preprocessing Architecture
  - [ ] Slide 4: Key EDA Visualizations (Age risk gradient, external score separation)
  - [ ] Slide 5: Methods (TF Logistic Regression, LightGBM, TF Tabular Deep NN)
  - [ ] Slide 6: Experimental Setup, Cross-Validation & Metric Selection
  - [ ] Slide 7: Main Results (ROC-AUC / PR-AUC comparison table across cohorts)
  - [ ] Slide 8: Subgroup Performance Evaluation (Thin-File Lift vs. Bureau-Backed)
  - [ ] Slide 9: Overfitting Diagnostics & Error Analysis
  - [ ] Slide 10: Conclusions, Production Viability & Future Work
  - [ ] Slide 11: Team Contributions & GitHub Repository Link
- [ ] **Delivery & Quality Gates**
  - [ ] **Slide Quality (30 pts)**: Clear, concise slides with organized, easy-to-read charts/plots supporting key points. **Owner: [Team]**
  - [ ] **Presentation Delivery (10 pts)**: Cohesive live delivery strictly within 12-minute cap; each member presents their section. **Owner: [Team]**
  - [ ] **Q&A Handling (10 pts)**: Confident handling of audience and instructor questions regarding modeling choices and leakage prevention. **Owner: [Team]**

#### Notes
- Presentation delivered live during Module 13 live session (worth 5% of project grade).

---

## 3. Rubric Score Alignment & Weighting

### Milestone Rubric (10% Total Project Grade) - [COMPLETED]
| Rubric Criteria | Points | Responsible Deliverable | Verification Command / Check |
|---|---|---|---|
| **Team Info** | 5 pts | Header of milestone report | 4 names + 4 `@berkeley.edu` emails |
| **Motivation** | 5 pts | Section 1 of report | Clear thin-file problem + 2 hypotheses |
| **Data Description** | 10 pts | Section 2 of report | Tables, sizes before/after, inputs, target |
| **Data Preprocessing** | 25 pts | Section 2 & `src/data/preprocess.py` | Imputation, outliers, joins, parquet exports |
| **EDA & Visualizations** | 25 pts | Section 3 & `notebooks/01_eda_*.ipynb` | $\ge 2$ labeled plots with interpretations |
| **Data Challenges** | 10 pts | Section 4 of report | 13.6M row aggregation, imbalance, Colab RAM |
| **Methods & Experiments** | 10 pts | Section 5 & `notebooks/03_*.ipynb`, `04_*.ipynb` | Baseline + GBDT + TensorFlow NN, CV, metrics |
| **Team Contributions** | 5 pts | Section 6 of report | Table of notebook ownership & specific tasks |
| **Submission Format** | 5 pts | Final PDF on Gradescope | $\le 3$ pages, clean layout, all members tagged |

### Final Project Report Rubric (20% Total Project Grade)
| Rubric Criteria | Points | Responsible Deliverable | Verification Command / Check |
|---|---|---|---|
| **Team Info** | 5 pts | Header of `Final_Project_Report.pdf` | 4 names + 4 `@berkeley.edu` emails |
| **Abstract** | 5 pts | Section 1 of report | 1 concise paragraph (problem, methods, findings) |
| **Introduction** | 5 pts | Section 2 of report | Importance + mandatory input/output sentence |
| **Related Work** | 5 pts | Section 3 of report | $\ge 2$ papers grouped by approach, SOTA comparison |
| **Dataset & Preprocessing** | 5 pts | Section 4 of report | Sizes before/after, splits, EDA implications |
| **Methods** | 30 pts | Section 5 of report | Baseline + 2 improvements; **TensorFlow mandatory** |
| **Experiments & Results** | 30 pts | Section 6 of report | Tuning, metrics, **subgroups**, overfitting analysis |
| **Conclusions** | 5 pts | Section 7 of report | Findings, winner explanation, limitations |
| **Contributions** | 5 pts | Section 8 of report | 1 contributor/notebook, Methods credit, GitHub URL |
| **Submission Format** | 5 pts | Final PDF on Gradescope | $\le 5$ pages (main body), Letter, all members tagged |

### Final Presentation Rubric (5% Total Project Grade)
| Rubric Criteria | Points | Responsible Deliverable | Verification Command / Check |
|---|---|---|---|
| **Presentation Content** | 50 pts | Slide deck & live talk | Covers all key report components |
| **Slide Quality** | 30 pts | Presentation visual design | Clear, concise slides, readable charts/tables |
| **Presentation Delivery** | 10 pts | Live delivery | Cohesive delivery $\le 12$ min, all members present |
| **Question Handling** | 10 pts | Live Q&A session | Confident, accurate answers demonstrating mastery |
