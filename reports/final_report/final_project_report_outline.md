# Behavioral Dynamics and Alternative Data for Thin-File Credit Default Prediction
**DATASCI 207: Applied Machine Learning at Scale | Final Project Report**  
**Team 1**: Advik Goel (`advik_goel@berkeley.edu`), Faisal Khasawneh (`faisal_k@berkeley.edu`), Roman Hall (`romanhall@berkeley.edu`), Tung La (`tung_la@berkeley.edu`)  
**GitHub Repository**: [DATASCI-207-FinalProject-Team1](https://github.com/romanmhall/DATASCI-207-FinalProject-Team1)

---

## Abstract
> **Assigned Lead**: Advik Goel  
> **Target Length**: 1 concise paragraph (~150 words)  
> **Rubric Criteria**: Summarize project motivation, ML methodology, and key empirical findings.

### [NOTES FOR ADVIK TO FILL IN]
- [ ] **Motivation**: Introduce the thin-file credit problem (unbanked/underbanked borrowers lacking traditional credit bureau records).
- [ ] **Machine Learning Methodology**: State the three model families used (TensorFlow Logistic Regression baseline, LightGBM GBDT tree ensembles, and TensorFlow Tabular Deep Neural Network).
- [ ] **Data Scope**: Mention multi-table aggregation across 26.9M behavioral transactions down to a 307k primary applicant matrix.
- [ ] **Key Findings**: State key results (Hybrid cohort achieves 0.7699 ROC-AUC on LightGBM and 0.7595 on TF NN; alternative behavioral features provide statistically significant performance lift specifically for thin-file applicants, confirming Hypothesis 2).

---

## 1. Introduction
> **Assigned Lead**: Advik Goel  
> **Target Length**: ~0.5 pages  
> **Rubric Criteria**: Problem importance, real-world context, and mandatory algorithmic input/output definition.

### [NOTES FOR ADVIK TO FILL IN]
- [ ] **Problem & Importance**: Explain why traditional credit scoring fails thin-file borrowers and perpetuates financial exclusion.
- [ ] **Research Question & Hypotheses**: State the core question (Can alternative behavioral transaction data substitute for credit bureau history?) and reiterate Hypothesis 1 (general hybrid lift) and Hypothesis 2 (disproportionate thin-file lift).
- [ ] **MANDATORY RUBRIC SENTENCE (Must be included verbatim or closely matched)**:
  > *"The input to our algorithm is tabular applicant demographic and historical multi-table credit transaction records, and we use TensorFlow Logistic Regression, LightGBM Gradient Boosted Decision Trees, and a TensorFlow Tabular Deep Neural Network to predict the binary probability of loan default within the target horizon."*
- [ ] **High-Level Scope**: Outline paper structure (Related Work $\rightarrow$ Dataset $\rightarrow$ Methods $\rightarrow$ Experiments $\rightarrow$ Discussion).

---

## 2. Related Work
> **Assigned Lead**: Faisal Khasawneh  
> **Target Length**: ~0.5 pages  
> **Rubric Criteria**: Review $\ge 2$ relevant papers/products, grouped by approach, discuss strengths/weaknesses, compare with your approach, identify SOTA.

### [NOTES FOR FAISAL TO FILL IN]
- [ ] **Approach Group 1 — Traditional Scorecards & Tree Ensembles**:
  - Review traditional linear/logistic credit scoring vs. modern gradient boosted decision trees (e.g., Chen & Guestrin 2016 XGBoost / Ke et al. 2017 LightGBM).
  - Discuss strengths (handling missingness natively, non-linear interactions, high interpretability via SHAP) and weaknesses (struggle with high-cardinality sparse embeddings).
- [ ] **Approach Group 2 — Alternative Data & Tabular Deep Learning**:
  - Review literature applying alternative behavioral data (telecom, utilities, transaction sequences) and tabular neural networks (e.g., Gorodnichenko et al. 2021; Borisov et al. 2022).
  - Discuss strengths (joint latent representation learning across multimodal fields) and weaknesses (computational overhead, overfitting risk on tabular data).
- [ ] **Comparison with Our Work**: Contrast how our hybrid cohort architecture combines static demographics, external bureau scores, and multi-table behavioral aggregates to specifically evaluate thin-file vs. bureau-backed lift.

---

## 3. Dataset, Preprocessing & Exploratory Data Analysis
> **Assigned Leads**: Roman Hall (Static EDA & Preprocessing) & Tung La (Multi-Table Relational Joins & Diagnostics)  
> **Target Length**: $\le 1$ page  
> **Rubric Criteria**: Data sources, sizes before/after, train/val/test splits, preprocessing steps, join diagnostics, and EDA implications.

### [NOTES FOR ROMAN TO FILL IN — Application EDA & Preprocessing]
- [ ] **Primary Data Source & Imbalance**: Describe `application_train.csv` (307,511 primary applicants, 122 raw features, 8.07% default rate, 11.39:1 imbalance ratio).
- [ ] **Data Splits**: Document the 70 / 10 / 20 stratified train / validation / test splits (Train: 215,257; Val: 30,751; Test: 61,503).
- [ ] **Anomaly Cleaning**: Document the `DAYS_EMPLOYED = 365,243` pension anomaly (5.40% default vs. 8.66% standard) and how flagging it prevented model bias.
- [ ] **Domain Ratio Engineering**: Document feature engineering for `PAYMENT_RATE`, `INCOME_CREDIT_RATIO`, and `ANNUITY_INCOME_RATIO`.
- [ ] **EDA Visualizations**: Reference and interpret figures:
  - *Figure 1*: Target class imbalance and top missingness (`EXT_SOURCE_1` missing 56%).
  - *Figure 2*: Age risk gradient (default rate drops monotonically from 12.3% for age 20-25 down to 5.2% for age 60+).
  - *Figure 3*: External score discrimination curves (`EXT_SOURCE_1/2/3`).

### [NOTES FOR TUNG TO FILL IN — Relational Joins & Memory Scaling]
- [ ] **Relational Data Scale**: Document the 4 relational tables totaling 26.9M rows (`bureau.csv`: 1.7M, `previous_application.csv`: 1.6M, `POS_CASH_balance.csv`: 10.0M, `installments_payments.csv`: 13.6M).
- [ ] **Join Diagnostics (Addressing Professor Uri's Feedback)**:
  - Report match coverage across primary applicants: Bureau (85.7%), Previous Apps (94.6%), POS-Cash (94.1%), Installments (94.8%).
  - Confirm 0 duplicate matches / 0 index fan-out collisions during left-joins back to primary `SK_ID_CURR`.
  - Document thin-file population size: 44,020 applicants (14.3%) with zero bureau history (Group B).
- [ ] **Memory Scaling & Engineering Challenge**:
  - Explain why 26.9M rows exceeded standard Colab RAM (12.7 GB).
  - Document the three solutions: column pruning (`usecols`), sequential garbage-collected processing, and numeric downcasting (saving 30% to 78.4% memory).
  - **Downcasting vs. Row Sampling Experiment (Uri's Note)**: Explain why downcasting was chosen over row sampling (sampling 20% dropped `INST_DELAY_MAX` correlation to 0.523 and missed 5.8% of applicants, whereas downcasting had 0% information loss).
  - Export: Final 153-feature matrix saved once as snappy-compressed Parquet (37.2 MB).

---

## 4. Methods
> **Assigned Leads**: Faisal Khasawneh (LightGBM & Baseline) & Advik Goel (TensorFlow Architecture)  
> **Target Length**: 1 to 1.5 pages  
> **Rubric Criteria**: At least 3 algorithms (Baseline + 2 Improvements), justification, paragraph on how each works, and **TensorFlow strictly used for logistic regression and neural networks**.

### [NOTES FOR FAISAL TO FILL IN]
- [ ] **Algorithm 1 — Baseline: TensorFlow Logistic Regression**:
  - Describe architecture: Single `Dense(1, activation='sigmoid')` layer with $L_2$ weight regularization (`1e-4`), binary cross-entropy loss, and Adam optimizer.
  - Explain why appropriate: Provides an interpretable, convex linear separability benchmark directly comparable to industry credit scorecards.
- [ ] **Algorithm 2 — Improvement 1: LightGBM Gradient Boosted Decision Trees**:
  - Describe how it works: Leaf-wise tree growth, histogram-based splitting, gradient-based one-side sampling (GOSS).
  - Explain why appropriate: Natively routes missing values (crucial for sparse external scores and bureau fields), handles non-linear interactions, and scales efficiently.
  - Document class imbalance handling: `scale_pos_weight = 11.39` to counterbalance 8.07% default rate.

### [NOTES FOR ADVIK TO FILL IN]
- [ ] **Algorithm 3 — Improvement 2: Deep Tabular Neural Network (TensorFlow / Keras)**:
  - Describe architecture: Input layer (numerical + categorical), Entity Embeddings for categorical features, Dense hidden layers (units: 64 to 256), `BatchNormalization`, `Dropout` ($p=0.2$ to $0.3$) for regularization, ReLU activations, and final Sigmoid output.
  - Training configuration: Adam optimizer ($\eta = 10^{-3}$), early stopping on validation ROC-AUC with patience = 5.
  - Explain why appropriate: Tests whether deep joint latent representations and non-linear entity embeddings can discover complex interactions in credit transactions that decision trees miss.

---

## 5. Experiments, Results & Discussion
> **Assigned Leads**: All Team Members  
> **Target Length**: 1 to 2 pages  
> **Rubric Criteria**: Hyperparameter tuning, validation strategy, evaluation metrics (ROC-AUC, PR-AUC, separate Precision and Recall), subgroup evaluations, overfitting analysis, model comparison.

### [NOTES FOR ROMAN TO FILL IN — Static Feature Ablation]
- [ ] Document the static application baseline ablation results from `notebooks/01_eda_home_credit.ipynb`:
  - Demographics only: 0.5834 ROC-AUC.
  - Adding loan amounts & ratios: ~0.6085 ROC-AUC.
  - Adding external scores (`EXT_SOURCE`): 0.7230 ROC-AUC (showing external scores provide the bulk of static discrimination).

### [NOTES FOR FAISAL TO FILL IN — Tree Ensemble Benchmarks & SHAP]
- [ ] Document 5-fold cross-validation performance across the 4 cohorts from `notebooks/03_alternative_features.ipynb`:
  - Cohort 1 (Traditional): LightGBM Val ROC-AUC = 0.7128.
  - Cohort 2 (Alternative): LightGBM Val ROC-AUC = 0.7307.
  - Cohort 3 (Hybrid): LightGBM Val ROC-AUC = **0.7699**, PR-AUC = **0.2607**, F1 = 0.3170.
  - Cohort 4 (No Bureau): LightGBM Val ROC-AUC = 0.7270.
- [ ] Document the **+0.0571 ROC-AUC lift** of Hybrid over Traditional alone.
- [ ] Interpret SHAP feature attributions from best LightGBM model:
  - Traditional bureau & static features: 57.7% of total feature importance.
  - Alternative behavioral features: 35.9% of importance.
  - Engineered domain ratios: 6.4% of importance.

### [NOTES FOR ADVIK TO FILL IN — Deep Learning, Subgroups & Precision/Recall]
- [ ] **Neural Network Performance**: Document TF Neural Net results across cohorts from `notebooks/04_model_experiments.ipynb` (Cohort 1: 0.7018 $\rightarrow$ Cohort 3 Hybrid: **0.7595 ROC-AUC**, 0.2482 PR-AUC).
- [ ] **Precision & Recall Separation (Addressing Professor Uri's Feedback)**:
  - Tabulate explicit Precision and Recall at calibrated decision thresholds ($\tau \approx 0.13$ to $0.17$ for TF models, $\tau \approx 0.60$ for weighted LightGBM).
  - Discuss the business trade-off: capturing ~40–47% of all defaults (Recall) at ~25–32% Precision.
- [ ] **Mandatory Subgroup Performance Analysis**:
  - Segment test set into **Group A (Bureau History, $N \approx 263k$)** vs. **Group B (Thin-File, $N \approx 44k$)**.
  - Evaluate Gender Demographic Parity: Male vs. Female applicants.
- [ ] **Hypothesis 2 Empirical Validation (Bootstrap Significance)**:
  - Reference `fig_h2_bootstrap.png`: show that the performance lift from alternative behavioral data is statistically significantly larger for thin-file applicants ($\Delta\text{AUC}_{\text{thin-file}} > \Delta\text{AUC}_{\text{bureau}}$).

### [NOTES FOR TEAM — Overfitting Analysis & Model Comparison]
- [ ] **Overfitting Diagnostics**: Compare Train AUC vs. Validation AUC across models (e.g., TF NN Train 0.7926 vs. Val 0.7595 shows mild overfitting controlled by Dropout and EarlyStopping).
- [ ] **Final Model Selection**: Justify selecting LightGBM as the champion production model (highest ROC-AUC of 0.7699, superior PR-AUC of 0.2607, faster inference, and native missing value handling).

---

## 6. Conclusions & Limitations
> **Assigned Lead**: Roman Hall  
> **Target Length**: 1 to 2 paragraphs  
> **Rubric Criteria**: Key takeaways, best-performing algorithm explanation, real-world limitations, and future improvements.

### [NOTES FOR ROMAN TO FILL IN]
- [ ] **Summary of Work & Findings**: Restate that multi-table alternative behavioral features provide substantial predictive lift (+0.057 AUC) and effectively substitute for traditional bureau history for unbanked applicants.
- [ ] **Best Model Justification**: Summarize why LightGBM won over deep neural networks on tabular credit data (tabular inductive bias, missingness routing).
- [ ] **Limitations**: Acknowledge dataset age, non-time-series snapshot structure, and lack of macro-economic interest rate variables.
- [ ] **Future Work (Addressing Professor Uri's Feedback)**:
  - Distributed out-of-core pipeline scaling using PySpark, DuckDB, or Polars for enterprise-scale transaction streams.
  - Graph Neural Networks (GNNs) to model co-borrower and transaction network topologies.

---

## 7. Team Contributions & Notebook Ownership Log
> **Assigned Lead**: Roman Hall  
> **Target Length**: ~0.5 pages (Counts toward the 5-page limit)  
> **Rubric Criteria**: Breakdown of specific tasks, single notebook ownership per member, sequential cell execution, and GitHub link.

### [NOTES FOR ROMAN TO FILL IN — Table Breakdown]
| Team Member | Email Address | Owned Notebook Track | Core Methods & Experiments Contributions |
|---|---|---|---|
| **Roman Hall** | `romanhall@berkeley.edu` | `notebooks/01_eda_home_credit.ipynb` | Application dataset ingestion, missingness heatmaps, `DAYS_EMPLOYED` anomaly cleaning, financial ratio engineering (`PAYMENT_RATE`, `INCOME_CREDIT_RATIO`), 4 milestone figures, and **TensorFlow Logistic Regression static feature ablation benchmark (3-fold CV with Precision/Recall)**. |
| **Tung La** | `tung_la@berkeley.edu` | `notebooks/02_eda_relational_tables.ipynb` | Multi-table relational aggregation pipeline across 4 tables (26.9M rows), memory downcasting optimization, join diagnostics (match rates & duplicate verification), and **memory scaling benchmark (downcasting vs. row sampling)**. |
| **Faisal Khasawneh** | `faisal_k@berkeley.edu` | `notebooks/03_alternative_features.ipynb` | Feature cohort partitioning (Traditional, Alternative, Hybrid, Thin-File), 5-fold cross-validation framework, LightGBM tree ensemble hyperparameter tuning (`scale_pos_weight = 11.39`), and **SHAP feature attribution analysis**. |
| **Advik Goel** | `advik_goel@berkeley.edu` | `notebooks/04_model_experiments.ipynb` | TensorFlow Logistic Regression baseline, Deep Tabular Neural Network with entity embeddings, hyperparameter tuning grid, **subgroup evaluation (Thin-file vs. Bureau, Gender parity), and Hypothesis 2 bootstrap significance test**. |

- [ ] Note: Confirm that all four notebooks on GitHub have single-contributor ownership and sequential cell numbering starting at 1.
- [ ] GitHub Repository URL: `https://github.com/romanmhall/DATASCI-207-FinalProject-Team1`

---

## 8. References (Excluded from 5-Page Limit)
> **Assigned Lead**: Team  
> **Target Placement**: Starts on Page 6 (Does not count against the 5-page rubric limit)

### [NOTES FOR TEAM TO FILL IN]
- [ ] [1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- [ ] [2] Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- [ ] [3] Gorodnichenko, Y., et al. (2021). The credit score of the unbanked: Predicting default with alternative data. *Journal of Financial Economics*.
- [ ] [4] Borisov, V., et al. (2022). Deep neural networks and tabular data: A survey. *IEEE Transactions on Neural Networks and Learning Systems*.

---
