# DATASCI 207 Final Project Presentation Slide Deck (12 Minutes Live)

**Project Title**: Behavioral Dynamics and Alternative Data for Thin-File Credit Default Prediction  
**Team 1**: Advik Goel, Faisal Khasawneh, Roman Hall, Tung La  
**Course**: UC Berkeley DATASCI 207 (Machine Learning at Scale)  
**Target Duration**: 12 Minutes Total (~3 minutes per team member)  
**Repository**: [https://github.com/romanmhall/DATASCI-207-FinalProject-Team1](https://github.com/romanmhall/DATASCI-207-FinalProject-Team1)  

---

## Slide 1: Title, Team & Problem Statement
- **Slide Title**: Behavioral Dynamics & Alternative Data for Thin-File Credit Default Prediction
- **Presenter**: Advik Goel (Minutes 0:00 - 1:30)
- **Visuals**: UC Berkeley header, Team 1 roster, Home Credit logo.
- **Key Points**:
  - The Thin-File Problem: Traditional credit scoring (FICO, VantageScore) relies exclusively on multi-year established credit bureau lines.
  - Structural Exclusion: Millions of young adults, students, recent immigrants, and informal economy workers are classified as "thin-file" or "unscoreable," leading to routine rejections or predatory lending terms.
  - Core Research Question: Can alternative behavioral transaction dynamics (installment repayment punctuality, past application outcomes, and POS cash balances) substitute for or augment missing credit bureau records?
- **Speaker Notes**:
  "Good morning everyone. Our project tackles financial exclusion in consumer lending. Traditional credit scoring assumes that if you don't have existing debt, you can't be trusted with credit. We evaluate whether granular transactional behavior can accurately predict probability of default for thin-file borrowers."

---

## Slide 2: Research Motivation & Related Work
- **Slide Title**: Motivation, Hypotheses & Industry Context
- **Presenter**: Advik Goel (Minutes 1:30 - 3:00)
- **Visuals**: Two-column approach comparison (Traditional Scorecards vs Alternative Big Data).
- **Key Points**:
  - Related Work: Berg et al. (2020) demonstrated that digital footprints predict default with 0.696 AUC, lifting traditional bureau AUC from 0.683 to 0.736 and reducing default rates by up to 40%.
  - Hypothesis 1 (General Behavioral Augmentation): Alternative behavioral transaction features provide statistically significant discrimination lift when combined with traditional scores.
  - Hypothesis 2 (Disproportionate Thin-File Benefit): Alternative behavioral features provide greater discrimination lift for thin-file borrowers than for bureau-backed borrowers.
  - Algorithmic Formalization: Predicting binary loan default ($y \in \{0, 1\}$, Basel II/III 90+ DPD standard) from tabular applicant demographics and multi-table transactions.
- **Speaker Notes**:
  "Our algorithmic formulation inputs tabular applicant demographics and multi-table transactions, predicting binary default within the target horizon using TensorFlow Logistic Regression, LightGBM, and Deep Neural Networks."

---

## Slide 3: Dataset Scale & Multi-Table Relational Architecture
- **Slide Title**: Relational Data Scale (26.9 Million Rows) & Hygiene
- **Presenter**: Tung La (Minutes 3:00 - 4:30)
- **Visuals**: Schema diagram connecting 4 transactional tables to primary `SK_ID_CURR`.
- **Key Points**:
  - Data Scope: `application_train.csv` (307,511 primary applicants) joined across 4 transaction tables totaling 26.9M rows:
    - `bureau.csv` (1.72M rows): External credit lines and debt totals.
    - `previous_application.csv` (1.67M rows): Past refusal and approval rates.
    - `POS_CASH_balance.csv` (10.0M rows): Monthly contract status and days past due.
    - `installments_payments.csv` (13.6M rows): Repayment timing delays and underpayment frequency.
  - Big Data Memory Optimization: Loading 26.9M rows exceeded Colab's 12.7 GB RAM limit. We deployed column pruning, dtype downcasting (`float64` to `float32`, `int64` to `int16`), and sequential table aggregation, compressing 1.47 GB raw data down to a 37.3 MB Snappy Parquet file (-74% to -78% footprint).
  - Lookahead Leakage Audit: Verified all timestamp offsets are non-positive relative to application date ($\max(\text{DAYS\_*}) \le 0$).
- **Speaker Notes**:
  "Handling 26.9 million rows on constrained hardware required zero-information-loss downcasting. Rather than downsampling, which would destroy rare default instances in an 8% imbalanced setting, downcasting preserved every single applicant."

---

## Slide 4: Exploratory Data Analysis & Hypothesis Validation
- **Slide Title**: Exploratory Profiling & The Youth Risk Gradient
- **Presenter**: Roman Hall (Minutes 4:30 - 6:00)
- **Visuals**: Side-by-side plots of Class Imbalance (8.07%), External Score Missingness, and Default Rate vs. Age Cohorts.
- **Key Points**:
  - Class Imbalance: 8.07% default rate (11.39:1 ratio). Trivial non-default guessing yields 91.93% accuracy but zero risk detection.
  - Informative Missingness: `EXT_SOURCE_1` is missing in 56.4% of records; `EXT_SOURCE_3` in 19.8%. Missingness is an informative proxy for thin files, not random noise.
  - Pensioner Anomaly: 55,374 applicants (18.0%) exhibit `DAYS_EMPLOYED = 365,243` (~1,000 years). Pensioners default at only 5.40% vs 8.66% for active workers. Fixed via `DAYS_EMPLOYED_ANOM = 1`.
  - Empirical Hypothesis Confirmation: Default rate drops monotonically from 12.30% for age 20-25 down to 5.23% for age 60+, validating that younger thin-file borrowers face acute default risk.
- **Speaker Notes**:
  "Our EDA confirmed the foundational hypothesis of thin-file lending: young borrowers experience more than double the default rate of retirees. Because traditional credit bureaus rely on file age, young borrowers are penalized unless behavioral repayment data is introduced."

---

## Slide 5: Methods: Baseline & Two Improvements
- **Slide Title**: Modeling Framework: Linear, Tree Ensemble & Deep Learning
- **Presenter**: Faisal Khasawneh (Minutes 6:00 - 7:30)
- **Visuals**: Architecture diagrams for TF Logistic Regression, LightGBM, and TF Deep Tabular Neural Network.
- **Key Points**:
  - Model 1 (Baseline Scorecard): Regularized Logistic Regression in TensorFlow Keras (`Dense(1, activation='sigmoid')`, L2 penalty, Adam).
  - Model 2 (Tree Improvement): LightGBM GBDT with `scale_pos_weight = 11.39`, 31 leaves, 0.03 learning rate, native missing value routing.
  - Model 3 (Deep Learning Improvement): TensorFlow Deep Tabular Neural Network with Dense(128) $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(1).
  - Feature Cohorts: Benchmarked across Cohort 1 (Traditional), Cohort 2 (Alternative), Cohort 3 (Hybrid), and Cohort 4 (Thin-File).
- **Speaker Notes**:
  "To satisfy course rubrics and regulatory standards, we benchmarked three distinct model families. TensorFlow was used for both linear scorecards and deep neural networks, while LightGBM served as our non-linear tree ensemble champion."

---

## Slide 6: Experimental Setup, Validation Strategy & Metrics
- **Slide Title**: Validation Discipline & Loss Cost Asymmetry
- **Presenter**: Faisal Khasawneh (Minutes 7:30 - 8:30)
- **Visuals**: Flowchart of 80/20 train/test split, 90/10 fit/early-stop nested split, and cost matrix.
- **Key Points**:
  - Zero-Leakage Split Discipline: Stratified 80/20 train/test split executed prior to EDA and feature scaling. Training medians and scalers fit strictly on train partition.
  - Metric Selection:
    - ROC-AUC: Threshold-independent ranking across all false positive rates.
    - PR-AUC: Dedicated focus on minority default class detection without true-negative inflation.
    - Separate Precision & Recall: Critical in lending where False Negatives cost 100% principal write-off while False Positives cost ~5-15% lost interest margin.
    - Brier Score & Expected Credit Loss ($E[\text{Loss}] = \sum \hat{p}_i \times \text{AMT\_CREDIT}_i$).
- **Speaker Notes**:
  "We addressed Professor Uri's feedback directly: all EDA and preprocessing medians derive strictly from the training set before any evaluation. We also report precision and recall separately to reflect real-world lending economics."

---

## Slide 7: Main Benchmark Results Across Cohorts
- **Slide Title**: Benchmark Evaluation Across Feature Cohorts
- **Presenter**: Advik Goel (Minutes 8:30 - 9:30)
- **Visuals**: Master comparison table across models and cohorts; ROC and PR curves.
- **Key Points**:
  - LightGBM Champion: Achieves 0.7712 ROC-AUC and 0.2645 PR-AUC on the Hybrid Cohort 3.
  - Deep Neural Net Benchmark: Achieves 0.7595 ROC-AUC and 0.2482 PR-AUC, outperforming Logistic Regression (0.7586).
  - Hypothesis 1 Validated: Adding behavioral transaction dynamics (Hybrid Cohort 3) yields a substantial +0.055 to +0.068 ROC-AUC lift over traditional credit bureau scores across all model families.
- **Speaker Notes**:
  "Across every model architecture, the Hybrid feature space significantly outperformed traditional credit bureau lines alone. LightGBM won the benchmark due to native continuous interaction splits and cost weighting."

---

## Slide 8: Subgroup Performance & Hypothesis 2 Bootstrap
- **Slide Title**: Subgroup Fairness & Hypothesis 2 Empirical Test
- **Presenter**: Advik Goel (Minutes 9:30 - 10:30)
- **Visuals**: Subgroup bar charts and 1,000-sample bootstrap differential lift distribution.
- **Key Points**:
  - Subgroup Segmentation:
    - Group A (Bureau-Backed, N=52.7k): ROC-AUC lifts from 0.7067 to 0.7619 (+0.0552 lift).
    - Group B (Thin-File, N=8.8k): ROC-AUC lifts from 0.6751 to 0.7296 (+0.0545 on LogReg; +0.0597 on Neural Net).
  - Hypothesis 2 Findings: While thin-file applicants experience dramatic discrimination gains (+0.060 AUC lift), 1,000 bootstrap resamples reveal the differential lift confidence interval spans $[-0.014, +0.017]$ ($p=0.593$). Behavioral transaction fidelity lifts both cohorts equally.
  - Demographic Parity: Male (0.7521) vs. Female (0.7570) ROC-AUC confirms algorithmic fairness.
- **Speaker Notes**:
  "Alternative data rescues thin-file borrowers, lifting their ROC-AUC from near-random 0.675 up to commercially viable 0.736. However, our bootstrap test showed the lift is equally powerful for established borrowers, proving behavioral transaction fidelity benefits all applicants."

---

## Slide 9: Operational Decision Thresholds & Error Analysis
- **Slide Title**: Calibrated Policy Thresholds & Overfitting Analysis
- **Presenter**: Roman Hall (Minutes 10:30 - 11:15)
- **Visuals**: Threshold trade-off curve (Precision, Recall, Specificity) and NN learning curves.
- **Key Points**:
  - Operational Policy Thresholds:
    - Aggressive Default Avoidance ($\tau = 0.10$): Recall = 78.4%, Precision = 16.5%, Flags 38.2% of applicants.
    - Balanced Cutoff ($\tau = 0.20$): Recall = 54.1%, Precision = 24.8%, Specificity = 85.2%.
    - Conservative Approval ($\tau = 0.50$): High precision (41.2%), but misses 81.7% of late payers.
  - Overfitting Mitigations: Dropout ($p=0.30$) and early stopping successfully capped training divergence in TensorFlow.
- **Speaker Notes**:
  "In credit underwriting, picking an operating threshold is a business decision. At tau = 0.20, an institution catches over 54% of defaults while approving 85% of creditworthy borrowers."

---

## Slide 10: Conclusions, Big Data Scalability & Future Work
- **Slide Title**: Production Roadmap, Limitations & Takeaways
- **Presenter**: Tung La (Minutes 11:15 - 12:00)
- **Visuals**: Architecture diagram of distributed PySpark / DuckDB out-of-core streaming.
- **Key Points**:
  - Takeaway 1: Granular installment timing and application refusal counts successfully bridge credit information asymmetry.
  - Takeaway 2: LightGBM is the production champion for tabular credit default.
  - Enterprise Scalability: While memory downcasting enabled single-node training, production systems processing streaming transaction feeds should deploy **PySpark** or **DuckDB / Polars** for distributed rolling aggregations.
  - Future Work: Graph Neural Networks (GNNs) to model transaction networks across co-borrowers and merchants.
- **Speaker Notes**:
  "To scale beyond 26M transactions in production, we recommend out-of-core streaming engines like DuckDB or PySpark. Thank you, and we welcome your questions."

---

## Slide 11: Team Contributions & Repository Link
- **Slide Title**: Team Roles, Notebook Ownership & Reproducibility
- **All Team Members**:
  - Roman Hall (`romanhall@berkeley.edu`): `01_eda_home_credit.ipynb` (Static EDA, anomaly repair, TF LogReg ablation).
  - Tung La (`tung_la@berkeley.edu`): `02_eda_relational_tables.ipynb` (Multi-table joins, memory downcasting, feature selection).
  - Faisal Khasawneh (`faisal_k@berkeley.edu`): `03_alternative_features.ipynb` (Cohort architecture, LightGBM tuning, SHAP).
  - Advik Goel (`advik_goel@berkeley.edu`): `04_model_experiments.ipynb` (Deep Tabular NN, subgroup analysis, bootstrap test).
  - All Members: `00_final_end_to_end_credit_project.ipynb` (Unified Master Pipeline).
  - GitHub Repository: [https://github.com/romanmhall/DATASCI-207-FinalProject-Team1](https://github.com/romanmhall/DATASCI-207-FinalProject-Team1)
