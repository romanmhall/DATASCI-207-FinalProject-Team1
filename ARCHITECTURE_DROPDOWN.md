# Repository Directory Architecture Dropdown

<!-- COPY FROM HERE INTO README.MD OR GITHUB WHEN READY -->
<details>
<summary><b>📂 Repository Directory Architecture (Click to expand)</b></summary>

```text
credit_project/
├── .gitignore                         # Excludes large data (>100MB), binary models, caches
├── README.md                          # Quickstart guide & project runbook
├── TEAM_TASK_CHECKLIST.md             # Team task breakdown, notebook ownership & milestone checklist
├── requirements.txt                   # Pinned python dependencies
│
├── configs/                           # Declarative pipeline configurations
│   ├── data_config.yaml               # Relational paths & feature cohort definitions
│   └── model_config.yaml              # Hyperparameters (Baseline, LightGBM, TensorFlow)
│
├── data/                              # Raw relational tables and processed feature matrices
│   ├── raw/
│   │   └── home_credit_default_risk/  # Kaggle CSV tables (307k primary, 13.6M installment rows)
│   └── processed/                     # Cleaned & partitioned parquet matrices for Colab execution
│
├── notebooks/                         # Core exploratory & experimental notebooks (1 per team member)
│   ├── 01_eda_home_credit.ipynb       # Track 1: Static application EDA, missingness & demographic plots
│   ├── 02_eda_relational_tables.ipynb # Track 2: Multi-table joins, installment & cash balance aggregations
│   ├── 03_alternative_features.ipynb  # Track 3: Cohort partitioning, baseline scorecards & LightGBM tuning
│   └── 04_model_experiments.ipynb     # Track 4: TensorFlow Deep Tabular NN, ROC/PR curves & subgroup analysis
│
├── saved_models/                      # Serialized model checkpoints (.joblib, .keras)
│
├── src/                               # Modular production python source code
│   ├── __init__.py
│   ├── data/
│   │   ├── download_datasets.py       # Automated Kaggle dataset downloader
│   │   └── preprocess.py              # Multi-table aggregations, missingness imputation & parquet splits
│   ├── features/
│   │   ├── build_features.py          # Cohort generator (Traditional vs. Alternative vs. Hybrid)
│   │   └── synthetic_generator.py     # Feature engineering & interaction domain ratios
│   ├── models/
│   │   ├── baseline.py                # Logistic Regression (ElasticNet) & Decision Tree baselines
│   │   ├── tree_models.py             # LightGBM & XGBoost tree ensemble routines
│   │   └── tf_neural_net.py           # TensorFlow 2.x Deep Tabular Neural Network (Entity Embeddings)
│   ├── evaluation/
│   │   └── metrics.py                 # Imbalance metrics: ROC-AUC, PR-AUC, Brier score, Expected Loss
│   ├── visualization/
│   │   └── plots.py                   # Reusable plotting tools for ROC, PR, SHAP, and distributions
│   └── utils.py                       # Logging, random seed management, and DataFrame I/O
│
├── reports/                           # Academic deliverables & milestone reports
│   ├── proposal/                      # Project proposal deliverable
│   │   └── final_project_proposal.pdf # Gradescope submitted proposal
│   ├── milestone/                     # Milestone deliverable (Due Oct 12, 2026, <=3 pages)
│   │   ├── milestone_report.md        # 3-page milestone writeup draft
│   │   ├── milestone_task_breakdown.md# Milestone task checklist
│   │   └── figures/                   # Labeled visual figures for report
│   ├── final_report/                  # Final term paper (<=5 pages)
│   └── presentation/                  # 12-minute slide deck assets
│
└── tests/                             # Directory for tests to run across project
    └── test_pipeline.py               # Pipeline validation tests (preprocessing, cohorts, models)
```

</details>
<!-- COPY UNTIL HERE -->
