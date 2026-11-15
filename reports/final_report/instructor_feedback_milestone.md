# Instructor Feedback: Milestone Report (Professor Uri)

**Date Received**: October 26, 2026  
**Context**: Review of `Final Project Milestone Submission.pdf`  
**Purpose**: Guidance and evaluation criteria for the Final Project Report and Notebooks.

---

## 1. Core Evaluation Directives (Must Include in Final Deliverables)

### A. Report Precision and Recall Separately
- **Professor Note**: *"My one main comment is that I always like to see precision and recall separately as well... Personally, I am always also interested in seeing precision and recall separately."*
- **Action for Notebooks (`03`, `04`) & Final Report**:
  - Do not rely solely on aggregate threshold-free metrics (ROC-AUC, PR-AUC).
  - Explicitly tabulate **Precision**, **Recall**, **F1-score**, and **Specificity** at calibrated decision thresholds ($\tau \in \{0.10, 0.20, 0.50\}$) across all models.
  - Frame trade-offs in credit default context: 
    - **Recall (Sensitivity)**: Catching actual defaulters (minimizing credit losses).
    - **Precision**: Avoiding false alarms that turn away creditworthy applicants.

### B. Evaluate Table Joins (Match Rates & Cardinality)
- **Professor Note**: *"If you can, try and examine and/or evaluate the join of the four tables (are there duplicate matches, missing matches, bad matches, etc.)"*
- **Action for Notebook (`02`) & Dataset Section**:
  - Document join coverage per `SK_ID_CURR`:
    - Bureau match rate (~84% matched, ~16% thin-file with zero bureau lines).
    - Previous application match rate and installments coverage.
    - Cardinality distributions (min, median, max records per primary ID).
    - Confirm zero duplicate index collisions or fan-out errors after aggregation.

### C. Missingness Impact & Information Value
- **Professor Note**: *"Missing values... we keep them rather than drop them. Appreciated! Always worth investigating and evaluating whether a missing value means something and how performance is affected, with and without."*
- **Action for Notebooks & Methods/Discussion**:
  - Emphasize that missingness in external scores (`EXT_SOURCE_1` missing 56%) and bureau history is informative domain signal, not random sensor drop.
  - Document performance with explicit missingness indicator flags vs. native LightGBM missingness routing.

### D. Big Data Scaling & Alternative Architectures
- **Professor Note**: *"did not fit in Colab's 12.7 GB of RAM... downcast numeric types. A few things to consider or try. You can do the join outside of colab using tools that can deal with limited memory (maybe spark), you can try sampling (not sure if that's better or worse than downcasting types)."*
- **Action for Engineering Challenges & Future Work Sections**:
  - Justify downcasting + snappy Parquet as zero-information-loss vs. down-sampling (which sacrifices minority class default instances in an already imbalanced 8% problem).
  - Highlight PySpark / Polars / DuckDB out-of-core streaming as production scaling alternatives in the Conclusions/Future Work section.

---

## 2. Positive Feedback to Preserve

- **Motivation**: Problem definition of "thin-file" borrowers and clearly stated hypotheses are strong.
- **Data Engineering**: Snappy Parquet export confirmed as the correct pipeline architecture.
- **Model Progression**: Full evaluation hierarchy across Logistic Regression $\rightarrow$ Decision Trees $\rightarrow$ LightGBM $\rightarrow$ Neural Networks strongly validated.
