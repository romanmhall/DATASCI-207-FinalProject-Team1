from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NOT_F = ['SK_ID_CURR', 'TARGET']

CB_F = [
    'BUREAU_ACTIVE_LINES',
    'BUREAU_TOTAL_LINES',
    'BUREAU_TOTAL_OVERDUE_DEBT',
    'BUREAU_MAX_DAYS_OVERDUE',
    'BUREAU_TOTAL_DEBT',
    'BUREAU_OLDEST_CREDIT_DAYS',
    'HAS_BUREAU_HISTORY',
]

BEH_F = [
    'PREV_REFUSAL_COUNT',
    'PREV_APPLICATION_COUNT',
    'PREV_APPROVAL_RATE',
    'PREV_MEAN_DOWN_PAYMENT_PCT',
    'POS_MONTH_COUNT',
    'POS_MOST_RECENT_MONTH',
    'POS_MEAN_DPD',
    'POS_MAX_DPD',
    'POS_MEAN_DPD_DEF',
    'POS_COMPLETED_COUNT',
    'POS_ACTIVE_COUNT',
    'INST_DELAY_MEAN',
    'INST_DELAY_MAX',
    'INST_UNDERPAYMENT_FREQ',
    'INST_TOTAL_PAYMENTS',
]

EXT_F = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

RATIO_F = ['PAYMENT_RATE', 'INCOME_CREDIT_RATIO', 'ANNUITY_INCOME_RATIO']


def get_static_features(df: pd.DataFrame) -> List[str]:
    """Extract static applicant features excluding IDs, targets, and engineered subsets."""
    excluded = set(NOT_F + CB_F + BEH_F + EXT_F + RATIO_F)
    return [col for col in df.columns if col not in excluded]


def get_cohort_definitions(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dictionary of the four experimental feature cohorts."""
    static_f = get_static_features(df)
    return {
        'Cohort 1': static_f + CB_F,
        'Cohort 2': BEH_F + EXT_F,
        'Cohort 3': static_f + CB_F + BEH_F + EXT_F + RATIO_F,
        'Cohort 4': static_f + BEH_F + RATIO_F,
    }


def prepare_features(
    df: pd.DataFrame,
    features: List[str],
    fit_idx: pd.Index,
    es_idx: pd.Index,
    val_idx: pd.Index,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Clean, impute, encode, and standardize features without test data leakage.

    Numeric missing values are imputed strictly using the fit cohort medians.
    Categorical variables are filled with 'Missing' and one-hot encoded.
    All matrices are standardized using fit cohort statistics and clipped to [-5, 5].
    """
    X_fit = df.loc[fit_idx, features].copy()
    X_es = df.loc[es_idx, features].copy()
    X_val = df.loc[val_idx, features].copy()

    text_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in features if c not in text_cols]

    # Impute numeric missing values strictly with training fit median
    medians = X_fit[num_cols].median()
    for X in (X_fit, X_es, X_val):
        X[num_cols] = X[num_cols].fillna(medians)

    # Impute categorical missing values and one-hot encode
    for X in (X_fit, X_es, X_val):
        for c in text_cols:
            X[c] = X[c].astype('object').fillna('Missing')

    X_fit = pd.get_dummies(X_fit, columns=text_cols, dtype=float)
    X_es = pd.get_dummies(X_es, columns=text_cols, dtype=float).reindex(columns=X_fit.columns, fill_value=0.0)
    X_val = pd.get_dummies(X_val, columns=text_cols, dtype=float).reindex(columns=X_fit.columns, fill_value=0.0)

    # Standardize strictly using training fit statistics and clip outliers
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = np.clip(scaler.transform(X_fit), -5, 5).astype('float32')
    X_es_scaled = np.clip(scaler.transform(X_es), -5, 5).astype('float32')
    X_val_scaled = np.clip(scaler.transform(X_val), -5, 5).astype('float32')

    return X_fit_scaled, X_es_scaled, X_val_scaled, list(X_fit.columns)
