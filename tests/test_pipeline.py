import numpy as np
import pandas as pd

from src.utils import set_seeds
from src.features.build_features import (
    CB_F,
    BEH_F,
    EXT_F,
    RATIO_F,
    get_cohort_definitions,
    prepare_features,
)
from src.models.baseline import build_tf_logreg
from src.models.tf_neural_net import build_tf_neural_net
from src.models.tree_models import build_lightgbm_classifier
from src.evaluation.metrics import (
    evaluate_predictions,
    evaluate_threshold_table,
    best_f1_threshold,
)


def test_seed_setting():
    set_seeds(42)
    val1 = np.random.rand()
    set_seeds(42)
    val2 = np.random.rand()
    assert val1 == val2


def test_cohort_definitions():
    mock_df = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'TARGET': [0, 1, 0],
        'CODE_GENDER': ['M', 'F', 'M'],
        'BUREAU_ACTIVE_LINES': [1, 2, 0],
        'BUREAU_TOTAL_LINES': [2, 3, 0],
        'BUREAU_TOTAL_OVERDUE_DEBT': [0.0, 0.0, 0.0],
        'BUREAU_MAX_DAYS_OVERDUE': [0, 0, 0],
        'BUREAU_TOTAL_DEBT': [100.0, 200.0, 0.0],
        'BUREAU_OLDEST_CREDIT_DAYS': [1000, 2000, 0],
        'HAS_BUREAU_HISTORY': [1, 1, 0],
        'PREV_REFUSAL_COUNT': [0, 1, 0],
        'PREV_APPLICATION_COUNT': [1, 2, 0],
        'PREV_APPROVAL_RATE': [1.0, 0.5, 0.0],
        'PREV_MEAN_DOWN_PAYMENT_PCT': [0.1, 0.2, 0.0],
        'POS_MONTH_COUNT': [10, 12, 0],
        'POS_MOST_RECENT_MONTH': [-1, -2, 0],
        'POS_MEAN_DPD': [0, 0, 0],
        'POS_MAX_DPD': [0, 0, 0],
        'POS_MEAN_DPD_DEF': [0, 0, 0],
        'POS_COMPLETED_COUNT': [1, 1, 0],
        'POS_ACTIVE_COUNT': [1, 0, 0],
        'INST_DELAY_MEAN': [0.0, 1.0, 0.0],
        'INST_DELAY_MAX': [0.0, 5.0, 0.0],
        'INST_UNDERPAYMENT_FREQ': [0.0, 0.1, 0.0],
        'INST_TOTAL_PAYMENTS': [10, 12, 0],
        'EXT_SOURCE_1': [0.5, 0.4, np.nan],
        'EXT_SOURCE_2': [0.6, 0.5, 0.4],
        'EXT_SOURCE_3': [0.7, 0.6, 0.5],
        'PAYMENT_RATE': [0.05, 0.06, 0.04],
        'INCOME_CREDIT_RATIO': [0.3, 0.2, 0.4],
        'ANNUITY_INCOME_RATIO': [0.1, 0.15, 0.08],
    })
    cohorts = get_cohort_definitions(mock_df)
    assert 'Cohort 1' in cohorts
    assert 'Cohort 2' in cohorts
    assert 'Cohort 3' in cohorts
    assert 'Cohort 4' in cohorts
    assert 'CODE_GENDER' in cohorts['Cohort 1']


def test_prepare_features_leak_free():
    df = pd.DataFrame({
        'num1': [10.0, np.nan, 30.0, 40.0, 50.0],
        'cat1': ['A', 'B', np.nan, 'A', 'B'],
        'TARGET': [0, 0, 1, 0, 1],
    })
    fit_idx = pd.Index([0, 1, 2])
    es_idx = pd.Index([3])
    val_idx = pd.Index([4])

    X_fit, X_es, X_val, cols = prepare_features(df, ['num1', 'cat1'], fit_idx, es_idx, val_idx)
    assert X_fit.shape[0] == 3
    assert X_es.shape[0] == 1
    assert X_val.shape[0] == 1
    assert not np.isnan(X_fit).any()
    assert not np.isnan(X_es).any()
    assert not np.isnan(X_val).any()


def test_models_compilation():
    logreg = build_tf_logreg(n_features=10)
    assert logreg.output_shape == (None, 1)

    nn = build_tf_neural_net(n_features=10, units=32)
    assert nn.output_shape == (None, 1)

    lgb_model = build_lightgbm_classifier()
    assert lgb_model.scale_pos_weight == 11.39


def test_evaluation_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.05, 0.3, 0.8, 0.6, 0.4, 0.7])
    credit = np.array([1000, 2000, 1500, 3000, 5000, 4000, 2500, 6000])

    m = evaluate_predictions(y_true, y_prob, threshold=0.5, credit_amounts=credit)
    assert m['ROC-AUC'] > 0.8
    assert 'Precision' in m
    assert 'Recall' in m
    assert 'Predicted_Loss_M' in m

    thresh_df = evaluate_threshold_table(y_true, y_prob, thresholds=[0.2, 0.5])
    assert len(thresh_df) == 2
    assert 'Precision' in thresh_df.columns
    assert 'Recall (Sensitivity)' in thresh_df.columns
