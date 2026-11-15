from typing import Any, Dict, Optional
import lightgbm as lgb
import numpy as np


def build_lightgbm_classifier(
    scale_pos_weight: float = 11.39,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    n_estimators: int = 300,
    random_state: int = 1234,
    **kwargs: Any,
) -> lgb.LGBMClassifier:
    """Build a tuned LightGBM Classifier configured for severe class imbalance."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMClassifier:
    """Fit a LightGBM Classifier on training arrays with optional validation evaluation."""
    if params is None:
        params = {}
    model = build_lightgbm_classifier(**params)
    if X_val is not None and y_val is not None:
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
        except Exception:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model
