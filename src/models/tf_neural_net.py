from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tf_neural_net(
    n_features: int,
    units: int = 128,
    dropout: float = 0.3,
    lr: float = 1e-3,
) -> keras.Model:
    """Construct and compile a Deep Tabular Neural Network with BatchNorm and Dropout."""
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')],
    )
    return model


def train_tf_model(
    model: keras.Model,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_es: np.ndarray,
    y_es: np.ndarray,
    epochs: int = 40,
    batch_size: int = 1024,
    patience: int = 5,
    verbose: int = 0,
) -> Tuple[keras.Model, Dict[str, list]]:
    """Train a TensorFlow model with EarlyStopping on validation ROC-AUC."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_es, y_es),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return model, history.history
