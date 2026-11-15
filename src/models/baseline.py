import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tf_logreg(n_features: int, l2_reg: float = 1e-4, lr: float = 1e-3) -> keras.Model:
    """Construct and compile a regularized Logistic Regression model in TensorFlow Keras."""
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        ),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')],
    )
    return model
