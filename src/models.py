"""
Model builders for NYC Taxi duration prediction.
Includes a simple baseline and Keras MLP architectures used in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Baseline
from sklearn.linear_model import Ridge

def build_baseline_model(alpha: float = 1.0) -> Ridge:
    """Baseline regression model (fast benchmark)."""
    return Ridge(alpha=alpha, random_state=42)


# Keras MLP models
def _require_tf():
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        raise RuntimeError("TensorFlow is required for neural network models. Install from requirements.txt") from e


@dataclass
class MLPConfig:
    hidden_units: List[int]
    activation: str
    learning_rate: float


def build_mlp(input_dim: int, cfg: MLPConfig):
    _require_tf()
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for u in cfg.hidden_units:
        model.add(tf.keras.layers.Dense(u, activation=cfg.activation))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


MODEL_CONFIGS = {
    "model1": MLPConfig(hidden_units=[64], activation="tanh", learning_rate=0.003),
    "model2": MLPConfig(hidden_units=[128, 64], activation="relu", learning_rate=0.001),
    "model3": MLPConfig(hidden_units=[256, 128, 64], activation="relu", learning_rate=0.0005),
}
