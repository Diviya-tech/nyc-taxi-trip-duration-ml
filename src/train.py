"""
Train script (optional) to reproduce results outside notebooks.

Usage:
  python -m src.train --data_path data/nyc_taxi_data.npy --model model3
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .preprocessing import prepare_arrays
from .models import build_baseline_model, build_mlp, MODEL_CONFIGS


def load_bundle(path: str):
    bundle = np.load(path, allow_pickle=True).item()
    return bundle["X_train"], bundle["X_test"], bundle["y_train"], bundle.get("y_test", None)


def eval_regression(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model", choices=["baseline", "model1", "model2", "model3"], default="model3")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    X_train, X_test, y_train, y_test = load_bundle(args.data_path)

    X_tr, X_val, X_te, y_tr, y_val, _ = prepare_arrays(X_train, X_test, y_train=y_train)

    if args.model == "baseline":
        m = build_baseline_model(alpha=1.0)
        m.fit(X_tr, y_tr)
        val_pred = m.predict(X_val)
        print("Validation:", eval_regression(y_val, val_pred))
        return

    cfg = MODEL_CONFIGS[args.model]
    model = build_mlp(input_dim=X_tr.shape[1], cfg=cfg)

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred = model.predict(X_val).reshape(-1)
    metrics = eval_regression(y_val, val_pred)
    print("Validation:", metrics)

    # If y_test exists, evaluate on test too
    if y_test is not None:
        # NOTE: to evaluate on test, you must preprocess X_test the same way; here we used prepare_arrays output X_te
        y_test_t = np.log1p(np.asarray(y_test, dtype=np.float32))
        test_pred = model.predict(X_te).reshape(-1)
        print("Test:", eval_regression(y_test_t, test_pred))


if __name__ == "__main__":
    main()
