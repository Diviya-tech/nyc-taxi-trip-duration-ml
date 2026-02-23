"""
Preprocessing utilities for NYC Taxi Trip Duration.
Designed to mirror the steps described in the project write-up:
- Drop non-numeric columns
- Median imputation
- Align columns
- log1p transform on target
- Standard scaling
- Train/validation split
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class StandardScaler:
    """Minimal standard scaler (mean 0, std 1) to avoid hidden magic."""
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # avoid divide-by-zero
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric columns."""
    return df.select_dtypes(include=[np.number]).copy()


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with per-column median."""
    return df.fillna(df.median(numeric_only=True))


def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure identical columns + ordering."""
    test_df = test_df.reindex(columns=train_df.columns)
    return train_df, test_df


def prepare_arrays(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Returns:
      X_tr, X_val, X_te, y_tr, y_val, scaler
    """
    X_train = median_impute(keep_numeric(X_train))
    X_test = median_impute(keep_numeric(X_test))
    X_train, X_test = align_columns(X_train, X_test)

    X = X_train.to_numpy(dtype=np.float32)
    X_te = X_test.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_te = scaler.transform(X_te)

    if y_train is None:
        return X, None, X_te, None, None, scaler

    y = np.log1p(np.asarray(y_train, dtype=np.float32))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    return X_tr, X_val, X_te, y_tr, y_val, scaler
