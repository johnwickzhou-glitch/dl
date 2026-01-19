# -*- coding: utf-8 -*-
"""
dataPrepare_v3.py

Data pipeline for `newdata.csv` (one-step forecasting).

- Target: Time(sec)
- Sample: X = past `lookback` Time(sec), y = next Time(sec)
- Grouping: each (Moment, number) forms an independent sequence
- Split: train = morning + noon + part of afternoon (by number),
         test  = remaining afternoon (by number)
- Normalization: z-score fitted on training data only
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler1D:
    """Z-score scaler for 1D values (fit on training data only)."""
    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler1D":
        x = np.asarray(x, dtype=np.float32)
        self.mean = float(np.mean(x))
        self.std = float(np.std(x) + self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * self.std + self.mean


class TimeSecWindowDataset(Dataset):
    """
    Sliding-window dataset for one-step prediction.

    X: [lookback, 1]  (past Time(sec))
    y: [1]            (next Time(sec))
    """
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int,
        target_shift: int,
        value_col: str,
        group_col: str,
        sort_col: Optional[str] = None,
    ):
        self.lookback = int(lookback)
        self.target_shift = int(target_shift)
        self.value_col = value_col
        self.group_col = group_col
        self.sort_col = sort_col

        X_list = []
        y_list = []

        for _, g in df.groupby(group_col):
            # Optional: define within-sequence order via a stable sort.
            if sort_col and sort_col in g.columns:
                g = g.sort_values(sort_col, kind="mergesort")  # stable
            arr = g[value_col].to_numpy(dtype=np.float32)

            L = self.lookback
            s = self.target_shift
            if len(arr) < L + s:
                continue

            for i in range(0, len(arr) - L - s + 1):
                x = arr[i:i + L].reshape(L, 1)
                y = arr[i + L + s - 1].reshape(1)
                X_list.append(x)
                y_list.append(y)

        if len(X_list) == 0:
            self.X = np.zeros((0, self.lookback, 1), dtype=np.float32)
            self.y = np.zeros((0, 1), dtype=np.float32)
        else:
            self.X = np.stack(X_list, axis=0).astype(np.float32)
            self.y = np.stack(y_list, axis=0).astype(np.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


@dataclass
class SplitInfo:
    """Split summary and key configuration."""
    train_rows: int
    test_rows: int
    train_moment_counts: Dict[str, int]
    test_moment_counts: Dict[str, int]
    lookback: int
    target_shift: int
    value_col: str
    moment_col: str
    number_col: str
    group_col: str
    afternoon_train_ratio: float
    seed: int


def _load_and_clean(
    csv_path: str,
    moment_col: str,
    number_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Load csv, sanitize columns, and build group_id = Moment_number."""
    df = pd.read_csv(csv_path)

    need = [moment_col, number_col, value_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"CSV missing column `{c}`. Found columns: {list(df.columns)}")

    df[moment_col] = df[moment_col].astype(str).str.strip().str.lower()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[moment_col, number_col, value_col]).reset_index(drop=True)

    # One sequence per (Moment, number).
    df["group_id"] = df[moment_col].astype(str) + "_" + df[number_col].astype(str)
    return df


def get_dataset_time_v3(
    batch_size: int,
    csv_path: str,
    lookback: int = 6,
    target_shift: int = 1,
    value_col: str = "Time(sec)",
    moment_col: str = "Moment",
    number_col: str = "number",
    afternoon_train_ratio: float = 0.60,
    seed: int = 42,
    standardize: bool = True,
    sort_within_group: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, StandardScaler1D, SplitInfo]:
    """
    Build DataLoaders for one-step Time(sec) forecasting.

    Returns:
        train_loader, test_loader, scaler, split_info
    """
    df = _load_and_clean(csv_path, moment_col=moment_col, number_col=number_col, value_col=value_col)

    train_df = df[df[moment_col].isin(["morning", "noon"])].copy()
    aft_df = df[df[moment_col] == "afternoon"].copy()
    if len(aft_df) == 0:
        raise ValueError("No rows with Moment == 'afternoon' in csv.")

    # Split afternoon by `number` to reduce leakage.
    aft_numbers = aft_df[number_col].dropna().unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(aft_numbers)

    split_n = max(1, int(len(aft_numbers) * float(afternoon_train_ratio)))
    train_numbers = set(aft_numbers[:split_n])
    test_numbers = set(aft_numbers[split_n:])

    train_aft = aft_df[aft_df[number_col].isin(train_numbers)].copy()
    test_df = aft_df[aft_df[number_col].isin(test_numbers)].copy()

    train_df = pd.concat([train_df, train_aft], axis=0).reset_index(drop=True)

    # Fit scaler on training data only.
    scaler = StandardScaler1D()
    if standardize:
        scaler.fit(train_df[value_col].to_numpy(dtype=np.float32))
        train_df[value_col] = scaler.transform(train_df[value_col].to_numpy(dtype=np.float32))
        test_df[value_col] = scaler.transform(test_df[value_col].to_numpy(dtype=np.float32))
    else:
        scaler.fit(train_df[value_col].to_numpy(dtype=np.float32))

    # Windowing within each group_id sequence.
    train_ds = TimeSecWindowDataset(
        train_df, lookback=lookback, target_shift=target_shift,
        value_col=value_col, group_col="group_id", sort_col=sort_within_group
    )
    test_ds = TimeSecWindowDataset(
        test_df, lookback=lookback, target_shift=target_shift,
        value_col=value_col, group_col="group_id", sort_col=sort_within_group
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"Dataset too small after windowing. train_samples={len(train_ds)} test_samples={len(test_ds)}. "
            f"Try lookback=3/4 or check data."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    split_info = SplitInfo(
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_moment_counts=train_df[moment_col].value_counts().to_dict(),
        test_moment_counts=test_df[moment_col].value_counts().to_dict(),
        lookback=int(lookback),
        target_shift=int(target_shift),
        value_col=value_col,
        moment_col=moment_col,
        number_col=number_col,
        group_col="group_id",
        afternoon_train_ratio=float(afternoon_train_ratio),
        seed=int(seed),
    )
    return train_loader, test_loader, scaler, split_info
