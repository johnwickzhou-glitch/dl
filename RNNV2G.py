import os
import math
import json
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


@dataclass
class SplitInfo:
    train_rows: int
    val_rows: int
    train_moment_counts: Dict[str, int]
    val_moment_counts: Dict[str, int]
    lookback: int
    target_shift: int
    value_col: str
    moment_col: str
    number_col: str
    sort_col: str
    afternoon_train_ratio: float
    seed: int


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StandardScaler1D:
    """Z-score scaler fitted on training data only."""
    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.eps = eps

    def fit(self, x: np.ndarray):
        x = x.astype(np.float32)
        self.mean = float(np.mean(x))
        self.std = float(np.std(x) + self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class TimeSeqDataset(Dataset):
    """
    For each (Moment, number), sort by Distance(mts) and build one-step samples:
      X: past `lookback` Time(sec) values, shape [lookback, 1]
      y: next Time(sec), shape [1]
    """
    def __init__(self,
                 df: pd.DataFrame,
                 lookback: int,
                 target_shift: int,
                 moment_col: str,
                 number_col: str,
                 value_col: str,
                 sort_col: str):
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for (_, _), g in df.groupby([moment_col, number_col], sort=False):
            if sort_col in g.columns:
                g = g.sort_values(sort_col, kind="mergesort")
            else:
                g = g.sort_index()

            arr = pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy(dtype=np.float32)

            L, s = lookback, target_shift
            if len(arr) < L + s:
                continue

            for i in range(len(arr) - L - s + 1):
                X_list.append(arr[i:i + L].reshape(L, 1))
                y_list.append(arr[i + L + s - 1].reshape(1))

        self.X = np.stack(X_list, axis=0) if X_list else np.zeros((0, lookback, 1), dtype=np.float32)
        self.y = np.stack(y_list, axis=0) if y_list else np.zeros((0, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class RNNOneStepRegressor(nn.Module):
    """
    Vanilla RNN (tanh) + linear head for one-step regression.
    Note: input is z-scored, so tanh (-1~1) is appropriate.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # [B,T,H]
        return self.fc(out[:, -1, :]) # [B,1]


def calc_metrics_std(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    Metrics on standardized (z-scored) values.

    Returns: MSE / RMSE / MAE / MAPE
    - MAPE follows a signed definition (can be negative):
        mean((y_pred - y_true) / y_true)
      (No percentage scaling; no absolute value.)
    - Values with |y_true| < eps are skipped to avoid division-by-zero.
    """
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    mask = np.abs(y_true) >= eps
    if np.any(mask):
        mape = float(np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]))
    else:
        mape = float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def make_split(df: pd.DataFrame,
               moment_col: str,
               number_col: str,
               value_col: str,
               sort_col: str,
               afternoon_train_ratio: float,
               seed: int,
               lookback: int,
               target_shift: int) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Train: all morning + all noon + a subset of afternoon (by number).
    Val:   remaining afternoon (by number).
    """
    df = df.copy()
    df[moment_col] = df[moment_col].astype(str).str.strip().str.lower()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[moment_col, value_col]).reset_index(drop=True)

    aft = df[df[moment_col] == "afternoon"].copy()
    if len(aft) == 0:
        raise ValueError("No afternoon rows found. Check Moment values.")

    nums = sorted(aft[number_col].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(nums)

    k = max(1, int(len(nums) * afternoon_train_ratio))
    train_nums = set(nums[:k])
    val_nums = set(nums[k:])

    train_df = df[df[moment_col].isin(["morning", "noon"])].copy()
    train_df = pd.concat([train_df, aft[aft[number_col].isin(train_nums)]], ignore_index=True)
    val_df = aft[aft[number_col].isin(val_nums)].copy().reset_index(drop=True)

    info = SplitInfo(
        train_rows=len(train_df),
        val_rows=len(val_df),
        train_moment_counts=train_df[moment_col].value_counts().to_dict(),
        val_moment_counts=val_df[moment_col].value_counts().to_dict(),
        lookback=lookback,
        target_shift=target_shift,
        value_col=value_col,
        moment_col=moment_col,
        number_col=number_col,
        sort_col=sort_col,
        afternoon_train_ratio=afternoon_train_ratio,
        seed=seed,
    )
    return train_df.reset_index(drop=True), val_df, info


def eval_mse_std(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def save_train_loss_plot(losses: List[float], out_png: str, title: str) -> None:
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE Loss (std)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    seed = 42
    lookback = 4
    target_shift = 1
    batch_size = 64
    max_epochs = 80
    lr = 1e-3
    afternoon_train_ratio = 0.6

    patience = 15
    min_delta = 1e-5

    csv_path = "./data/newdata.csv"
    moment_col = "Moment"
    number_col = "number"
    value_col = "Time(sec)"
    sort_col = "Distance(mts)"

    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    model_tag = "RNNV2G"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CSV:", csv_path)

    df = pd.read_csv(csv_path)
    for c in [moment_col, number_col, value_col]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}. Found: {list(df.columns)}")

    train_df, val_df, info = make_split(
        df=df,
        moment_col=moment_col,
        number_col=number_col,
        value_col=value_col,
        sort_col=sort_col,
        afternoon_train_ratio=afternoon_train_ratio,
        seed=seed,
        lookback=lookback,
        target_shift=target_shift,
    )
    print("\n===== Split Info =====")
    print(info)

    scaler = StandardScaler1D().fit(train_df[value_col].to_numpy(dtype=np.float32))
    train_df[value_col] = scaler.transform(train_df[value_col].to_numpy(dtype=np.float32))
    val_df[value_col] = scaler.transform(val_df[value_col].to_numpy(dtype=np.float32))

    train_ds = TimeSeqDataset(train_df, lookback, target_shift, moment_col, number_col, value_col, sort_col)
    val_ds = TimeSeqDataset(val_df, lookback, target_shift, moment_col, number_col, value_col, sort_col)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty dataset after windowing: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    xb, yb = next(iter(train_loader))
    print(f"\n[Shape Check] X: {tuple(xb.shape)} Y: {tuple(yb.shape)}")

    model = RNNOneStepRegressor(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_losses: List[float] = []
    best_val = float("inf")
    best_path = os.path.join(result_dir, f"{model_tag}_best.pt")

    bad = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        train_mse = total / max(n, 1)
        train_losses.append(train_mse)

        val_mse = eval_mse_std(model, val_loader, device)

        improved = (best_val - val_mse) > min_delta
        if improved:
            best_val = val_mse
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep:03d}/{max_epochs} | Train MSE: {train_mse:.6f} | Val MSE(std): {val_mse:.6f} | Best: {best_val:.6f}")

        if bad >= patience:
            print(f"\n[EarlyStopping] Stop at epoch {ep}, best Val MSE(std) = {best_val:.6f}")
            break

    train_json = os.path.join(result_dir, f"{model_tag}_train_loss.json")
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump({"train_mse_loss": train_losses}, f, indent=2)

    train_png = os.path.join(result_dir, f"{model_tag}_train_loss.png")
    save_train_loss_plot(train_losses, train_png, f"{model_tag} Train Loss")

    print("\nSaved train loss json:", train_json)
    print("Saved train loss png :", train_png)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            ps.append(pred)
            ys.append(y.numpy())

    y_pred_std = np.concatenate(ps, axis=0)
    y_true_std = np.concatenate(ys, axis=0)

    metrics_std = calc_metrics_std(y_true_std, y_pred_std)
    metrics_path = os.path.join(result_dir, f"{model_tag}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_std, f, indent=2)

    print("\n===== VAL/TEST METRICS (standardized / z-score) =====")
    for k, v in metrics_std.items():
        print(f"{k}: {v:.6f}")
    print("Saved metrics to:", metrics_path)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
