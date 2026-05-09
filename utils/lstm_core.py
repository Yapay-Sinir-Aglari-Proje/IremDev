"""
PyTorch LSTM ile kullanılan ortak yapı taşları.

- `load_aggregate_frame`: processed.parquet içindeki MinMax ölçekli sütunları
  zaman damgasına göre ortalar (çok lot → tek zaman serisi).
- `make_sequences` / `prepend_context`: train_lstm ve inference ile aynı pencere mantığı.
- `OccupancyLSTM`: çok girdili LSTM; çıktı tek skaler (bir sonraki adımın occupancy_rate_mm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch.nn as nn

from utils.data_pipeline import FEATURE_COLS_FOR_SCALING


def mm_cols() -> list[str]:
    """Ölçeklenmiş özellik sütun adları (data_pipeline.FEATURE_COLS_FOR_SCALING + '_mm')."""
    return [f"{c}_mm" for c in FEATURE_COLS_FOR_SCALING]


def load_aggregate_frame(parquet_path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    cols = mm_cols()
    agg = (
        df.groupby("LastUpdated", sort=False)
        .agg({**{c: "mean" for c in cols}, "split": "first"})
        .reset_index()
        .sort_values("LastUpdated", kind="mergesort")
        .reset_index(drop=True)
    )
    return agg


def prepend_context(prev: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Önceki split’in son satırlarını başa ekleyerek val/test dizilerinde bağlam kaybını önler."""
    if len(prev) == 0:
        return block
    return np.vstack([prev, block])


def make_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for i in range(len(data) - time_step):
        x_list.append(data[i : i + time_step])
        y_list.append(data[i + time_step, 0])
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


class OccupancyLSTM(nn.Module):
    """İki katmanlı LSTM + dropout + tek çıkışlık doğrusal başlık."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        y, _ = self.lstm(x)
        last = y[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def inv_occupancy_rate(y_mm: np.ndarray, scaler) -> np.ndarray:
    """MinMax ile ölçeklenmiş tahmini gerçek [0,1] doluluk oranı aralığına çevirir."""
    i = FEATURE_COLS_FOR_SCALING.index("occupancy_rate")
    lo, hi = float(scaler.data_min_[i]), float(scaler.data_max_[i])
    return y_mm * (hi - lo) + lo
