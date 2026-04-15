"""
LSTM doluluk tahminlerinden RL uyumlu state feature'ları üretir.

Bu modül, kısa vadeli otopark doluluk tahminlerini
gelecekte Gymnasium tabanlı DQN/PPO ortamlarında ve
FastAPI servislerinde kullanılabilecek yapılandırılmış
durum feature'larına dönüştürür.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "timestamp",
    "parking_id",
    "predicted_occupancy",
    "capacity",
    "latitude",
    "longitude",
}

DEFAULT_INPUT_PATH = Path("data/processed/lstm_predictions.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/rl_prediction_states.csv")


def load_predictions(input_path: Path) -> pd.DataFrame:
    """Tahmin verisini yükler ve gerekli kolon yapısını doğrular."""

    if not input_path.exists():
        raise FileNotFoundError(f"Tahmin dosyası bulunamadı: {input_path}")

    try:
        predictions = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(
            f"Tahmin dosyası okunamadı: {input_path}"
        ) from exc

    missing_cols = REQUIRED_COLUMNS - set(predictions.columns)

    if missing_cols:
        raise ValueError(
            f"Eksik gerekli kolonlar: {sorted(missing_cols)}. "
            f"Bulunan kolonlar: {sorted(predictions.columns)}"
        )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], errors="coerce"
    )

    if predictions["timestamp"].isna().any():
        raise ValueError("Tahmin verisinde geçersiz timestamp değerleri bulundu.")

    return predictions


def _normalize_empty_slots_within_timestamp(df: pd.DataFrame) -> pd.Series:
    """Aynı timestamp içindeki boş slot sayılarını 0-1 arasında normalize eder."""

    grouped = df.groupby("timestamp")["predicted_empty_slots"]

    min_vals = grouped.transform("min")
    max_vals = grouped.transform("max")

    denominator = (max_vals - min_vals).replace(0, np.nan)

    normalized = (df["predicted_empty_slots"] - min_vals) / denominator

    # Aynı zaman diliminde tüm değerler eşitse 1.0 atanır
    return normalized.fillna(1.0).clip(0.0, 1.0)


def create_rl_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Doluluk tahminlerinden RL feature'larını üretir."""

    df = predictions.copy()

    if (df["capacity"] <= 0).any():
        raise ValueError("Tüm satırlarda capacity değeri sıfırdan büyük olmalıdır.")

    # Doluluk değerini fiziksel aralıkta sınırla
    df["predicted_occupancy"] = df["predicted_occupancy"].clip(lower=0)
    df["predicted_occupancy"] = np.minimum(
        df["predicted_occupancy"], df["capacity"]
    )

    # Boş slot sayısı
    df["predicted_empty_slots"] = (
        df["capacity"] - df["predicted_occupancy"]
    )

    # Doluluk oranı
    df["occupancy_ratio"] = (
        df["predicted_occupancy"] / df["capacity"]
    ).clip(0.0, 1.0)

    # Availability score
    df["availability_score"] = _normalize_empty_slots_within_timestamp(df)

    # Risk seviyeleri
    conditions = [
        df["occupancy_ratio"] < 0.5,
        (df["occupancy_ratio"] >= 0.5)
        & (df["occupancy_ratio"] < 0.8),
        df["occupancy_ratio"] >= 0.8,
    ]

    risk_levels = ["LOW", "MEDIUM", "HIGH"]
    df["risk_level"] = np.select(
        conditions, risk_levels, default="HIGH"
    )

    # Saat bilgisi (cyclical encoding)
    hour = df["timestamp"].dt.hour.astype(float)

    df["time_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["time_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Placeholder feature
    df["distance_placeholder"] = 0.0

    # Beklenen bekleme süresi
    df["expected_wait_time"] = df["occupancy_ratio"] * 10.0

    return df


def build_state_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Seçilen sayısal feature'lardan RL state vector oluşturur."""

    vector_columns = [
        "occupancy_ratio",
        "predicted_empty_slots",
        "availability_score",
        "expected_wait_time",
        "time_sin",
        "time_cos",
    ]

    def _to_vector(row: pd.Series) -> List[float]:
        return [float(row[col]) for col in vector_columns]

    df = df.copy()

    df["rl_state_vector"] = df[vector_columns].apply(
        _to_vector, axis=1
    )

    return df


def save_rl_states(df: pd.DataFrame, output_path: Path) -> None:
    """RL için hazır state tablosunu CSV olarak kaydeder."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False)
    except Exception as exc:
        raise IOError(
            f"RL state dosyası kaydedilemedi: {output_path}"
        ) from exc


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Uçtan uca pipeline: yükle -> feature üret -> state vector -> kaydet."""

    try:
        predictions = load_predictions(input_path)

        rl_features = create_rl_features(predictions)
        rl_states = build_state_vector(rl_features)

        save_rl_states(rl_states, output_path)

        print("RL state hazırlama başarıyla tamamlandı.")

    except Exception as exc:
        LOGGER.exception(
            "RL state hazırlama başarısız oldu: %s", exc
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    main()