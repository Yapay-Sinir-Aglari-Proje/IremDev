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
from tensorflow.keras.models import load_model


LOGGER = logging.getLogger(__name__)

# RL pipeline için zorunlu tahmin kolonları
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
DEFAULT_MODEL_PATH = Path("models/lstm_parking_model.h5")
DEFAULT_TEST_PATH = Path("data/processed/test.csv")
TIME_STEP = 12

# Farklı veri kaynakları için olası kolon isimleri
_TIMESTAMP_CANDIDATES = ("timestamp", "LastUpdated", "last_updated")
_PARKING_ID_CANDIDATES = ("parking_id", "ParkingID", "park_id", "SystemCodeNumber")
_CAPACITY_CANDIDATES = ("capacity", "Capacity")
_LATITUDE_CANDIDATES = ("latitude", "Latitude", "lat")
_LONGITUDE_CANDIDATES = ("longitude", "Longitude", "lon", "lng")
_OCCUPANCY_CANDIDATES = ("occupancy", "Occupancy")
_OCCUPANCY_RATE_CANDIDATES = ("occupancy_rate", "OccupancyRate")


def _require_column(df: pd.DataFrame, candidates: tuple[str, ...], field_name: str) -> str:
    """Aday kolonlar arasından ilk eşleşeni bulur, yoksa net hata döndürür."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Gerekli alan bulunamadı: {field_name}. "
        f"Beklenen kolonlardan biri: {list(candidates)}"
    )


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Aday kolonlar arasından ilk eşleşeni döndürür, yoksa None verir."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _create_sequences(values: np.ndarray, time_step: int) -> np.ndarray:
    """Tek boyutlu seri için LSTM girdi dizileri oluşturur."""

    if len(values) < time_step:
        return np.empty((0, time_step, 1), dtype=float)

    # Örn: [v1,v2,v3,v4], time_step=3 -> [v1,v2,v3], [v2,v3,v4]
    # Böylece model kaydırmalı pencere mantığıyla ardışık örnekler görür.
    sequences = [values[i : i + time_step] for i in range(len(values) - time_step + 1)]
    return np.asarray(sequences, dtype=float).reshape(-1, time_step, 1)


def generate_lstm_predictions(
    output_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    time_step: int = TIME_STEP,
) -> pd.DataFrame:
    """Eksik CSV durumunda LSTM modelini kullanarak tahmin tablosu üretir."""

    LOGGER.info("Generating LSTM predictions...")
    # Çıkış klasörü yoksa oluştur (production ortamı için güvenli)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Model dosyası yoksa fallback devam etmez, net hata ver
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model dosyası bulunamadı: {model_path}. "
            "Önce model eğitimi çalıştırılmalı."
        )

    # Test verisi, sequence tabanlı tahmin üretimi için zorunlu
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test verisi bulunamadı: {test_path}. "
            "Önce data preparation pipeline çalıştırılmalı."
        )

    # Eğitilmiş LSTM modelini diskten yükle
    try:
        # Sadece inference için modeli derlemeden yükle; Keras sürüm farkı
        # kaynaklı legacy H5 deserialize hatalarını engeller.
        model = load_model(model_path, compile=False)
    except Exception as exc:
        raise ValueError(f"LSTM model yüklenemedi: {model_path}") from exc

    # Tahmin üretilecek ham veriyi oku
    try:
        test_df = pd.read_csv(test_path)
    except Exception as exc:
        raise ValueError(f"Test verisi okunamadı: {test_path}") from exc

    # Veri setindeki gerçek kolon adlarını aday listelerden çöz
    timestamp_col = _require_column(test_df, _TIMESTAMP_CANDIDATES, "timestamp")
    parking_id_col = _require_column(test_df, _PARKING_ID_CANDIDATES, "parking_id")
    capacity_col = _require_column(test_df, _CAPACITY_CANDIDATES, "capacity")
    latitude_col = _find_column(test_df, _LATITUDE_CANDIDATES)
    longitude_col = _find_column(test_df, _LONGITUDE_CANDIDATES)
    if latitude_col is None or longitude_col is None:
        LOGGER.warning(
            "Latitude/longitude kolonları test verisinde yok; varsayılan 0.0 değerleri kullanılacak."
        )

    # Zaman damgasını parse et; geçersiz satırları sonradan eleyeceğiz
    timestamp = pd.to_datetime(test_df[timestamp_col], errors="coerce")
    if timestamp.isna().all():
        raise ValueError("Test verisindeki timestamp alanı tamamen geçersiz.")

    test_df = test_df.copy()
    test_df["__timestamp"] = timestamp
    # Modele kronolojik akış vermek için zamana göre sırala
    test_df = test_df.dropna(subset=["__timestamp"]).sort_values("__timestamp")

    if test_df.empty:
        raise ValueError("Tahmin üretimi için geçerli timestamp içeren kayıt yok.")

    # occupancy_rate hazırsa onu kullan, yoksa occupancy/capacity ile üret
    if _OCCUPANCY_RATE_CANDIDATES[0] in test_df.columns:
        occupancy_rate = pd.to_numeric(
            test_df[_OCCUPANCY_RATE_CANDIDATES[0]], errors="coerce"
        )
    elif _OCCUPANCY_RATE_CANDIDATES[1] in test_df.columns:
        occupancy_rate = pd.to_numeric(
            test_df[_OCCUPANCY_RATE_CANDIDATES[1]], errors="coerce"
        )
    else:
        occupancy_col = _require_column(test_df, _OCCUPANCY_CANDIDATES, "occupancy")
        capacity_numeric = pd.to_numeric(test_df[capacity_col], errors="coerce")
        occupancy_numeric = pd.to_numeric(test_df[occupancy_col], errors="coerce")
        occupancy_rate = occupancy_numeric / capacity_numeric.replace(0, np.nan)

    occupancy_rate = occupancy_rate.replace([np.inf, -np.inf], np.nan)
    valid_mask = occupancy_rate.notna()
    # Geçersiz oranları düşürüp model girdisini temizle
    test_df = test_df.loc[valid_mask].copy()
    occupancy_rate = occupancy_rate.loc[valid_mask].clip(0.0, 1.0)

    if len(test_df) < time_step:
        raise ValueError(
            f"LSTM tahmini için en az {time_step} satır gerekli, bulunan: {len(test_df)}"
        )

    min_rate = float(occupancy_rate.min())
    max_rate = float(occupancy_rate.max())
    # Basit MinMax ölçekleme (eğitim kodundaki yaklaşımı korur)
    if np.isclose(max_rate, min_rate):
        scaled = np.zeros(len(occupancy_rate), dtype=float)
    else:
        scaled = ((occupancy_rate - min_rate) / (max_rate - min_rate)).to_numpy(dtype=float)

    # time_step pencereleri ile LSTM giriş tensörü oluştur
    sequences = _create_sequences(scaled, time_step=time_step)
    if sequences.size == 0:
        raise ValueError("LSTM input sequence üretilemedi.")

    # Sequence başına bir adım ileri doluluk oranı tahmini üret
    try:
        pred_scaled = model.predict(sequences, verbose=0).reshape(-1)
    except Exception as exc:
        raise RuntimeError("LSTM model tahmini başarısız oldu.") from exc

    pred_scaled = np.clip(pred_scaled, 0.0, 1.0)
    # Ölçeği tekrar gerçek occupancy_rate aralığına geri çevir
    pred_rate = pred_scaled * (max_rate - min_rate) + min_rate

    # İlk tahmin, time_step tamamlandıktan sonraki satıra karşılık gelir
    pred_index = test_df.index[time_step - 1 :]
    base_df = test_df.loc[pred_index].copy()

    capacity_values = pd.to_numeric(base_df[capacity_col], errors="coerce")
    if capacity_values.isna().any():
        raise ValueError("Capacity alanında sayısal olmayan değerler var.")

    # Tahmin edilen oranı kapasite ile çarpıp fiziksel occupancy değerine çevir
    predicted_occupancy = np.clip(pred_rate * capacity_values.to_numpy(dtype=float), 0.0, None)
    predicted_occupancy = np.minimum(predicted_occupancy, capacity_values.to_numpy(dtype=float))

    latitude_values = (
        pd.to_numeric(base_df[latitude_col], errors="coerce").to_numpy()
        if latitude_col is not None
        else np.zeros(len(base_df), dtype=float)
    )
    longitude_values = (
        pd.to_numeric(base_df[longitude_col], errors="coerce").to_numpy()
        if longitude_col is not None
        else np.zeros(len(base_df), dtype=float)
    )

    predictions = pd.DataFrame(
        {
            "timestamp": base_df["__timestamp"].to_numpy(),
            "parking_id": base_df[parking_id_col].to_numpy(),
            "predicted_occupancy": predicted_occupancy,
            "capacity": capacity_values.to_numpy(dtype=float),
            "latitude": latitude_values,
            "longitude": longitude_values,
        }
    )

    if predictions[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Latitude/longitude alanlarında geçersiz değerler var.")

    try:
        predictions.to_csv(output_path, index=False)
    except Exception as exc:
        raise IOError(f"LSTM tahmin dosyası kaydedilemedi: {output_path}") from exc

    LOGGER.info("LSTM tahminleri üretildi ve kaydedildi: %s", output_path)
    return predictions


def load_predictions(input_path: Path) -> pd.DataFrame:
    """Tahmin verisini yükler ve gerekli kolon yapısını doğrular."""

    # CSV yoksa pipeline kırılmaz; otomatik LSTM fallback çalışır
    if not input_path.exists():
        LOGGER.warning(
            "Tahmin dosyası bulunamadı (%s), fallback başlatılıyor.", input_path
        )
        predictions = generate_lstm_predictions(output_path=input_path)
    else:
        try:
            predictions = pd.read_csv(input_path)
        except Exception as exc:
            raise ValueError(
                f"Tahmin dosyası okunamadı: {input_path}"
            ) from exc

    if predictions.empty:
        raise ValueError(f"Tahmin verisi boş: {input_path}")

    missing_cols = REQUIRED_COLUMNS - set(predictions.columns)

    if missing_cols:
        raise ValueError(
            f"Eksik gerekli kolonlar: {sorted(missing_cols)}. "
            f"Bulunan kolonlar: {sorted(predictions.columns)}"
        )

    # Sonraki feature adımlarında datetime operasyonları yapılacağı için
    # timestamp alanını burada kesin olarak datetime'a çeviriyoruz.
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], errors="coerce"
    )

    if predictions["timestamp"].isna().any():
        raise ValueError("Tahmin verisinde geçersiz timestamp değerleri bulundu.")

    return predictions


def _normalize_empty_slots_within_timestamp(df: pd.DataFrame) -> pd.Series:
    """Aynı timestamp içindeki boş slot sayılarını 0-1 arasında normalize eder."""

    # Her zaman diliminde otoparkları kendi aralarında kıyaslarız;
    # farklı timestamp'leri doğrudan birbiriyle karşılaştırmayız.
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
    # Basit bir proxy: doluluk arttıkça bekleme süresi lineer artsın.
    # (İleride gerçek saha verisi ile daha gerçekçi modele çevrilebilir.)
    df["expected_wait_time"] = df["occupancy_ratio"] * 10.0

    return df


def build_state_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Seçilen sayısal feature'lardan RL state vector oluşturur."""

    # Bu sıra önemlidir: RL ajanı state vektörünü sabit feature sırası ile okur.
    # Sıra değişirse eğitimli policy'nin beklediği temsil de değişir.
    vector_columns = [
        "occupancy_ratio",
        "predicted_empty_slots",
        "availability_score",
        "expected_wait_time",
        "time_sin",
        "time_cos",
    ]

    def _to_vector(row: pd.Series) -> List[float]:
        # CSV/NumPy dönüşümlerinde tip sürprizi olmaması için float'a zorla.
        return [float(row[col]) for col in vector_columns]

    df = df.copy()

    df["rl_state_vector"] = df[vector_columns].apply(
        _to_vector, axis=1
    )

    return df


def save_rl_states(df: pd.DataFrame, output_path: Path) -> None:
    """RL için hazır state tablosunu CSV olarak kaydeder."""

    # Klasör yoksa CI/CD veya temiz ortamda ilk çalıştırmada da hata vermesin.
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
        LOGGER.info("Loading RL feature pipeline...")
        # 1) LSTM tahminlerini yükle (gerekirse otomatik üret)
        predictions = load_predictions(input_path)

        # 2) RL için feature engineering adımı
        rl_features = create_rl_features(predictions)
        # 3) Gymnasium/DQN/PPO için state vector oluştur
        rl_states = build_state_vector(rl_features)

        # 4) Çıktıyı servislenebilir CSV formatında kaydet
        save_rl_states(rl_states, output_path)

        LOGGER.info("RL state hazırlama başarıyla tamamlandı.")

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