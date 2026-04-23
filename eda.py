"""
Keşifsel veri analizi (EDA): işlenmiş train/val/test birleşiminde grafikler.

Grafikler `output/` altına kaydedilir (ekranda göstermek için Agg backend;
sunucuda çalışırken pencere açılmaz).
"""

from __future__ import annotations

import matplotlib

# Türkçe karakterli başlıklar ve sunucu ortamı için etkileşimsiz backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_PROCESSED, OUTPUT_DIR, ensure_output


def load_processed_timeseries() -> pd.DataFrame:
    """
    Zaman sırasını koruyarak train → val → test dosyalarını üst üste birleştirir.
    (Shuffle yok; EDA da aynı kronolojiyi yansıtır.)
    """
    paths = [
        DATA_PROCESSED / "train.csv",
        DATA_PROCESSED / "val.csv",
        DATA_PROCESSED / "test.csv",
    ]
    chunks = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"İşlenmiş veri yok: {p}. Önce `python data_preparation.py` çalıştırın."
            )
        chunks.append(pd.read_csv(p))
    df = pd.concat(chunks, ignore_index=True)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
    df = df.dropna(subset=["LastUpdated"])
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df = df.dropna(subset=["Occupancy", "Capacity"])
    df = df[df["Capacity"] > 0]
    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]
    df = df.sort_values("LastUpdated", kind="mergesort")
    return df


def _save_fig(path, title: str) -> None:
    """Ortak kayıt: başlık + sıkı bbox."""
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(df: pd.DataFrame, out_dir) -> None:
    """Zaman serisi doluluk oranı (tüm otopark satırları kronolojik)."""
    plt.figure(figsize=(12, 4))
    plt.plot(df["LastUpdated"], df["occupancy_rate"], linewidth=0.8)
    plt.xlabel("Zaman")
    plt.ylabel("Doluluk oranı")
    _save_fig(out_dir / "1_zaman_serisi_doluluk.png", "Zaman serisi doluluk oranı")


def plot_hourly_average(df: pd.DataFrame, out_dir) -> None:
    """Saatlik ortalama doluluk."""
    df = df.copy()
    df["hour"] = df["LastUpdated"].dt.hour
    hourly = df.groupby("hour", sort=True)["occupancy_rate"].mean()
    plt.figure(figsize=(10, 4))
    hourly.plot(kind="bar", color="steelblue")
    plt.xlabel("Saat")
    plt.ylabel("Ortalama doluluk oranı")
    _save_fig(out_dir / "2_saatlik_ortalama.png", "Saatlik ortalama doluluk")


def plot_daily_average(df: pd.DataFrame, out_dir) -> None:
    """Haftanın gününe göre ortalama doluluk."""
    cats = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df = df.copy()
    df["day"] = pd.Categorical(
        df["LastUpdated"].dt.day_name(), categories=cats, ordered=True
    )
    daily = df.groupby("day", observed=False)["occupancy_rate"].mean()
    plt.figure(figsize=(10, 4))
    daily.plot(kind="bar", color="coral")
    plt.xlabel("Gün")
    plt.ylabel("Ortalama doluluk oranı")
    _save_fig(out_dir / "3_gunluk_ortalama.png", "Günlük ortalama doluluk")


def plot_histogram(df: pd.DataFrame, out_dir) -> None:
    """Doluluk oranı histogramı."""
    plt.figure(figsize=(8, 4))
    df["occupancy_rate"].hist(bins=40, color="seagreen", edgecolor="white")
    plt.xlabel("Doluluk oranı")
    plt.ylabel("Frekans")
    _save_fig(out_dir / "4_histogram_doluluk_orani.png", "Doluluk oranı dağılımı")


def main() -> None:
    ensure_output()
    out_dir = OUTPUT_DIR
    df = load_processed_timeseries()

    plot_time_series(df, out_dir)
    plot_hourly_average(df, out_dir)
    plot_daily_average(df, out_dir)
    plot_histogram(df, out_dir)

    print(f"[EDA] Grafikler kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
