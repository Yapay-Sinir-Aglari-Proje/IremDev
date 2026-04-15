import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    CSV dosyasını yükler ve veri boyutunu ekrana yazdırır.
    """
    # CSV dosyasını oku
    df = pd.read_csv(file_path)

    # İlk veri boyutunu göster
    print(f"Initial shape: {df.shape}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veriyi temizler:
    - Tekrarlanan satırları siler
    - Negatif doluluk değerlerini kaldırır
    - Capacity'den büyük Occupancy değerlerini kaldırır
    """
    # Temizlik öncesi satır sayısı
    initial_rows = len(df)

    # Duplicate kayıtları kaldır
    df = df.drop_duplicates()

    # Geçersiz occupancy değerlerini temizle
    df = df[df["Occupancy"] >= 0]
    df = df[df["Occupancy"] <= df["Capacity"]]

    # Kaç satır silindiğini hesapla
    removed_rows = initial_rows - len(df)
    print(f"Rows removed during cleaning: {removed_rows}")

    return df


def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    LastUpdated kolonunu datetime formatına çevirir,
    hatalı tarihleri siler ve veriyi zamana göre sıralar.
    """
    # Tarih kolonunu datetime formatına çevir
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")

    # Hatalı dönüşen tarihleri sil
    df = df.dropna(subset=["LastUpdated"])

    # Zamana göre küçükten büyüğe sırala
    df = df.sort_values(by="LastUpdated", ascending=True).reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Veriyi zaman sırasını bozmadan
    %70 train, %15 validation, %15 test olarak ayırır.
    """
    # Toplam veri sayısı
    n = len(df)

    # Bölünme indekslerini hesapla
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    # Shuffle yapmadan zaman serisi mantığında böl
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def main() -> None:
    """
    Veri hazırlama pipeline'ını çalıştırır:
    yükleme -> temizleme -> datetime işleme -> split -> kaydetme
    """
    # Dosya yolları
    input_path = "data/raw/parking.csv"
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    # 1) Veriyi yükle
    df = load_data(input_path)

    # 2) Temizlik işlemleri
    df = clean_data(df)

    # Eksik veri kontrolü
    print("Eksik veri durumu:")
    print(df.isnull().sum())

    # İlk birkaç satırı kontrol amaçlı göster
    print("\nİlk 5 satır:")
    print(df.head())

    # 3) Tarih kolonunu işle
    df = process_datetime(df)

    # 4) Train / validation / test ayır
    train_df, val_df, test_df = split_data(df)

    # Boyutları kontrol et
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 5) İşlenmiş verileri kaydet
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


# Script doğrudan çalıştırıldığında main fonksiyonunu başlat
if __name__ == "__main__":
    main()