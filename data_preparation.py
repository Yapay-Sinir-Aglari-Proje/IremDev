import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """CSV veri setini yükler ve başlangıç boyutunu yazdırır."""
    df = pd.read_csv(file_path)

    print(f"Initial shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Veri setini temizler: tekrar eden kayıtları ve hatalı doluluk değerlerini siler."""
    initial_rows = len(df)

    df = df.drop_duplicates()

    df = df[df["Occupancy"] >= 0]
    df = df[df["Occupancy"] <= df["Capacity"]]

    removed_rows = initial_rows - len(df)

    print(f"Rows removed during cleaning: {removed_rows}")
    return df


def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """LastUpdated sütununu datetime formatına çevirir ve artan şekilde sıralar."""
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")

    df = df.dropna(subset=["LastUpdated"])
    df = df.sort_values(by="LastUpdated", ascending=True).reset_index(drop=True)

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Veriyi karıştırmadan train/validation/test setlerine böler."""
    n = len(df)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def main() -> None:
    input_path = "data/raw/parking.csv"
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    df = load_data(input_path)

    df = clean_data(df)

    print("Eksik veri durumu:")
    print(df.isnull().sum())

    print("\nİlk 5 satır:")
    print(df.head())

    df = process_datetime(df)

    train_df, val_df, test_df = split_data(df)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    main()