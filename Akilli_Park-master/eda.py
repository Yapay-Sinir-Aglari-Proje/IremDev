import pandas as pd
import matplotlib.pyplot as plt

# 1. Veriyi okuma
df = pd.read_csv("data/raw/parking.csv")

# 2. Temizlik yapıldı
df = df.drop_duplicates()
df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
df = df.sort_values("LastUpdated")

# 3. Doluluk oranı hesaplaması
df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]

# 4. Zaman grafiği
plt.figure(figsize=(10,4))
plt.plot(df["LastUpdated"], df["occupancy_rate"])
plt.title("Zamana Göre Doluluk")
plt.xlabel("Zaman")
plt.ylabel("Doluluk Oranı")
plt.show()

# 5. Saatlik analiz yapıldı
df["hour"] = df["LastUpdated"].dt.hour
hourly = df.groupby("hour")["occupancy_rate"].mean()

hourly.plot(kind="bar")
plt.title("Saatlik Doluluk")
plt.xlabel("Saat")
plt.ylabel("Ortalama Doluluk")
plt.show()

# 6. Günlük analiz yapıldı
df["day"] = df["LastUpdated"].dt.day_name()
daily = df.groupby("day")["occupancy_rate"].mean()

daily.plot(kind="bar")
plt.title("Günlük Doluluk")
plt.xlabel("Gün")
plt.ylabel("Ortalama Doluluk")
plt.show()

# 7. Histogram
df["occupancy_rate"].hist(bins=30)
plt.title("Doluluk Dağılımı")
plt.xlabel("Doluluk Oranı")
plt.ylabel("Frekans")
plt.show()