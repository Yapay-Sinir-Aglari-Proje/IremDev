import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 


# VERI YUKLEME
# train,validation ve test setlerini dahil ediyoruz.

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")


# DOLULUK ORANI HESAPLAMA
# occupancy ve capacity kolonları üzerinden doluluk oranını hesaplıyoruz.
# Bu, modelimizin tahmin etmeye çalışacağı hedef değişken olacak
# .
for df in [train_df, val_df, test_df]:
    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]


# GEREKLI KOLONLARI SEÇME
# tarih,saat vb degerler arindirilip sadece occupancy_rate kolonunu kullanarak model egitilecek.

train = train_df[["occupancy_rate"]].values
val   = val_df[["occupancy_rate"]].values
test  = test_df[["occupancy_rate"]].values


# NORMALIZASYON
# MinMaxScaler kullanarak veriler 0-1 arasina cekiliyor.Egitimin daha hizli ve kararli olmasi icin.

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)


# SEQUENCE 
# kisaca son 1 satte ne olduysa ona bakip tahmin ureticek.

def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

time_step = 12 
X_train, y_train = create_dataset(train_scaled, time_step)
X_val, y_val     = create_dataset(val_scaled, time_step)
X_test, y_test   = create_dataset(test_scaled, time_step)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# MIMARI OLUŞTURMA

model = Sequential()
# İlk katmana Dropout ekleyerek overfitting riskini azalttık
model.add(LSTM(64, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2)) 
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()


# MODEL EGITIMI

# EKLEME: Model iyileşmeyi durdurursa eğitimi kes
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, # Early stopping olduğu için epoch sayısını artırabiliriz
    batch_size=32,
    callbacks=[early_stop], # Callback eklendi
    verbose=1
)


# MODEL KAYDETME
# basirli model lstm_parking_model.h5 olarak kaydedilecek.

# RL aşamasında bu dosyayı 'load_model' ile çağıracaksın
model.save("models/lstm_parking_model.h5")
print("\n[INFO] Model 'models/lstm_parking_model.h5' olarak kaydedildi.")



# LOSS GRAFİĞİ
# eger train loss ve val loss birlikte asagi iniyorsa model iyi egitiliyor demektir. 
# Eger val loss artarken train loss azaliyor ise overfitting olabilir.

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Eğitim ve Doğrulama Kaybı (Loss)")
plt.legend()
plt.show()


# MODEL DEGERLENDIRME
# modelin basrisi burada olculuyor. 
# RMSE ve MAE degerleri hesaplanarak modelin tahmin performansi degerlendiriliyor. 

test_pred = model.predict(X_test)

# Ölçeği geri al
test_pred_rescaled = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test)

rmse = mean_squared_error(y_test_real, test_pred_rescaled, squared=False)
mae = mean_absolute_error(y_test_real, test_pred_rescaled)

print("\n===== MODEL PERFORMANSI =====")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# TAHMIN GRAFİĞİ
# gerçek ve tahmin edilen doluluk oranlarını görselleştirerek modelin performansını daha iyi anlayabiliriz.
plt.figure(figsize=(15,5))
plt.plot(y_test_real[:200], label="Gerçek Değerler", color='blue') # İlk 200 adımı görmek daha net sonuç verir
plt.plot(test_pred_rescaled[:200], label="LSTM Tahmini", color='red', linestyle='--')
plt.legend()
plt.title("Otopark Doluluk Oranı Tahmini (Test Seti - Kesit)")
plt.xlabel("Zaman Adımı")
plt.ylabel("Doluluk Oranı")
plt.show()



# 12. RL İÇİN GELECEK DURUM (STATE) HAZIRLIĞI

last_seq = test_scaled[-time_step:].reshape(1, time_step, 1)
future_pred = model.predict(last_seq)
future_pred_real = scaler.inverse_transform(future_pred)

print("\n===== RL İÇİN HAZIRLIK =====")
print(f"Bir sonraki tahmini doluluk: {future_pred_real[0][0]:.4f}")