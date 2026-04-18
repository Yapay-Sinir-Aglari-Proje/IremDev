# Akıllı Park (YSA entegrasyonu)

Akıllı otopark: LSTM doluluk tahmini, RL simülasyon ortamı (Gymnasium), veri işleme ve RL durum vektörleri (`models/prepare_prediction_states.py`).

Kaynak: [beyzayilmaz1/YSA](https://github.com/beyzayilmaz1/YSA) ile hizalıdır.

## Çalıştırma

1. Bağımlılıklar: `pip install -r requirements.txt`
2. Veri: `python data_preparation.py` → `data/processed/`
3. LSTM: `python lstm_model.py` (çıktı: `models/lstm_parking_model.h5`, `output/*.png`)
4. RL ortamı: `python -m parking_rl.smart_parking_env`
5. (İsteğe bağlı) RL state CSV: `python models/prepare_prediction_states.py` — `data/processed/lstm_predictions.csv` yoksa modelden tahmin üretir, `rl_prediction_states.csv` yazar.

## Bağımlılıklar

`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `gymnasium` (sürümler için `requirements.txt`).
