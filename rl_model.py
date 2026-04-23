import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from parking_rl.smart_parking_env import SmartParkingEnv

def main():
    print("PPO Modeli Eğitim Süreci Başlatılıyor...")
    
    # ---------------------------------------------------------
    # 1. ORTAMLARIN HAZIRLANMASI
    # ---------------------------------------------------------
    print("Veri seti ortamları (Train, Val, Test) yükleniyor...")
    train_env = SmartParkingEnv(data_path="data/processed/train.csv")
    val_env = SmartParkingEnv(data_path="data/processed/val.csv")
    test_env = SmartParkingEnv(data_path="data/processed/test.csv")

    # ---------------------------------------------------------
    # 2. DENETİM SİSTEMİ (EVAL CALLBACK)
    # ---------------------------------------------------------
    # Modelin performansını doğrulama seti üzerinde periyodik olarak test eden callback
    os.makedirs("models", exist_ok=True)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path='models/',
        log_path='models/', 
        eval_freq=5000, 
        deterministic=True, 
        render=False,
        verbose=1
    )

    # ---------------------------------------------------------
    # 3. MODELİN OLUŞTURULMASI VE EĞİTİM
    # ---------------------------------------------------------
    print("PPO algoritması yapılandırılıyor...")
    # ent_coef parametresi modelin yeni durumları keşfetmesini teşvik eder
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003, n_steps=2048, ent_coef=0.01)

    print("Eğitim süreci başlatıldı (100.000 adım)...")
    model.learn(total_timesteps=100000, callback=eval_callback)

    model.save("models/ppo_parking_model_final")
    print("Eğitim tamamlandı. Final modeli 'models/ppo_parking_model_final.zip' olarak kaydedildi.")

    # ---------------------------------------------------------
    # 4. PERFORMANS DEĞERLENDİRMESİ
    # ---------------------------------------------------------
    print("\n--- Model Performans Değerlendirmesi ---")
    print("En iyi performans gösteren model yükleniyor ve test ediliyor...")
    
    try:
        best_model = PPO.load("models/best_model")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    # Modeli üç farklı alt veri seti üzerinde test ediyoruz
    mean_train, std_train = evaluate_policy(best_model, train_env, n_eval_episodes=10)
    mean_val, std_val = evaluate_policy(best_model, val_env, n_eval_episodes=10)
    mean_test, std_test = evaluate_policy(best_model, test_env, n_eval_episodes=10)

    print(f"Train Seti Ortalama Ödül: {mean_train:.2f} +/- {std_train:.2f}")
    print(f"Val Seti Ortalama Ödül:   {mean_val:.2f} +/- {std_val:.2f}")
    print(f"Test Seti Ortalama Ödül:  {mean_test:.2f} +/- {std_test:.2f}")
    
    if mean_test > -1.0:
        print("\nDeğerlendirme: Model test verisi üzerinde başarılı bir şekilde genelleme yapabiliyor.")
    else:
        print("\nDeğerlendirme: Model test verisi üzerinde beklenen performansı gösteremedi.")

if __name__ == "__main__":
    main()