import numpy as np
from stable_baselines3 import PPO
from parking_rl.smart_parking_env import SmartParkingEnv

def main():
    print("\n" + "="*50)
    print(" 📊 GÖREV 7: PERFORMANS DEĞERLENDİRME VE ANALİZ ")
    print("="*50 + "\n")

    # 1. Ortam ve Modelin Yüklenmesi
    print("Test ortamı (test.csv) ve eğitilmiş model yükleniyor...")
    try:
        env = SmartParkingEnv(data_path="data/processed/test.csv")
        model = PPO.load("models/best_model")
    except Exception as e:
        print(f"Hata: Ortam veya model yüklenemedi. Detay: {e}")
        return

    # 2. Metrik Değişkenleri
    episodes = 100  # 100 farklı park arama senaryosu
    success_count = 0
    total_steps_list = []
    total_rewards_list = []

    print(f"{episodes} farklı senaryo üzerinde ajan test ediliyor. Lütfen bekleyin...\n")

    # 3. Simülasyon Döngüsü (100 Senaryo)
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        ep_reward = 0

        # Ajan otoparkı bulana kadar (veya süre bitene kadar) hamle yapar
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            ep_reward += reward

        # Başarı Kriteri: Eğer günün sonunda ajan artı puan aldıysa başarılı sayılır
        if ep_reward > 0:
            success_count += 1

        total_steps_list.append(steps)
        total_rewards_list.append(ep_reward)

    # 4. İstatistiklerin Hesaplanması
    success_rate = (success_count / episodes) * 100
    avg_steps = np.mean(total_steps_list)
    avg_reward = np.mean(total_rewards_list)

    # Dönüşümler (Gerçekçi bir analiz için adımları fiziksel birimlere çeviriyoruz)
    # Varsayım: Haritada atılan 1 adım = Ortalama 2 dakika zaman kaybı ve 0.5 km yol
    avg_time_minutes = avg_steps * 2.0
    avg_distance_km = avg_steps * 0.5
    
    # Maliyet: Ortalama yakıt tüketimi (Örn: km başına 3 TL yakıt maliyeti)
    avg_cost_tl = avg_distance_km * 3.0

    # 5. Profesyonel Çıktı Raporu
    print("-" * 50)
    print(" 📈 PERFORMANS ANALİZ RAPORU ")
    print("-" * 50)
    print(f"Test Edilen Senaryo Sayısı    : {episodes}")
    print(f"Başarı Oranı                  : %{success_rate:.1f}")
    print(f"Ortalama Park Bulma Süresi    : {avg_time_minutes:.1f} Dakika ({avg_steps:.1f} Adım)")
    print(f"Ortalama Kat Edilen Mesafe    : {avg_distance_km:.2f} km")
    print(f"Ortalama Yakıt Maliyeti       : {avg_cost_tl:.2f} TL")
    print(f"Ortalama Ödül Skoru           : {avg_reward:.2f}")
    print("-" * 50)

    if success_rate >= 75.0:
        print("\nSİSTEM ONAYI: Ajan oldukça verimli çalışıyor ve hedeflere ulaşıyor.")
    else:
        print("\nSİSTEM UYARISI: Başarı oranı hedeflenenin altında. Çevre parametreleri incelenmeli.")

if __name__ == "__main__":
    main()