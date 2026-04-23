"""
Eğitilmiş PPO politikasını test ortamında değerlendirir.

- Öncelik: models/best_model.zip (EvalCallback çıktısı)
- Yoksa: models/ppo_parking_model_final.zip

Metrikler (100 bölüm): başarı oranı (hedefe varış), ortalama adım, ortalama ödül.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from paths import DATA_PROCESSED, MODELS_DIR
from parking_rl.smart_parking_env import SmartParkingEnv


def load_eval_model():
    """En iyi model yolu yoksa final modele düş."""
    candidates = [
        MODELS_DIR / "best_model.zip",
        MODELS_DIR / "best_model",
        MODELS_DIR / "ppo_parking_model_final.zip",
    ]
    for p in candidates:
        if p.exists():
            return PPO.load(str(p)), p
    raise FileNotFoundError(
        f"Hiçbir PPO ağırlığı bulunamadı. Önce rl_model.py çalıştırın. Aranan: {candidates}"
    )


def main() -> None:
    print("=" * 50)
    print(" PPO performans değerlendirmesi (test.csv)")
    print("=" * 50)

    test_csv = DATA_PROCESSED / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test verisi yok: {test_csv}")

    env = SmartParkingEnv(data_path=test_csv, randomize_start_time=True)
    model, path = load_eval_model()
    print(f"Yüklenen model: {path}")

    n_episodes = 100
    successes = 0
    steps_list = []
    rewards_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 2024)
        terminated = False
        truncated = False
        steps = 0
        ep_reward = 0.0
        arrived_flag = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            ep_reward += float(reward)
            if bool(info.get("arrived", False)):
                arrived_flag = True

        if arrived_flag:
            successes += 1
        steps_list.append(steps)
        rewards_list.append(ep_reward)

    success_rate = 100.0 * successes / n_episodes
    avg_steps = float(np.mean(steps_list))
    avg_reward = float(np.mean(rewards_list))

    print("-" * 50)
    print(f" Bölüm sayısı      : {n_episodes}")
    print(f" Başarı (varış) %  : {success_rate:.1f}")
    print(f" Ortalama adım     : {avg_steps:.2f}")
    print(f" Ortalama ödül     : {avg_reward:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
