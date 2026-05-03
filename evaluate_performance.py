"""
Eğitilmiş PPO politikasını GridParkingEnv (mode=train) üzerinde değerlendirir.

- Öncelik: models/best_model.zip (EvalCallback çıktısı)
- Yoksa: models/ppo_parking_model_final.zip

Metrikler (100 bölüm): hedefe varış oranı, ortalama adım, ortalama ödül.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from paths import MODELS_DIR
from parking_rl.grid_parking_env import GridParkingEnv, TRAIN_PPO_ENV_KWARGS


def load_eval_model():
    """En iyi model yolu yoksa final modele düş."""
    # SB3 ActorCriticPolicy zip yüklemesi için lr_schedule gerekir
    custom_objects = {"lr_schedule": lambda _: 3e-4}
    candidates = [
        MODELS_DIR / "best_model.zip",
        MODELS_DIR / "best_model",
        MODELS_DIR / "ppo_parking_model_final.zip",
    ]
    for p in candidates:
        if p.exists():
            return PPO.load(str(p), custom_objects=custom_objects), p
    raise FileNotFoundError(
        f"Hiçbir PPO ağırlığı bulunamadı. Önce rl_model.py çalıştırın. Aranan: {candidates}"
    )


def main() -> None:
    print("=" * 50)
    print(" PPO performans değerlendirmesi (GridParkingEnv train)")
    print("=" * 50)

    env = GridParkingEnv(**TRAIN_PPO_ENV_KWARGS)
    model, path = load_eval_model()
    print(f"Yüklenen model: {path}")
    print(f"Inference policy sınıfı: {type(model.policy).__name__}")

    n_episodes = 100
    successes = 0
    steps_list = []
    rewards_list = []

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=ep + 2024)
        terminated = False
        truncated = False
        steps = 0
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _step_info = env.step(int(action))
            steps += 1
            ep_reward += float(reward)

        if terminated:
            successes += 1
        steps_list.append(steps)
        rewards_list.append(ep_reward)

    success_rate = 100.0 * successes / n_episodes
    avg_steps = float(np.mean(steps_list))
    avg_reward = float(np.mean(rewards_list))

    print("-" * 50)
    print(f" Bölüm sayısı      : {n_episodes}")
    print(f" Başarı (hedef) %  : {success_rate:.1f}")
    print(f" Ortalama adım     : {avg_steps:.2f}")
    print(f" Ortalama ödül     : {avg_reward:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
