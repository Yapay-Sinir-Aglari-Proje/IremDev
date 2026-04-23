"""
PPO ile SmartParkingEnv üzerinde politika eğitimi.

- train_env / val_env / test_env ayrı CSV’lerden kurulur.
- EvalCallback doğrulama ortamında periyodik ölçüm yapar; en iyi ağırlıklar
  `models/best_model.zip` altında saklanır (Stable-Baselines3 varsayılanı).
- Final politika: models/ppo_parking_model_final.zip
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from paths import DATA_PROCESSED, MODELS_DIR, ensure_models
from parking_rl.smart_parking_env import SmartParkingEnv


def _resolve_best_model_path() -> Path | None:
    """Önce best_model.zip, yoksa best_model aranır."""
    for name in ("best_model.zip", "best_model"):
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


def main() -> None:
    ensure_models()

    train_path = DATA_PROCESSED / "train.csv"
    val_path = DATA_PROCESSED / "val.csv"
    test_path = DATA_PROCESSED / "test.csv"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"İşlenmiş veri eksik: {p}. Önce data_preparation çalıştırın.")

    print("[PPO] Ortamlar yükleniyor (train / val / test)...")
    train_env = SmartParkingEnv(data_path=train_path)
    val_env = SmartParkingEnv(data_path=val_path)
    test_env = SmartParkingEnv(data_path=test_path)

    # eval_freq: en az bir rollout tamamlanabilsin (n_steps ile uyumlu)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(MODELS_DIR / "ppo_eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print("[PPO] Öğrenme başlıyor (100k timestep)...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=100_000, callback=eval_callback)

    final_path = MODELS_DIR / "ppo_parking_model_final.zip"
    # Stable-Baselines3 kayıtta dosya adına otomatik .zip ekler
    model.save(str(MODELS_DIR / "ppo_parking_model_final"))
    print(f"[PPO] Final model kaydı: {final_path}")

    print("\n[PPO] En iyi model ile evaluate_policy...")
    best = _resolve_best_model_path()
    if best is None:
        print("[PPO] Uyarı: best_model bulunamadı, final model kullanılıyor.")
        eval_model = PPO.load(str(final_path))
    else:
        print(f"[PPO] Yüklenen en iyi model: {best}")
        eval_model = PPO.load(str(best))

    for name, env in ("train", train_env), ("val", val_env), ("test", test_env):
        mean_r, std_r = evaluate_policy(
            eval_model, env, n_eval_episodes=10, deterministic=True
        )
        print(f"  {name}: ortalama ödül = {mean_r:.3f} ± {std_r:.3f}")


if __name__ == "__main__":
    main()
