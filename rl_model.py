"""
PPO ile GridParkingEnv (mode=train) üzerinde politika eğitimi.

Grid dünyada ajan boş park (yeşil) hücrelere giderek sabit hedefe ulaşmayı öğrenir;
episode boyunca hedef ve doluluk haritası statiktir (stabil PPO).

Aksiyon maskesi: gözlemin son 4 boyutu geçerli yönleri kodlar; MaskedActorCriticPolicy
logitleri maskeler (geçersiz yön olasılığı 0). Alternatif: stable-baselines3-contrib
MaskablePPO + ActionMasker.

- train_env / val_env / test_env aynı sınıftan, farklı Gymnasium tohumlarıyla kurulur.
- EvalCallback doğrulama ortamında periyodik ölçüm yapar; en iyi ağırlıklar
  `models/best_model.zip` altında saklanır (Stable-Baselines3 varsayılanı).
- Final politika: models/ppo_parking_model_final.zip
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from paths import LOGS_DIR, MODELS_DIR, ensure_logs_dir, ensure_models
from parking_rl.grid_parking_env import GridParkingEnv, TRAIN_PPO_ENV_KWARGS
from parking_rl.masked_policy import MaskedActorCriticPolicy, masked_policy_kwargs


def _resolve_best_model_path() -> Path | None:
    """Önce best_model.zip, yoksa best_model aranır."""
    for name in ("best_model.zip", "best_model"):
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


class ActionHistogramCallback(BaseCallback):
    """Toplanan ayrık aksiyonları logs/actions.csv olarak yazar (rl_visualizer)."""

    def __init__(self, out_path: Path):
        super().__init__(0)
        self.out_path = out_path
        self._counts: Counter[int] = Counter()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).reshape(-1):
                self._counts[int(a)] += 1
        return True

    def _on_training_end(self) -> None:
        if not self._counts:
            return
        actions = sorted(self._counts.keys())
        df = pd.DataFrame({"action": actions, "count": [self._counts[k] for k in actions]})
        df.to_csv(self.out_path, index=False)


class RolloutActionLogCallback(BaseCallback):
    """Her rollout sonunda bu turda seçilen ayrık aksiyonların özetini stdout’a yazar."""

    def __init__(self) -> None:
        super().__init__(0)
        self._buf: list[int] = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._buf.extend(int(x) for x in np.asarray(actions).reshape(-1))
        return True

    def _on_rollout_end(self) -> None:
        if not self._buf:
            return
        c = Counter(self._buf)
        names = ("UP", "DOWN", "LEFT", "RIGHT")
        parts = [f"{names[k]}={c[k]}" for k in sorted(c.keys())]
        print(f"[PPO] Rollout aksiyon özeti ({len(self._buf)} adım): " + ", ".join(parts))
        self._buf.clear()


def main() -> None:
    ensure_models()
    ensure_logs_dir()

    print("[PPO] Grid ortamları kuruluyor (train / val / test, mode=train)...")
    # Episode ödül/uzunluk: logs/train.monitor.csv (rl_visualizer ile uyumlu)
    # debug_checks=False: her adımda grid/hedef doğrulaması yapmaz; eğitim CPU'da belirgin hızlanır.
    train_env = Monitor(GridParkingEnv(**TRAIN_PPO_ENV_KWARGS), str(LOGS_DIR / "train"))
    val_env = GridParkingEnv(**TRAIN_PPO_ENV_KWARGS)
    test_env = GridParkingEnv(**TRAIN_PPO_ENV_KWARGS)

    # eval_freq: en az bir rollout tamamlanabilsin (n_steps ile uyumlu)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(MODELS_DIR / "ppo_eval_logs"),
        eval_freq=8_192,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    action_log = ActionHistogramCallback(LOGS_DIR / "actions.csv")

    print("[PPO] Öğrenme başlıyor (100k timestep)...")
    # n_steps: her güncellemeden önce toplanan ortam adımı; düşük değer = daha sık güncelleme,
    # rollout turu daha kısa sürer (1024 tipik olarak 2048'e göre ~yarı süre, öğrenme hâlâ stabil).
    model = PPO(
        MaskedActorCriticPolicy,
        train_env,
        policy_kwargs=masked_policy_kwargs(),
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        ent_coef=0.01,
    )
    # Kayıp eğrileri: logs/progress.csv (train/policy_gradient_loss, train/value_loss)
    model.set_logger(configure(str(LOGS_DIR), ["stdout", "csv"]))
    model.learn(
        total_timesteps=100_000,
        callback=[eval_callback, action_log, RolloutActionLogCallback()],
    )

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
