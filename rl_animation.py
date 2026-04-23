"""
RL ajanının park etmeyi nasıl öğrendiğini animasyonla gösterir.

Öncelik: SmartParkingEnv + PPO modeli (varsa).
Yoksa: 10x10 demo grid + rastgele politika — klasör asla boş kalmaz.

Çıktı: animations/training_animation.mp4 (ffmpeg varsa) veya .gif (yoksa).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib import colors as mcolors

from paths import DATA_PROCESSED, MODELS_DIR, PROJECT_ROOT

# --- Görsel (kullanıcı spesifikasyonu) ---
_COLOR_AGENT = "#1f77b4"  # mavi
_COLOR_GOAL = "#ffdd57"  # sarı
_COLOR_OCCUPIED = "#d62728"  # kırmızı
_COLOR_EMPTY = "#2ca02c"  # yeşil — düşük doluluklu otopark (veya demo yürünebilir hücre)
_COLOR_GRID = "#d3d3d3"  # açık gri
_COLOR_ARROW = "#ff7f0e"  # turuncu
_COLOR_BACKGROUND = "#e8e8e8"  # SmartParking rasterında otoparkın düşmediği hücre

# Grid hücre kodları (imshow ListedColormap sırasıyla eşleşir)
CELL_BACKGROUND = 0  # harita kutusu; gerçek otopark projeksiyonu yok
CELL_EMPTY = 1  # otopark var ve doluluk < %85 (veya demo yolu)
CELL_OCCUPIED = 2  # doluluk >= %85 veya demo engel
CELL_GOAL = 3  # seçilen hedef otopark veya demo hedef

_ANIM_DIR = PROJECT_ROOT / "animations"
_DEFAULT_MP4 = _ANIM_DIR / "training_animation.mp4"
_DEFAULT_GIF = _ANIM_DIR / "training_animation.gif"

# Kayıtlı video/GIF kare hızı (düşük = daha yavaş dosya)
_DEFAULT_FILE_FPS = 3
# Ekrandaki plt.show önizlemesi: FuncAnimation interval (ms); yüksek = daha yavaş
_DEFAULT_PREVIEW_MS_PER_FRAME = 650

_ACTION_NAMES_DEMO = ("UP", "DOWN", "LEFT", "RIGHT")


def _ensure_anim_dir() -> Path:
    _ANIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[rl_animation] Çıktı klasörü hazır: {_ANIM_DIR.resolve()}")
    return _ANIM_DIR


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def create_demo_env(
    *,
    size: int = 10,
    seed: int = 0,
    obstacle_ratio: float = 0.22,
    max_steps: int = 200,
) -> "DemoGridEnv":
    """
    10x10 grid, ajan (0,0), hedef (9,9), rastgele dolu hücreler.
    Aksiyonlar: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (origin='lower' ile uyumlu).
    Ödül: hedef +10, her adım -0.1.
    """
    return DemoGridEnv(
        size=size,
        seed=seed,
        obstacle_ratio=obstacle_ratio,
        max_steps=max_steps,
    )


@dataclass
class DemoGridEnv:
    size: int = 10
    seed: int = 0
    obstacle_ratio: float = 0.22
    max_steps: int = 200
    rng: np.random.Generator = field(init=False, repr=False)
    grid: np.ndarray = field(init=False, repr=False)
    agent: Tuple[int, int] = field(init=False)
    goal: Tuple[int, int] = (9, 9)
    step_count: int = field(default=0, init=False)
    _grid_seed: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.action_space = spaces.Discrete(4)
        self.grid = np.full((self.size, self.size), CELL_EMPTY, dtype=int)
        self.agent = (0, 0)
        self._grid_seed = int(self.seed)
        self._build_grid()

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append((nr, nc))
        return out

    def _reachable(self, grid: np.ndarray) -> bool:
        gr, gc = self.goal
        start = (0, 0)
        if grid[0, 0] == CELL_OCCUPIED or grid[gr, gc] == CELL_OCCUPIED:
            return False
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) == (gr, gc):
                return True
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) in seen:
                    continue
                if int(grid[nr, nc]) == CELL_OCCUPIED:
                    continue
                seen.add((nr, nc))
                stack.append((nr, nc))
        return False

    def _build_grid(self) -> None:
        gr, gc = self.goal
        n_cells = self.size * self.size
        n_obs = int(round(self.obstacle_ratio * n_cells))
        n_obs = max(0, min(n_obs, n_cells - 3))

        for attempt in range(400):
            g = np.full((self.size, self.size), CELL_EMPTY, dtype=int)
            inner = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in ((0, 0), (gr, gc))]
            self.rng.shuffle(inner)
            for r, c in inner[:n_obs]:
                g[r, c] = CELL_OCCUPIED
            g[gr, gc] = CELL_GOAL
            if self._reachable(g):
                self.grid = g
                print(f"[rl_animation] Demo grid oluşturuldu (deneme {attempt + 1}), engel sayısı={n_obs}.")
                return
        # Son çare: engelsiz
        g = np.full((self.size, self.size), CELL_EMPTY, dtype=int)
        g[gr, gc] = CELL_GOAL
        self.grid = g
        print("[rl_animation] Uyarı: rastgele engellerde yol bulunamadı; engelsiz demo grid kullanılıyor.")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None and int(seed) != int(self._grid_seed):
            self.rng = np.random.default_rng(seed)
            self.seed = int(seed)
            self._grid_seed = int(seed)
            self._build_grid()
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent = (0, 0)
        self.step_count = 0
        obs = self._obs()
        return obs, {"mode": "demo"}

    def _obs(self) -> np.ndarray:
        return np.array([self.agent[0], self.agent[1]], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        dr_dc = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
        dr, dc = dr_dc[int(action) % 4]
        r, c = self.agent
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if int(self.grid[nr, nc]) != CELL_OCCUPIED:
                r, c = nr, nc
        self.agent = (r, c)
        self.step_count += 1

        at_goal = self.agent == self.goal
        reward = 10.0 if at_goal else -0.1
        terminated = bool(at_goal)
        truncated = self.step_count >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {"mode": "demo"}

    def to_draw_state(
        self,
        *,
        step: int,
        action: Any,
        reward: float,
        prev_agent: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """draw_frame için durum sözlüğü."""
        display = self.grid.copy()
        gr, gc = self.goal
        if display[gr, gc] != CELL_GOAL:
            display[gr, gc] = CELL_GOAL

        ar, ac = self.agent
        dx, dy = 0.0, 0.0
        if prev_agent is not None:
            pr, pc = prev_agent
            dx = (ac + 0.5) - (pc + 0.5)
            dy = (ar + 0.5) - (pr + 0.5)
        return {
            "grid": display,
            "agent": (ar, ac),
            "step": int(step),
            "action": action,
            "reward": float(reward),
            "arrow": (dx, dy) if prev_agent is not None else None,
        }


def draw_frame(ax: plt.Axes, state: Dict[str, Any]) -> None:
    """Tek kare: grid + ajan + ok; başlıkta Step, Action, Reward."""
    ax.clear()
    grid = np.asarray(state["grid"], dtype=int)
    h, w = grid.shape
    ag_r, ag_c = state["agent"]
    ag_r, ag_c = float(ag_r), float(ag_c)

    cmap = mcolors.ListedColormap(
        [_COLOR_BACKGROUND, _COLOR_EMPTY, _COLOR_OCCUPIED, _COLOR_GOAL]
    )
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(0, w, 0, h),
    )

    for g in np.arange(0, w + 1, 1):
        ax.axvline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)
    for g in np.arange(0, h + 1, 1):
        ax.axhline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)

    ax.scatter(
        [ag_c + 0.5],
        [ag_r + 0.5],
        s=420,
        color=_COLOR_AGENT,
        edgecolors="white",
        linewidths=1.4,
        zorder=5,
    )

    arrow = state.get("arrow")
    if arrow is not None:
        dx, dy = float(arrow[0]), float(arrow[1])
        mag = math.hypot(dx, dy)
        if mag > 1e-9:
            dx /= mag
            dy /= mag
            ax.quiver(
                ag_c + 0.5,
                ag_r + 0.5,
                dx * 0.42,
                dy * 0.42,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=_COLOR_ARROW,
                width=0.014,
                zorder=6,
            )

    step = int(state.get("step", 0))
    action = state.get("action", "-")
    reward = float(state.get("reward", 0.0))
    ax.set_title(
        f"Step: {step}   |   Action: {action}   |   Reward: {reward:.3f}",
        fontsize=12,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _latlon_to_cell(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    rows: int,
    cols: int,
) -> Tuple[int, int]:
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_span = max(lon_max - lon_min, 1e-9)
    c = int(np.clip((lon - lon_min) / lon_span * cols, 0, cols - 1))
    r = int(np.clip((lat - lat_min) / lat_span * rows, 0, rows - 1))
    return r, c


def _smart_parking_to_state(
    env: Any,
    *,
    step: int,
    action: Any,
    reward: float,
    info: Dict[str, Any],
    grid_shape: Tuple[int, int] = (18, 18),
) -> Dict[str, Any]:
    """SmartParkingEnv anlık görüntüsünü draw_frame uyumlu state'e çevirir."""
    rows, cols = grid_shape
    lat_min, lat_max = float(env.lat_min), float(env.lat_max)
    lon_min, lon_max = float(env.lon_min), float(env.lon_max)

    # Tüm kutu başta arka plan: otopark koordinatı bu hücreye düşmüyorsa yeşil değil gri kalır
    grid = np.full((rows, cols), CELL_BACKGROUND, dtype=int)

    target_idx = int(info.get("target_index", getattr(env, "_target_idx", 0)))
    t_idx = int(info.get("time_index", getattr(env, "_time_idx", 0)))
    snap = env._merge_snapshot(t_idx)  # noqa: SLF001

    for i, lot in enumerate(env.lots):
        r, c = _latlon_to_cell(
            lot.latitude, lot.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
        )
        occ, cap = snap.get(lot.parking_id, (0.0, 1.0))
        cap = max(1.0, float(cap))
        occ_ratio = float(np.clip(float(occ) / cap, 0.0, 1.0))
        if i == target_idx:
            grid[r, c] = CELL_GOAL
        elif occ_ratio >= 0.85:
            grid[r, c] = CELL_OCCUPIED
        else:
            grid[r, c] = CELL_EMPTY

    v = env.vehicle
    ag_r, ag_c = _latlon_to_cell(
        v.latitude, v.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
    )
    tlot = env.lots[target_idx]
    tr, tc = _latlon_to_cell(
        tlot.latitude, tlot.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
    )
    dx = (tc + 0.5) - (ag_c + 0.5)
    dy = (tr + 0.5) - (ag_r + 0.5)

    return {
        "grid": grid,
        "agent": (ag_r, ag_c),
        "step": int(step),
        "action": action,
        "reward": float(reward),
        "arrow": (dx, dy),
    }


def _policy_action(policy: Any, obs: Any, env: Any) -> int:
    if hasattr(policy, "predict"):
        act, _ = policy.predict(obs, deterministic=True)
        return int(np.asarray(act).reshape(()))
    out = policy(obs)
    return int(np.asarray(out).reshape(()))


def run_episode(
    env: Any,
    policy: Optional[Any] = None,
    *,
    seed: int = 42,
    max_steps: int = 200,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Bir episode boyunca her adım için draw_frame uyumlu state listesi döndürür.
    mode: 'demo' | 'smart' | None (None ise ortam tipinden otomatik seçilir).
    """
    if mode is None:
        mode = "demo" if isinstance(env, DemoGridEnv) else "smart"

    trajectory: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=seed)
    prev_agent: Optional[Tuple[int, int]] = None

    if mode == "demo":
        assert isinstance(env, DemoGridEnv)
        trajectory.append(
            env.to_draw_state(step=0, action="-", reward=0.0, prev_agent=None)
        )
        prev_agent = tuple(env.agent)
    else:
        init_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        init_info.setdefault("target_index", int(getattr(env, "_target_idx", 0)))
        init_info.setdefault("time_index", int(getattr(env, "_time_idx", 0)))
        trajectory.append(
            _smart_parking_to_state(env, step=0, action="-", reward=0.0, info=init_info)
        )

    for t in range(max_steps):
        if policy is None:
            if hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
                action = int(env.action_space.sample())
            else:
                action = 0
        else:
            action = _policy_action(policy, obs, env)

        if mode == "demo":
            prev_agent = tuple(env.agent)

        obs, reward, terminated, truncated, step_info = env.step(action)
        merged = {**(info if isinstance(info, dict) else {}), **step_info}

        if mode == "demo":
            assert isinstance(env, DemoGridEnv)
            aname = _ACTION_NAMES_DEMO[int(action) % 4]
            trajectory.append(
                env.to_draw_state(
                    step=t + 1,
                    action=aname,
                    reward=float(reward),
                    prev_agent=prev_agent,
                )
            )
        else:
            trajectory.append(
                _smart_parking_to_state(
                    env,
                    step=t + 1,
                    action=int(action),
                    reward=float(reward),
                    info=merged,
                )
            )

        if terminated or truncated:
            break
        info = merged

    if len(trajectory) == 0:
        print("[rl_animation] Uyarı: trajectory boş; tek karelik yedek state eklendi.")
        if isinstance(env, DemoGridEnv):
            trajectory.append(
                env.to_draw_state(step=0, action="N/A", reward=0.0, prev_agent=None)
            )

    print(f"[rl_animation] Episode toplam kare: {len(trajectory)}")
    return trajectory


def _try_load_smart_stack() -> Tuple[Optional[Any], Optional[Any], str]:
    """(env, policy, mode) — başarısızsa (None, None, '')."""
    train_csv = DATA_PROCESSED / "train.csv"
    if not train_csv.is_file():
        print(f"[rl_animation] SmartParkingEnv atlandı — dosya yok: {train_csv}")
        return None, None, ""

    try:
        from parking_rl.smart_parking_env import SmartParkingEnv
    except Exception as exc:
        print(f"[rl_animation] SmartParkingEnv import hatası: {exc}")
        return None, None, ""

    try:
        env = SmartParkingEnv(
            data_path=train_csv,
            max_episode_steps=120,
            randomize_start_time=False,
        )
        print(f"[rl_animation] SmartParkingEnv yüklendi: {train_csv}")
    except Exception as exc:
        print(f"[rl_animation] SmartParkingEnv kurulamadı: {exc}")
        return None, None, ""

    policy = None
    for name in (
        "best_model.zip",
        "best_model",
        "ppo_parking_model_final.zip",
        "ppo_parking_model_final",
    ):
        p = MODELS_DIR / name
        if p.exists():
            try:
                from stable_baselines3 import PPO

                policy = PPO.load(str(p))
                print(f"[rl_animation] PPO modeli yüklendi: {p}")
            except Exception as exc:
                print(f"[rl_animation] PPO yüklenemedi ({p}): {exc}")
                policy = None
            break

    if policy is None:
        print("[rl_animation] PPO bulunamadı veya yüklenemedi; SmartParking için rastgele politika.")

    return env, policy, "smart"


def _save_animation(
    fig: plt.Figure,
    trajectory: List[Dict[str, Any]],
    *,
    fps: int = _DEFAULT_FILE_FPS,
) -> Path:
    _ensure_anim_dir()

    def _update(frame_idx: int) -> None:
        if frame_idx < len(trajectory):
            draw_frame(fig.axes[0], trajectory[frame_idx])

    n_frames = max(1, len(trajectory))
    anim = mplanim.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=max(50, int(1000 / max(1, fps))),
        blit=False,
    )

    used_ffmpeg = _ffmpeg_available()
    if used_ffmpeg:
        out_path = _DEFAULT_MP4
        print(f"[rl_animation] ffmpeg bulundu; MP4 kaydediliyor: {out_path}")
        try:
            writer = mplanim.FFMpegWriter(fps=fps, metadata={"title": "Akıllı Park RL"})
            anim.save(str(out_path), writer=writer, dpi=120)
        except Exception as exc:
            print(f"[rl_animation] MP4 kaydı başarısız: {exc}")
            used_ffmpeg = False

    if not used_ffmpeg:
        out_path = _DEFAULT_GIF
        print(
            "[rl_animation] ffmpeg bulunamadı veya MP4 yazılamadı; "
            "Pillow ile GIF kaydediliyor."
        )
        try:
            writer = mplanim.PillowWriter(fps=fps)
            anim.save(str(out_path), writer=writer, dpi=120)
        except Exception as exc:
            print(f"[rl_animation] GIF kaydı başarısız: {exc}")
            # Son çare: tek kare PNG
            out_path = _ANIM_DIR / "training_animation_fallback.png"
            draw_frame(fig.axes[0], trajectory[0])
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            print(f"[rl_animation] Yedek statik görüntü yazıldı: {out_path}")

    plt.close(fig)
    if out_path.exists():
        print(f"[rl_animation] Dosya boyutu: {out_path.stat().st_size} byte — {out_path}")
    return out_path


def generate_animation(
    *,
    episode_seed: int = 42,
    fps: int = _DEFAULT_FILE_FPS,
    preview_ms_per_frame: int = _DEFAULT_PREVIEW_MS_PER_FRAME,
) -> Path:
    """
    Gerçek ortam + model (mümkünse) veya demo grid ile animasyon üretir;
    MP4 veya GIF yazar ve plt.show() ile gösterir.

    Args:
        episode_seed: Episode ve (demo için) grid tekrarlanabilirliği.
        fps: Kaydedilen MP4/GIF saniyedeki kare sayısı (2–4 genelde rahat izlenir).
        preview_ms_per_frame: Ekranda plt.show ile oynatırken kareler arası süre (ms);
            büyütmek animasyonu yavaşlatır (ör. 800–1200).
    """
    env: Any
    policy: Optional[Any]
    mode: str

    env, policy, mode = _try_load_smart_stack()
    if env is None:
        print("[rl_animation] Demo ortamına geçiliyor (create_demo_env).")
        env = create_demo_env(seed=episode_seed)
        policy = None
        mode = "demo"
    else:
        mode = "smart"

    print(f"[rl_animation] Mod: {mode}, seed={episode_seed}")

    trajectory = run_episode(env, policy, seed=episode_seed, max_steps=220)

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    draw_frame(ax, trajectory[0])

    out_path = _save_animation(fig, trajectory, fps=fps)

    preview_interval = max(120, int(preview_ms_per_frame))
    print(
        f"[rl_animation] Önizleme hızı: ~{preview_interval} ms/kare "
        f"(daha yavaş için generate_animation(preview_ms_per_frame=900) gibi artırın)."
    )

    # Gösterim (kaydedilen fig kapatıldı; önizleme için yeniden çiz)
    preview_fig, preview_ax = plt.subplots(figsize=(8.0, 8.0))

    def _preview_update(i: int) -> None:
        if i < len(trajectory):
            draw_frame(preview_ax, trajectory[i])

    preview_anim = mplanim.FuncAnimation(
        preview_fig,
        _preview_update,
        frames=max(1, len(trajectory)),
        interval=preview_interval,
        blit=False,
    )
    preview_fig._rl_animation_ref = preview_anim  # GC ile animasyonun silinmesini önle
    # show() animasyonu ekranda tutar
    print("[rl_animation] Pencere açılıyor (plt.show) — kapatınca script sonlanır.")
    plt.show()
    plt.close(preview_fig)

    return out_path


if __name__ == "__main__":
    print("Animasyon başlatılıyor...")
    try:
        generate_animation()
    except Exception as exc:
        print(f"[rl_animation] KRİTİK HATA: {exc}")
        _ensure_anim_dir()
        # Mutlaka bir çıktı
        emergency = _ANIM_DIR / "training_animation_error_note.txt"
        emergency.write_text(
            f"Animasyon üretilemedi.\nSebep: {exc!r}\n",
            encoding="utf-8",
        )
        print(f"[rl_animation] Hata notu yazıldı: {emergency}")
        raise
    print("Animasyon tamamlandı!")
