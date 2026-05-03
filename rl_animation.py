"""
RL ajanının park etmeyi nasıl öğrendiğini animasyonla gösterir.

Öncelik: GridParkingEnv (train/demo) + eğitilmiş PPO (models/).
İsteğe bağlı: --legacy-smart ile SmartParkingEnv + PPO (grid modeliyle gözlem uyumu gerekir).

Çıktı: animations/training_animation.mp4 (ffmpeg varsa) veya .gif (yoksa).
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from paths import DATA_PROCESSED, MODELS_DIR, PROJECT_ROOT
from parking_rl.grid_parking_env import (
    CELL_BUILDING,
    CELL_EMPTY,
    CELL_GOAL,
    CELL_OCCUPIED,
    INVALID_ACTION_PENALTY,
    GridParkingEnv,
    TRAIN_PPO_ENV_KWARGS,
)

# --- Görsel (kullanıcı spesifikasyonu) ---
_COLOR_AGENT = "#1f77b4"  # mavi
_COLOR_GOAL = "#ffdd57"  # sarı
_COLOR_GOAL_FLASH = "#fff4a3"  # hedef yanıp sönme (bonus)
_COLOR_OCCUPIED = "#d62728"  # kırmızı
_COLOR_EMPTY = "#2ca02c"  # yeşil — boş park
_COLOR_BUILDING = "#7f7f7f"  # bina / engel (gri)
_COLOR_GRID = "#d3d3d3"  # ızgara çizgisi
_COLOR_ARROW = "#ff7f0e"  # turuncu
_COLOR_BACKGROUND = "#e8e8e8"  # SmartParking rasterında otoparkın düşmediği hücre

# SmartParking raster: arka plan kutusu (grid ile aynı kod 0)
CELL_BACKGROUND = CELL_BUILDING

_ANIM_DIR = PROJECT_ROOT / "animations"
_DEFAULT_MP4 = _ANIM_DIR / "training_animation.mp4"
_DEFAULT_GIF = _ANIM_DIR / "training_animation.gif"

# Kayıtlı video/GIF kare hızı (düşük = daha yavaş dosya)
_DEFAULT_FILE_FPS = 3
# Ekrandaki plt.show önizlemesi: FuncAnimation interval (ms); yüksek = daha yavaş
_DEFAULT_PREVIEW_MS_PER_FRAME = 650

_ACTION_NAMES_DEMO = ("UP", "DOWN", "LEFT", "RIGHT")


def _legal_actions_label(legal: Any) -> str:
    if not isinstance(legal, list) or not legal:
        return ""
    parts = [_ACTION_NAMES_DEMO[int(i) % 4] for i in legal]
    return "Geçerli: " + ", ".join(parts)


def _ensure_anim_dir() -> Path:
    _ANIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[rl_animation] Çıktı klasörü hazır: {_ANIM_DIR.resolve()}")
    return _ANIM_DIR


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def create_grid_env(
    *,
    size: int = 10,
    mode: str = "demo",
    max_steps: int = 200,
    building_ratio: float = 0.18,
    occupied_ratio: float = 0.12,
    debug_checks: bool = False,
    step_debug_log: bool = False,
    match_train_mdp: bool = False,
) -> GridParkingEnv:
    """GridParkingEnv: train (statik hedef / grid) veya demo (dinamik doluluk).

    match_train_mdp: True ise rl_model ile aynı ödül / MDP parametreleri (TRAIN_PPO_ENV_KWARGS)
    üzerine boyut, mod ve max_steps gibi alanlar uygulanır.
    """
    m = "demo" if str(mode).lower() == "demo" else "train"
    kw: Dict[str, Any] = {
        "size": size,
        "mode": m,
        "max_episode_steps": max_steps,
        "building_ratio": building_ratio,
        "occupied_ratio": occupied_ratio,
        "debug_checks": debug_checks,
        "step_debug_log": step_debug_log,
    }
    if match_train_mdp:
        merged = dict(TRAIN_PPO_ENV_KWARGS)
        merged.update(kw)
        return GridParkingEnv(**merged)  # type: ignore[arg-type]
    return GridParkingEnv(**kw)  # type: ignore[arg-type]


def draw_grid_lines(ax: plt.Axes, h: int, w: int) -> None:
    for g in np.arange(0, w + 1, 1):
        ax.axvline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)
    for g in np.arange(0, h + 1, 1):
        ax.axhline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)


def draw_obstacles_and_parking(ax: plt.Axes, grid_base: np.ndarray) -> None:
    """Bina (0), boş (1), dolu (2) — sarı hedef ayrı katmanda (üzerine yazılmaz)."""
    h, w = grid_base.shape
    cmap = mcolors.ListedColormap([_COLOR_BUILDING, _COLOR_EMPTY, _COLOR_OCCUPIED])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(
        grid_base,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(0, w, 0, h),
        zorder=1,
    )


def draw_goal_marker(ax: plt.Axes, goal: Tuple[int, int], *, goal_flash: bool) -> None:
    gr, gc = goal
    face = _COLOR_GOAL_FLASH if goal_flash else _COLOR_GOAL
    edge = "#c9a000" if not goal_flash else "#ff9800"
    lw = 2.8 if goal_flash else 1.8
    ax.scatter(
        [gc + 0.5],
        [gr + 0.5],
        s=520,
        marker="s",
        facecolors=face,
        edgecolors=edge,
        linewidths=lw,
        zorder=8,
    )


def draw_agent_marker(ax: plt.Axes, agent: Tuple[int, int]) -> None:
    ar, ac = agent
    ax.scatter(
        [ac + 0.5],
        [ar + 0.5],
        s=420,
        color=_COLOR_AGENT,
        edgecolors="white",
        linewidths=1.4,
        zorder=10,
    )


def draw_frame(ax: plt.Axes, state: Dict[str, Any]) -> None:
    """draw_grid → park durumu → hedef → ajan sırası; SmartParking için eski 'grid' yolu korunur."""
    ax.clear()

    if state.get("kind") == "grid_parking":
        grid_base = np.asarray(state["grid_base"], dtype=int)
        h, w = grid_base.shape
        goal = state["goal"]
        ag_r, ag_c = state["agent"]
        ag_r, ag_c = float(ag_r), float(ag_c)

        draw_obstacles_and_parking(ax, grid_base)
        draw_grid_lines(ax, h, w)
        draw_goal_marker(ax, goal, goal_flash=bool(state.get("goal_flash")))
        draw_agent_marker(ax, (int(ag_r), int(ag_c)))

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
                    zorder=9,
                )

        step = int(state.get("step", 0))
        action = state.get("action", "-")
        reward = float(state.get("reward", 0.0))
        mode_lbl = state.get("env_mode", "")
        extra = f"   |   {mode_lbl}" if mode_lbl else ""
        inv = bool(state.get("invalid_move"))
        inv_txt = (
            f"   |   Blocked move ({int(INVALID_ACTION_PENALTY)})"
            if inv
            else ""
        )
        leg = _legal_actions_label(state.get("legal_actions"))
        leg_txt = f"   |   {leg}" if inv and leg else ""
        banner = state.get("banner")
        title = (
            f"Step: {step}   |   Action: {action}   |   Reward: {reward:.3f}"
            f"{inv_txt}{leg_txt}{extra}"
        )
        ax.set_title(title, fontsize=11)
        if banner:
            ax.text(
                0.5,
                -0.06,
                banner,
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
                color="#333333",
            )
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return

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
        zorder=1,
    )

    draw_grid_lines(ax, h, w)

    ax.scatter(
        [ag_c + 0.5],
        [ag_r + 0.5],
        s=420,
        color=_COLOR_AGENT,
        edgecolors="white",
        linewidths=1.4,
        zorder=10,
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
                zorder=9,
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


def _random_grid_action(env: GridParkingEnv) -> int:
    """Geçerli yönlerden üret (maskesiz rastgele aksiyon spam'ini önler)."""
    legal = env._legal_action_indices()
    if not legal:
        return int(env.action_space.sample())
    j = int(env.np_random.integers(0, len(legal)))
    return int(legal[j])


def run_episode(
    env: Any,
    policy: Optional[Any] = None,
    *,
    seed: int = 42,
    max_steps: int = 200,
    mode: Optional[str] = None,
    log_episode_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Bir episode boyunca her adım için draw_frame uyumlu state listesi döndürür.
    mode: 'grid_parking' | 'smart' | None (None ise ortam tipinden seçilir).
    """
    if mode is None:
        if isinstance(env, GridParkingEnv):
            mode = "grid_parking"
        else:
            mode = "smart"

    trajectory: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=seed)
    prev_agent: Optional[Tuple[int, int]] = None
    last_step_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}

    if mode == "grid_parking":
        assert isinstance(env, GridParkingEnv)
        s0 = env.to_draw_state(
            step=0,
            action="-",
            reward=0.0,
            prev_agent=None,
            goal_flash=False,
            banner=None,
            invalid_move=False,
        )
        s0["env_mode"] = f"mode={env.mode}"
        trajectory.append(s0)
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
            if isinstance(env, GridParkingEnv):
                action = _random_grid_action(env)
            elif hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
                action = int(env.action_space.sample())
            else:
                action = 0
        else:
            action = _policy_action(policy, obs, env)

        if mode == "grid_parking":
            prev_agent = tuple(env.agent)

        obs, reward, terminated, truncated, step_info = env.step(action)
        last_step_info = step_info
        merged = {**(info if isinstance(info, dict) else {}), **step_info}

        if mode == "grid_parking":
            assert isinstance(env, GridParkingEnv)
            aname = _ACTION_NAMES_DEMO[int(action) % 4]
            flash = bool(step_info.get("goal_flash", False))
            banner = (
                "Doluluk oranı nedeniyle hedef değiştirildi."
                if step_info.get("goal_reassigned_this_step")
                else None
            )
            st = env.to_draw_state(
                step=t + 1,
                action=aname,
                reward=float(reward),
                prev_agent=prev_agent,
                goal_flash=flash,
                banner=banner,
                invalid_move=bool(step_info.get("invalid_move", False)),
            )
            la = step_info.get("legal_actions")
            if isinstance(la, list):
                st["legal_actions"] = la
            st["env_mode"] = f"mode={env.mode}"
            trajectory.append(st)
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

    if (
        log_episode_summary
        and mode == "grid_parking"
        and isinstance(env, GridParkingEnv)
    ):
        summ = last_step_info.get("episode_summary")
        if summ is not None:
            print(f"[rl_animation] episode_summary: {summ}")
        else:
            print("[rl_animation] episode_summary: (yok — ortam episode bitirmeden kesildi olabilir)")

    if len(trajectory) == 0:
        print("[rl_animation] Uyarı: trajectory boş; tek karelik yedek state eklendi.")
        if isinstance(env, GridParkingEnv):
            st = env.to_draw_state(
                step=0,
                action="N/A",
                reward=0.0,
                prev_agent=None,
                goal_flash=False,
                invalid_move=False,
            )
            st["env_mode"] = f"mode={env.mode}"
            trajectory.append(st)

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

    policy = _load_ppo_policy()
    if policy is None:
        print("[rl_animation] PPO bulunamadı veya yüklenemedi; SmartParking için rastgele politika.")

    return env, policy, "smart"


def _load_ppo_policy() -> Optional[Any]:
    for name in (
        "best_model.zip",
        "best_model",
        "ppo_parking_model_final.zip",
        "ppo_parking_model_final",
    ):
        p = MODELS_DIR / name
        if not p.exists():
            continue
        try:
            from stable_baselines3 import PPO

            pol = PPO.load(
                str(p),
                custom_objects={"lr_schedule": lambda _: 3e-4},
            )
            print(f"[rl_animation] PPO modeli yüklendi: {p}")
            print(f"[rl_animation] policy sınıfı (inference): {type(pol.policy).__name__}")
            return pol
        except Exception as exc:
            print(f"[rl_animation] PPO yüklenemedi ({p}): {exc}")
    return None


def _try_load_grid_stack(
    *,
    grid_mode: str = "demo",
    step_debug_log: bool = False,
    match_train_mdp: bool = False,
) -> Tuple[Any, Optional[Any], str]:
    """GridParkingEnv + (varsa) PPO — grid eğitimi ile uyumlu."""
    policy = _load_ppo_policy()
    if policy is None:
        print(
            "[rl_animation] PPO bulunamadı; grid ortamında rastgele politika "
            "(yalnızca geçerli yönlerden) kullanılacak — SB3 inference yok."
        )
    env = create_grid_env(
        mode=grid_mode,
        max_steps=220,
        debug_checks=False,
        step_debug_log=step_debug_log,
        match_train_mdp=match_train_mdp,
    )
    return env, policy, "grid_parking"


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
    use_smart_parking: bool = False,
    grid_mode: str = "demo",
    step_debug_log: bool = False,
    show_preview: bool = True,
    match_train_mdp: bool = False,
    log_episode_summary: bool = False,
    plain_train_reward: bool = False,
) -> Path:
    """
    GridParkingEnv (varsayılan) veya SmartParking ile animasyon üretir;
    MP4 veya GIF yazar; show_preview=True ise plt.show() ile önizleme açar.

    Args:
        episode_seed: Episode tohumu (grid reset ile uyumlu).
        fps: Kaydedilen MP4/GIF saniyedeki kare sayısı (2–4 genelde rahat izlenir).
        preview_ms_per_frame: Ekranda plt.show ile oynatırken kareler arası süre (ms);
            büyütmek animasyonu yavaşlatır (ör. 800–1200).
        use_smart_parking: True ise CSV tabanlı SmartParkingEnv + uyumlu PPO dener.
        grid_mode: train veya demo — GridParkingEnv modu.
        step_debug_log: True ise her adımda konsola ACTION/VALID/OK/reward yazdırılır.
        show_preview: False ise sadece dosya yazılır (plt.show atlanır, CI / başsız ortam).
        match_train_mdp: True ise rl_model ile aynı ödül (TRAIN_PPO_ENV_KWARGS); train modunda
            varsayılan olarak zaten açılır (plain_train_reward ile kapatılır).
        log_episode_summary: Bölüm sonunda osilasyon / benzersiz hücre / aksiyon dağılımı yazdır.
        plain_train_reward: True ise train modunda bile osilasyon/tekrar cezası olmadan eski ödül.
    """
    env: Any
    policy: Optional[Any]
    mode: str

    use_train_mdp = (not plain_train_reward) and (
        match_train_mdp or str(grid_mode).lower() == "train"
    )
    if str(grid_mode).lower() == "train" and use_train_mdp:
        print(
            "[rl_animation] Train ödülü: TRAIN_PPO_ENV_KWARGS (osilasyon + tekrar ziyaret cezası, "
            "güçlü shaping) — kapatmak için --plain-train-reward"
        )

    if use_smart_parking:
        env, policy, mode = _try_load_smart_stack()
        if env is None:
            print("[rl_animation] SmartParking yok; grid yığınına düşülüyor.")
            env, policy, mode = _try_load_grid_stack(
                grid_mode=grid_mode,
                step_debug_log=step_debug_log,
                match_train_mdp=use_train_mdp,
            )
    else:
        env, policy, mode = _try_load_grid_stack(
            grid_mode=grid_mode,
            step_debug_log=step_debug_log,
            match_train_mdp=use_train_mdp,
        )

    print(f"[rl_animation] Animasyon modu: {mode}, seed={episode_seed}")

    trajectory = run_episode(
        env,
        policy,
        seed=episode_seed,
        max_steps=220,
        mode=mode,
        log_episode_summary=log_episode_summary or step_debug_log,
    )

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    draw_frame(ax, trajectory[0])

    out_path = _save_animation(fig, trajectory, fps=fps)

    if not show_preview:
        print("[rl_animation] Önizleme atlandı (--no-show). Çıktı:", out_path)
        return out_path

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
    parser = argparse.ArgumentParser(description="Akıllı Park grid RL animasyonu")
    parser.add_argument(
        "--mode",
        choices=("train", "demo"),
        default="demo",
        help="GridParkingEnv modu: train=statik hedef/grid, demo=dinamik doluluk",
    )
    parser.add_argument("--seed", type=int, default=42, help="Episode tohumu")
    parser.add_argument(
        "--legacy-smart",
        action="store_true",
        help="CSV SmartParkingEnv + PPO yolunu dene (grid modeliyle uyumsuz olabilir)",
    )
    parser.add_argument(
        "--step-debug-log",
        action="store_true",
        help="Grid ortamında her adım: state, action, mask, reward, osilasyon bayrağı",
    )
    parser.add_argument(
        "--train-mdp",
        action="store_true",
        help="Demo modunda bile rl_model ödül ayarlarını kullan (nadiren gerekir)",
    )
    parser.add_argument(
        "--plain-train-reward",
        action="store_true",
        help="Train modunda eski ödül (osilasyon/tekrar cezası yok); varsayılan train’de cezalar açık",
    )
    parser.add_argument(
        "--episode-summary",
        action="store_true",
        help="Bölüm bitince episode_summary (osilasyon sayısı, benzersiz hücre, aksiyon dağılımı)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="MP4/GIF kaydettikten sonra plt.show açma (başsız / otomasyon)",
    )
    args = parser.parse_args()
    try:
        generate_animation(
            episode_seed=args.seed,
            grid_mode=args.mode,
            use_smart_parking=bool(args.legacy_smart),
            step_debug_log=bool(args.step_debug_log),
            show_preview=not bool(args.no_show),
            match_train_mdp=bool(args.train_mdp),
            log_episode_summary=bool(args.episode_summary),
            plain_train_reward=bool(args.plain_train_reward),
        )
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
