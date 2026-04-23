"""Akıllı otopark RL simülasyonu: ortam, konum ve ödül modeli.

Kullanım:
    from parking_rl.smart_parking_env import SmartParkingEnv
"""

__all__ = ["SmartParkingEnv"]


def __getattr__(name: str):
    if name == "SmartParkingEnv":
        from parking_rl.smart_parking_env import SmartParkingEnv

        return SmartParkingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
