"""
noisy_sandbox.py — НЕДЕТЕРМИНИРОВАННАЯ среда поверх v4 Sandbox.

Провал с вероятностью flake_p транзиентно «проходит» (failure→running) — моделирует
реальный шум (нагрузка, гонки, мигающие проверки). Этим ломается несущее допущение
закона `verify→trust`: один успех ≠ безопасно. Сид фиксирован → воспроизводимо.
"""

import random

from devops_agent.v4.sandbox import Obs, Sandbox


class NoisySandbox(Sandbox):
    def __init__(self, truth: dict, flake_p: float = 0.3, seed: int = 0, network: str = "v5noise"):
        super().__init__(truth, network=network)
        self.flake_p = flake_p
        self._rng = random.Random(seed)

    def provision(self, name: str, knobs=None, **kw) -> Obs:
        obs = super().provision(name, knobs, **kw)
        if obs.phase != "running" and self._rng.random() < self.flake_p:
            return Obs("running", 0, False)     # транзиентный ЛОЖНЫЙ успех на провальном значении
        return obs
