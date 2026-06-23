"""
ProblemBuilder: BiosState → problem.pddl (чистый STRIPS, без стоимостей).

Логика:
  - configopts: union всех config_options по всем сервисам;
    если пусто — добавляем «cfg_default» (always-on dummy).
  - Memory feasibility проверяется агентом (ceiling); ProblemBuilder
    проверяет только config-достижимость (config_ok факты).
  - config_ok(svc, opt):
      сервис БЕЗ config_options → True для всех opt (always-on pass-through).
      сервис С  config_options  → True только если (svc, opt) ∉ bad_config.
  - Детерминированный порядок: sorted() на objects и init-фактах.
  - Никаких :functions, :metric, cost-фактов — принцип #8.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from devops_agent.bios import BiosState


class ProblemBuilder:
    def __init__(self, bios: BiosState, pddl_dir: str) -> None:
        self.bios = bios
        self.pddl_dir = pddl_dir

    def build(self, goal_service: str) -> str:
        """Генерирует problem.pddl и возвращает путь к файлу."""
        bios = self.bios
        if goal_service not in bios.services:
            raise ValueError(f"Unknown goal service: {goal_service!r}")

        # --- Configopts (детерминированный порядок) ---
        all_opts: list[str] = sorted({
            opt
            for svc in bios.services.values()
            for opt in svc.get("config_options", [])
        })
        if not all_opts:
            all_opts = ["cfg_default"]

        # --- Services (детерминированный порядок) ---
        all_svcs: list[str] = sorted(bios.services.keys())

        lines: list[str] = [
            "(define (problem devops-p)",
            "  (:domain devops)",
            "",
            "  (:objects",
            f"    {' '.join(all_svcs)} - service",
            f"    {' '.join(all_opts)} - configopt)",
            "",
            "  (:init",
        ]

        # config_ok facts (единственные факты в :init)
        for svc in all_svcs:
            has_config = bool(bios.services[svc].get("config_options"))
            for opt in all_opts:
                ok = True if not has_config else not bios.is_bad_config(svc, opt)
                if ok:
                    lines.append(f"    (config_ok {svc} {opt})")

        lines += [
            "  )",
            "",
            f"  (:goal (running {goal_service}))",
            ")",
            "",
        ]

        content = "\n".join(lines)
        path = os.path.join(self.pddl_dir, "problem.pddl")
        with open(path, "w") as f:
            f.write(content)
        return path
