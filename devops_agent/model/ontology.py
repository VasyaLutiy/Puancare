"""
WorldModel: загружает ontology.yaml и предоставляет типизированный API.

Используется world.py (classify), agent.py (classify, obvious_cause,
fallback_cause, repair_kind, repair_carve) и oracle.py (causes).
"""

from __future__ import annotations

import os

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "ontology.yaml")


class WorldModel:
    def __init__(self, data: dict) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str | None = None) -> "WorldModel":
        p = path or _DEFAULT_PATH
        with open(p) as f:
            data = yaml.safe_load(f)
        return cls(data)

    def classify(self, obs) -> str:
        """
        obs → phase по sensors (первое совпавшее правило).
        obs: любой объект с полями oom_killed и exit_code (Obs или SimpleNamespace).
        """
        for rule in self._data["sensors"]:
            when = rule.get("when")
            if when is None:  # else-правило
                return rule["phase"]
            if all(getattr(obs, k, None) == v for k, v in when.items()):
                return rule["phase"]
        return "error"

    @property
    def symptoms(self) -> list[str]:
        return list(self._data["symptoms"])

    @property
    def causes(self) -> list[str]:
        return list(self._data["causes"])

    def obvious_cause(self, symptom: str) -> str | None:
        return self._data.get("obvious", {}).get(symptom)

    def fallback_cause(self, symptom: str) -> str | None:
        return self._data.get("fallback", {}).get(symptom)

    def repair_kind(self, cause: str) -> str | None:
        """'mem_threshold' | 'bad_config' | None"""
        return self._data.get("repair", {}).get(cause, {}).get("kind")

    def repair_carve(self, cause: str) -> bool:
        return bool(self._data.get("repair", {}).get(cause, {}).get("carve", False))
