"""
nlu.py — КОМПИЛЯТОР (LLM, восприятие): сырой сигнал → JSON-фрагмент в словаре агента.

LLM = глаз без состояния: видит ТОЛЬКО сырьё одного сигнала (не граф агента) и выдаёт
конкретный JSON 6-мета-типов. Распознавание — НЕ здесь (это в recognizer, структурно).
Инструмент — utils_azure.AzureJSON(schema=...). Это и есть «вся надежда на NLU+LLM».
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

FRAGMENT_SCHEMA = {
    "label": "short type name of the thing (e.g. database, cache, gateway)",
    "resources": "JSON list of QUANTITATIVE knob kinds, each EXACTLY one of: ordered_monotone, bounded",
    "settings": "JSON list of CATEGORICAL knob kinds, each EXACTLY one of: categorical, boolean",
    "status": "name of the healthy/goal state if any, else empty string",
}


# named-рычаги (knob'ы = API мира) и их фикс-kind (kind — свойство knob'а, не выдумка LLM)
KNOB_KIND = {"mem": "ordered_monotone", "pool": "bounded", "config": "categorical"}

ENTITY_SCHEMA = {
    "label": "short type name of the thing",
    "levers": f"JSON list of knob names this thing has, each EXACTLY one of: {list(KNOB_KIND)}",
    "deps": "JSON list of names of OTHER things it depends on / connects to (from the raw), else []",
    "status": "healthy/goal state name, else empty string",
}


class LiveNLU:
    def __init__(self, env_path: str | None = None):
        from utils_azure import AzureJSON
        self._az = AzureJSON(env_path=env_path)
        self.n_calls = 0
        self.total_tokens = 0

    def perceive_entity(self, raw: dict, ident: str) -> dict:
        """Сырьё → кандидат {id,label,levers[names],kinds,deps[names],status}. levers ⊂ knob-API."""
        system = (
            "You are the PERCEPTION layer of an ops agent. Given RAW observables of one thing, "
            "say which platform KNOBS it has (only from the given list), what OTHER things it "
            "depends on (by name, from the raw), and its healthy state. This is a hypothesis — "
            "reality will verify it. Respond with valid JSON only."
        )
        r = self._az.ask(system=system, user=f"raw observables: {raw}", schema=ENTITY_SCHEMA)
        if getattr(self._az, "_last_usage", None) is not None:
            self.total_tokens += getattr(self._az._last_usage, "total_tokens", 0)
        self.n_calls += 1
        levers = [l for l in r.get("levers", []) if l in KNOB_KIND]
        return {
            "id": ident,
            "label": r.get("label", "thing"),
            "levers": levers,
            "kinds": {l: KNOB_KIND[l] for l in levers},
            "deps": list(r.get("deps", [])),
            "status": (r.get("status") or None),
        }

    def perceive(self, raw: dict, ident: str | None = None) -> dict:
        """Сырой сигнал (dict наблюдаемых) → фрагмент {id,label,resources,settings,status,deps}."""
        system = (
            "You are the PERCEPTION layer of an ops agent. Given RAW observables of one thing, "
            "describe it ONLY in this fixed vocabulary (do not invent fields). Quantitative knobs "
            "you can turn → resources (ordered_monotone = more-is-safer; bounded = safe window). "
            "Right/wrong-value knobs → settings (categorical/boolean). Respond with valid JSON only."
        )
        r = self._az.ask(system=system, user=f"raw observables: {raw}", schema=FRAGMENT_SCHEMA)
        if getattr(self._az, "_last_usage", None) is not None:
            self.total_tokens += getattr(self._az._last_usage, "total_tokens", 0)
        self.n_calls += 1
        res = [x for x in r.get("resources", []) if x in ("ordered_monotone", "bounded")]
        sett = [x for x in r.get("settings", []) if x in ("categorical", "boolean")]
        return {
            "id": ident or f"{r.get('label', 'thing')}_{self.n_calls}",
            "label": r.get("label", "thing"),
            "resources": res,
            "settings": sett,
            "status": (r.get("status") or None),
            "deps": [],
        }
