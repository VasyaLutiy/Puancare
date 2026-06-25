"""
proposer.py — LLM как генератор ГИПОТЕЗ для петли v4.

Петля НЕ содержит карты symptom→lever→kind. На незнакомом симптоме она спрашивает LLM:
«какой рычаг (из данной поверхности) это лечит и какой у него kind?». Это гипотеза —
проверяет docker. На рефьюте (стратегия не сошлась / противоречивый симптом на том же
рычаге) петля переспрашивает, отдав LLM конфликтующие симптомы как улику.

LLM получает: симптом + поверхность рычагов (имена + для categorical — варианты) + что уже
известно. Имён предикатов/смыслов в коде нет — всё семантическое здесь, на границе.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

PROPOSAL_SCHEMA = {
    "lever": "EXACTLY one lever key from the given surface (verbatim)",
    "kind": "one of: ordered_monotone | categorical | bounded",
    "low_symptom": "if kind=bounded: the symptom meaning value TOO LOW; else empty string",
    "high_symptom": "if kind=bounded: the symptom meaning value TOO HIGH; else empty string",
    "rationale": "one short line",
}

_KINDS = (
    "ordered_monotone — a knob where MORE is safer, monotone (e.g. a quantity you can keep raising); "
    "categorical — an unordered choice with right/wrong VALUES (try alternatives); "
    "bounded — a SAFE WINDOW where BOTH too-little AND too-much fail (needs the two directional symptoms)."
)


class Proposer:
    def __init__(self, env_path: str | None = None):
        from utils_azure import AzureJSON
        self._az = AzureJSON(env_path=env_path)
        self.n_calls = 0
        self.total_tokens = 0

    def propose(self, symptom: str, levers: dict, known: dict, refute: str | None = None) -> dict:
        """
        levers: {name: {"type": "numeric"|"categorical", "options"?: [...]}}.
        known:  {lever: kind} уже выученное (для контекста).
        refute: текст улики, если предыдущая гипотеза опровергнута песочницей.
        """
        system = (
            "You are the hypothesis generator for a self-growing planning agent that provisions an "
            "entity in a sandbox. A provisioning attempt FAILED with a symptom. Decide which LEVER "
            "(from the given surface) resolves it and its KIND. This is a HYPOTHESIS — a real sandbox "
            "will verify it; if wrong it will be refuted and you'll be asked again.\n"
            f"KINDS: {_KINDS}\n"
            "'lever' MUST be exactly one key from the surface. Respond with valid JSON only."
        )
        user = (
            f"symptom: {symptom!r}\n"
            f"lever surface: {levers}\n"
            f"already learned (lever->kind): {known}\n"
            + (f"REFUTED by sandbox — revise: {refute}\n" if refute else "")
        )
        r = self._az.ask(system=system, user=user, schema=PROPOSAL_SCHEMA)
        if getattr(self._az, "_last_usage", None) is not None:
            self.total_tokens += getattr(self._az._last_usage, "total_tokens", 0)
        self.n_calls += 1

        lever = r.get("lever")
        kind = r.get("kind")
        if lever not in levers:
            raise ValueError(f"LLM предложил рычаг {lever!r} вне поверхности {list(levers)}")
        if kind not in ("ordered_monotone", "categorical", "bounded"):
            raise ValueError(f"LLM предложил неизвестный kind {kind!r}")
        return r
