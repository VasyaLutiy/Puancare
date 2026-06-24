"""
proposer.py — SchemaProposer: на разрыве предлагает кандидат-фрагмент схемы (ГИПОТЕЗУ).

Обобщает v2 Oracle (причина→схема). Cost-gated ВЫЗЫВАЮЩИМ: meta_agent зовёт propose()
только когда covered_symptoms[symptom] пуст (холод) — как v2 звал oracle при пустом reverse_index.
Песочница — судья: неверная гипотеза опровергается интервенцией, не доверием LLM.

Live   = живой Azure (utils_azure.AzureJSON), жжёт токены, для пруфа/фальсификатора.
Replay = fixtures/proposals.json {symptom: proposal}, детерминир. регрессия без токенов.
"""

import json
import os
import sys
from pathlib import Path

# Корень проекта в sys.path (как в oracle.py) — для импорта utils_azure
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from devops_agent.v3.meta_schema import PROPOSAL_SCHEMA, validate_proposal

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "proposals.json")


class SchemaProposer:
    """Контракт: propose(symptom, ctx, known_actuators) → нормализованный фрагмент (dict)."""
    def __init__(self) -> None:
        self.n_calls = 0
        self.total_tokens = 0

    def propose(self, symptom: str, ctx: dict, known_actuators: list) -> dict:
        raise NotImplementedError


class LiveProposer(SchemaProposer):
    """
    Живой Azure. LLM = генератор ГИПОТЕЗ: какой рычаг разрешает симптом + его kind.
    Имя предиката НЕ берём у LLM (validate_proposal канонизирует из рычага). Паттерн = Oracle.suggest_causes.
    """
    def __init__(self, env_path: str | None = None) -> None:
        super().__init__()
        from utils_azure import AzureJSON
        self._az = AzureJSON(env_path=env_path)

    def propose(self, symptom: str, ctx: dict, known_actuators: list) -> dict:
        system = (
            "You are the schema-proposer of a SELF-GROWING PDDL planning domain that deploys services.\n"
            "A deploy attempt produced a failure SYMPTOM. Propose the ONE lever ('dimension') that resolves "
            "it, plus the variable 'kind' of that lever (the kind selects the resolution strategy).\n"
            f"'dimension' MUST be EXACTLY one of the available lever keys: {known_actuators} (verbatim string).\n"
            "'kind' is one of:\n"
            "  ordered_monotone — a resource you can give MORE of, monotone (more never hurts), e.g. memory;\n"
            "  categorical      — an unordered choice with right/wrong values, e.g. a config option;\n"
            "  boolean          — an on/off flag.\n"
            "This is a HYPOTHESIS; a sandbox will verify it by intervention. Respond with valid JSON only."
        )
        user = (
            f"symptom: {symptom!r}\n"
            f"context: {ctx}\n"
            f"available_levers: {known_actuators}"
        )
        result = self._az.ask(system=system, user=user, schema=PROPOSAL_SCHEMA)

        # Учёт токенов (как в oracle.py)
        if getattr(self._az, "_last_usage", None) is not None:
            self.total_tokens += getattr(self._az._last_usage, "total_tokens", 0)
        self.n_calls += 1

        return validate_proposal(result, known_actuators)


class ReplayProposer(SchemaProposer):
    """Записанные ответы LLM (offline). n_calls растёт (для кривой), токены не считаются."""
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        with open(path or _FIXTURES) as f:
            self._table = json.load(f)

    def propose(self, symptom: str, ctx: dict, known_actuators: list) -> dict:
        self.n_calls += 1
        raw = self._table.get(symptom)
        if raw is None:
            raise KeyError(f"нет replay-фикстуры для симптома {symptom!r}")
        return validate_proposal(raw, known_actuators)


# ---------------------------------------------------------------------------
# Self-test — ЖИВОЙ Azure (2 дешёвых вызова). Проверяет, что LLM как генератор
# гипотез верно мапит симптом → рычаг. Требует .env (есть на srv).
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== LiveProposer self-test (живой Azure, 2 вызова) ===\n")
    errors = []
    p = LiveProposer()
    levers = ["memory", "config"]
    cases = [
        ("oom_killed", {"service": "svc_a", "workload_class": "heavy",
                        "values": {"memory": 64, "config": "good"}}, "memory"),
        ("unhealthy", {"service": "svc_e", "workload_class": "standard",
                       "values": {"memory": 512, "config": "bad"}}, "config"),
    ]
    for symptom, ctx, expect_dim in cases:
        prop = p.propose(symptom, ctx, levers)
        print(f"  {symptom!r} → {prop}  (tokens={p.total_tokens})")
        if prop["dimension"] != expect_dim:
            errors.append(f"{symptom}: ожидали dimension={expect_dim!r}, got {prop['dimension']!r}")
        if prop["predicate_name"] not in ("mem_ok", "config_ok"):
            errors.append(f"{symptom}: каноничное имя предиката неожиданно {prop['predicate_name']!r}")

    print(f"\n  n_calls={p.n_calls}, total_tokens={p.total_tokens}")
    if p.n_calls != 2:
        errors.append(f"n_calls: ожидали 2, got {p.n_calls}")
    if p.total_tokens == 0:
        errors.append("total_tokens == 0 — учёт не работает")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("LiveProposer OK — живой LLM мапит симптом→рычаг, токены учтены.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v3.proposer --selftest")
