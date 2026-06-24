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
    Живой Azure. system = мета-грамматика + роль; user = симптом+ctx+рычаги;
    AzureJSON.ask(schema=PROPOSAL_SCHEMA) → dict; токены из _last_usage; validate_proposal.
    TODO(v3): реализовать для harness_v3 (следующая сессия). Паттерн — Oracle.suggest_causes.
    """
    def __init__(self, env_path: str | None = None) -> None:
        super().__init__()
        raise NotImplementedError("TODO(v3): LiveProposer для harness_v3 (живой пруф)")

    def propose(self, symptom: str, ctx: dict, known_actuators: list) -> dict:
        raise NotImplementedError("TODO(v3): LiveProposer.propose")


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
