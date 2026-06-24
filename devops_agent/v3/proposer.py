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

    def propose(self, symptom: str, ctx: dict, known_actuators: list[str]) -> dict:
        raise NotImplementedError


class LiveProposer(SchemaProposer):
    """
    system = мета-грамматика + роль («предложи, какой рычаг разрешает симптом»);
    user   = симптом + ctx + список known_actuators;
    AzureJSON.ask(system, user, schema=PROPOSAL_SCHEMA) → dict; токены из _last_usage.
    Затем validate_proposal(...). TODO(v3): порт паттерна Oracle.suggest_causes.
    """
    def __init__(self, env_path: str | None = None) -> None:
        super().__init__()
        # TODO(v3): from utils_azure import AzureJSON; self._az = AzureJSON(env_path=env_path)
        raise NotImplementedError("TODO(v3): LiveProposer.__init__")

    def propose(self, symptom: str, ctx: dict, known_actuators: list[str]) -> dict:
        raise NotImplementedError("TODO(v3): LiveProposer.propose")


class ReplayProposer(SchemaProposer):
    """Читает записанные proposals.json. Токены не считает (n_calls растёт, tokens=0)."""
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        with open(path or _FIXTURES) as f:
            self._table = json.load(f)

    def propose(self, symptom: str, ctx: dict, known_actuators: list[str]) -> dict:
        """TODO(v3): self.n_calls+=1; raw=self._table[symptom]; return validate_proposal(raw, known_actuators)."""
        raise NotImplementedError("TODO(v3): ReplayProposer.propose")
