"""
meta_bios.py — MetaBios: выращенный домен + выученные значения + амортизация.

Обобщение v2 BiosState: вместо рукописной схемы — растущий MetaDomain. Старт = только
рычаги (из agent_view) + пустой домен с одним deploy (БЕЗ mem_ok/config_ok — их минтит опыт).
covered_symptoms — зеркало v2 reverse_index (кэш symptom→измерение, основа амортизации).
"""

from dataclasses import dataclass, field

from devops_agent.services import agent_view
from devops_agent.v3.meta_schema import (
    Actuator,
    Intervention,
    MetaDomain,
    Predicate,
    canonical_predicate_name,
)

MEM_BASE: int = 64        # порт bios.py
MEM_CEILING: int = 16384


@dataclass
class MetaBios:
    """Живёт на весь сеанс — схема и значения копятся между задачами."""
    services: dict[str, dict]
    actuators: dict[str, Actuator]
    domain: MetaDomain
    strategies: dict[str, str]               # dimension → kind
    covered_symptoms: dict[str, str]         # symptom → dimension (кэш; пусто = холод → LLM)
    # выученные значения (порт v2):
    mem_threshold: dict[str, int] = field(default_factory=dict)       # wc → mib
    mem_threshold_svc: dict[str, int] = field(default_factory=dict)   # svc → mib (карвинг)
    bad_config: set = field(default_factory=set)                      # {(svc, opt)}
    incumbent_config: dict[str, str] = field(default_factory=dict)
    llm_calls: int = 0                       # счётчик; харнесс снимает дельту per-task

    @classmethod
    def seed(cls, svc_names: list[str]) -> "MetaBios":
        """
        Семя: рычаги memory(default=MEM_BASE) и config(default=incumbent из agent_view),
        домен с единственной интервенцией deploy (establishes=[running], deploy_requires=[]).
        НИ mem_ok, НИ config_ok — доказательство, что они порождены, не вписаны.
        TODO(v3): см. DevopsPlanV3.md §Семя.
        """
        raise NotImplementedError("TODO(v3): seed — рычаги + пустой deploy")

    def mint_dimension(self, dimension: str, kind: str, symptom: str) -> str:
        """
        Вшить новое измерение по подтверждаемой гипотезе LLM:
          predicate {dim}_ok (canonical) → domain.predicates
          intervention set_{dim} (establishes=[{dim}_ok]) → domain.interventions
          deploy_requires += {dim}_ok ; strategies[dimension]=kind ; covered_symptoms[symptom]=dimension
        Возвращает имя предиката. TODO(v3): §Петля cold-branch.
        """
        raise NotImplementedError("TODO(v3): mint_dimension")

    def unmint_dimension(self, dimension: str, symptom: str) -> None:
        """Откат минтинга при опровержении (refute). TODO(v3)."""
        raise NotImplementedError("TODO(v3): unmint при refute")

    def is_covered(self, symptom: str) -> str | None:
        """Известное измерение для симптома или None (None → холод → LLM)."""
        return self.covered_symptoms.get(symptom)

    def value_for(self, actuator_key: str, svc: str):
        """
        Значение рычага для World.apply: если измерение минтнуто и есть стратегия — её initial;
        иначе дефолт актуатора (memory→known threshold|MEM_BASE, config→incumbent).
        TODO(v3): связать со strategies.strategy_for.
        """
        raise NotImplementedError("TODO(v3): value_for")

    def record_running(self, svc: str, mem: int, config: str) -> None:
        """Успех: записать порог памяти (class|carve), incumbent_config. Порт agent.py success-ветки. TODO(v3)."""
        raise NotImplementedError("TODO(v3): record_running")

    def mark_bad_config(self, svc: str, opt: str) -> None:
        self.bad_config.add((svc, opt))

    def is_bad_config(self, svc: str, opt: str) -> bool:
        return (svc, opt) in self.bad_config
