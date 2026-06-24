"""
meta_schema.py — ФИКСИРОВАННАЯ мета-грамматика v3 (СЕМЯ, рукописное, доменно-независимое).

Это «грамматика операбельности», а не словарь devops: что значит быть управляемым ЧЕМ УГОДНО.
6 слотов-типов; в MVP активны Entity(service), Resource(память), Setting(config), Status(фазы).
Dependency / темпоральные агрегаты — вне MVP (см. DevopsPlanV3.md §Scope cuts).

КЛЮЧЕВОЕ РЕШЕНИЕ: имя PDDL-предиката ВЫВОДИТСЯ из рычага (canonical_predicate_name),
а не берётся из текста LLM → убивает синонимию (mem_ok vs memory_sufficient) и даёт
детерминизм сгенерённого домена. LLM предлагает только СЕМАНТИКУ (рычаг↔симптом↔kind).
"""

from dataclasses import dataclass, field


# --- Мета-типы (слоты грамматики) -------------------------------------------

@dataclass
class Actuator:
    """Рычаг, который агент умеет дёргать (ДАН из agent_view, не выучивается)."""
    key: str          # "memory" | "config"
    world_key: str    # ключ в World.apply(spec): "mem_bucket" | "config"
    default: object   # значение до минтинга стратегии: memory→MEM_BASE, config→incumbent


@dataclass
class Predicate:
    """Булев предикат-«тень» рычага. name выводится из bound_to (НЕ из LLM)."""
    name: str         # канонично: mem_ok / config_ok
    bound_to: str     # ключ актуатора, чьей тенью является предикат


@dataclass
class Intervention:
    """PDDL-действие. set_<dim> устанавливает предикат; deploy — терминальное."""
    name: str                    # "set_mem" | "set_config" | "deploy"
    requires: list[str] = field(default_factory=list)      # имена предикатов → :precondition
    establishes: list[str] = field(default_factory=list)   # имена предикатов → :effect
    binds_actuator: str | None = None                       # ключ рычага или None (deploy)


@dataclass
class MetaDomain:
    """Растущий объект-домен: что выучено. domain_gen сериализует его в PDDL."""
    predicates: dict[str, Predicate] = field(default_factory=dict)
    interventions: dict[str, Intervention] = field(default_factory=dict)
    deploy_requires: list[str] = field(default_factory=list)  # растёт по мере минтинга


# --- JSON-дискриминатор для LLM (передаётся в AzureJSON.ask(schema=...)) -----

PROPOSAL_SCHEMA = {
    "dimension": "ключ рычага из ДАННОЙ поверхности актуаторов, что разрешает этот симптом",
    "kind": "один из: ordered_monotone | categorical | boolean",
    "predicate_name": "человекочитаемое имя предиката (только для лога)",
    "rationale": "одна строка: почему именно этот рычаг",
}

VALID_KINDS = ("ordered_monotone", "categorical", "boolean")

# Соответствие рычаг → каноничное имя предиката (совпадает с рукописным v2-доменом)
_CANON = {"memory": "mem_ok", "config": "config_ok"}


def canonical_predicate_name(dimension: str) -> str:
    """Детерминированное имя предиката из ключа рычага. TODO(v3): _CANON.get(dim, f'{dim}_ok')."""
    raise NotImplementedError("TODO(v3): см. DevopsPlanV3.md §Канонизация имён")


def validate_proposal(raw: dict, known_actuators: list[str]) -> dict:
    """
    Проверить ответ LLM против грамматики и вернуть нормализованный фрагмент.
    Правила: raw['dimension'] ∈ known_actuators; raw['kind'] ∈ VALID_KINDS.
    TODO(v3): на нарушении — ValueError (вызовет ретрай у proposer'а).
    """
    raise NotImplementedError("TODO(v3): валидация дискриминатора")
