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
    requires: list = field(default_factory=list)      # имена предикатов → :precondition (positive)
    establishes: list = field(default_factory=list)   # имена предикатов → :effect
    binds_actuator: str | None = None                  # ключ рычага или None (deploy)


@dataclass
class MetaDomain:
    """
    Растущий объект-домен. domain_gen сериализует его в PDDL.
      predicates  — только МИНТНУТЫЕ предикаты (mem_ok, config_ok); running эмитится отдельно.
      interventions — только МИНТНУТЫЕ set_* действия; deploy эмитится отдельно из deploy_requires.
      deploy_requires — предикаты, которых требует deploy (растёт при каждом минтинге).
    """
    predicates: dict = field(default_factory=dict)        # name -> Predicate
    interventions: dict = field(default_factory=dict)     # name -> Intervention (set_* only)
    deploy_requires: list = field(default_factory=list)   # имена предикатов


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
    """Детерминированное имя предиката из ключа рычага (антисиноним + детерминизм домена)."""
    return _CANON.get(dimension, f"{dimension}_ok")


def validate_proposal(raw: dict, known_actuators: list) -> dict:
    """
    Проверить ответ LLM против грамматики и вернуть нормализованный фрагмент.
    Нарушение → ValueError (proposer словит и ретрайнет / агент опровергнет).
    Имя предиката НЕ доверяем LLM — выводим канонически; raw-имя сохраняем как _llm_name (лог).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"proposal не dict: {raw!r}")
    dim = raw.get("dimension")
    kind = raw.get("kind")
    if dim not in known_actuators:
        raise ValueError(f"dimension {dim!r} не из поверхности рычагов {known_actuators}")
    if kind not in VALID_KINDS:
        raise ValueError(f"kind {kind!r} не из {VALID_KINDS}")
    return {
        "dimension": dim,
        "kind": kind,
        "predicate_name": canonical_predicate_name(dim),  # каноничное, не LLM-текст
        "_llm_name": raw.get("predicate_name"),           # что предложил LLM (лог)
        "rationale": raw.get("rationale", ""),
    }
