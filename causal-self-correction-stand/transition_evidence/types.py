"""Public immutable value types and internal shelf value objects."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, TypeAlias, TypedDict, runtime_checkable

JsonScalar: TypeAlias = None | bool | int | float | str
Observation: TypeAlias = Mapping[str, JsonScalar]
SnapshotPath: TypeAlias = str | PathLike[str]


class InterventionDescription(TypedDict):
    variable: str
    operation: str


class ActionDescription(TypedDict):
    id: str
    intervention: InterventionDescription


class StepResponse(TypedDict):
    action_id: str
    outcome: JsonScalar
    obs_after: Observation


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    action_id: str
    changed: frozenset[str]
    unchanged: frozenset[str]
    unknown: frozenset[str]


@dataclass(frozen=True, slots=True)
class CanonicalState:
    """An immutable canonical snapshot projection."""

    _mapping: Mapping[str, object]
    _json_bytes: bytes

    def to_json_bytes(self) -> bytes:
        return self._json_bytes

    def as_mapping(self) -> Mapping[str, object]:
        return self._mapping


@runtime_checkable
class EnvironmentPort(Protocol):
    def observe(self) -> Observation: ...

    def action_space(self) -> Sequence[ActionDescription]: ...

    def step(self, action_id: str) -> StepResponse: ...


@dataclass(frozen=True, slots=True)
class Variable:
    id: str
    domain_tokens: frozenset[str]
    domain_values: tuple[JsonScalar, ...]
    observable: bool


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    target: str
    operation: str

    def as_mapping(self) -> dict[str, object]:
        return {"id": self.id, "intervention": {"variable": self.target, "operation": self.operation}}


@dataclass(frozen=True, slots=True)
class AtlasHypothesis:
    content_id: str
    source_ref: str
    cause: str
    effect: str
    relation: str


@dataclass(frozen=True, slots=True)
class Policy:
    support_min: int
    refute_min: int
    candidate_generation: str
    proposed_prediction: str = "unknown"


@dataclass(frozen=True, slots=True)
class Atlas:
    variables: Mapping[str, Variable]
    actions: Mapping[str, Action]
    hypotheses: tuple[AtlasHypothesis, ...]
    policy: Policy
    atlas_id: str

    @property
    def observable_ids(self) -> tuple[str, ...]:
        return tuple(sorted(variable.id for variable in self.variables.values() if variable.observable))


@dataclass(frozen=True, slots=True)
class EvidenceDelta:
    supports: int = 0
    contradicts: int = 0


@dataclass(frozen=True, slots=True)
class HypothesisState:
    id: str
    source_ref: str | None
    cause: str
    effect: str
    relation: str
    provenance: str
    supports: int = 0
    contradicts: int = 0
    supporting_evidence: tuple[int, ...] = ()
    contradicting_evidence: tuple[int, ...] = ()

    def status(self, policy: Policy) -> str:
        if self.contradicts >= policy.refute_min:
            return "refuted"
        if self.supports >= policy.support_min and self.contradicts == 0:
            return "supported"
        return "proposed"


@dataclass(frozen=True, slots=True)
class Experience:
    id: int
    before: Mapping[str, JsonScalar]
    prediction: Mapping[str, object]
    action: Mapping[str, object]
    outcome: JsonScalar
    after: Mapping[str, JsonScalar]
    informative: bool
    comparison: Mapping[str, object]
    prediction_commitment: str
    previous_entry_commitment: str
    entry_commitment: str
    pre_chain_commitment: str
    post_chain_commitment: str


@dataclass(frozen=True, slots=True)
class Shelf:
    version: int
    current_observation: Mapping[str, JsonScalar] | None
    hypotheses: Mapping[str, HypothesisState]
    experiences: tuple[Experience, ...]
    chain_commitment: str
    state_commitment: str


def readonly_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Recursively expose a JSON-like mapping without mutable containers."""

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return MappingProxyType({str(key): freeze(child) for key, child in value.items()})
