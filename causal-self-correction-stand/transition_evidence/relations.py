"""Extensible relation evaluation, with the v1 ``cochanges`` evaluator."""

from __future__ import annotations

from typing import Mapping, Protocol

from .canonical import scalar_token
from .types import EvidenceDelta, Observation


class RelationEvaluator(Protocol):
    name: str

    def evaluate(
        self, before: Observation, after: Observation, cause: str, effect: str
    ) -> EvidenceDelta: ...


class CochangesEvaluator:
    name = "cochanges"

    def evaluate(
        self, before: Observation, after: Observation, cause: str, effect: str
    ) -> EvidenceDelta:
        # Hypotheses over non-observable endpoints are allowed by the atlas
        # endpoint rule but cannot receive evidence from an observable transition.
        if cause not in before or cause not in after or effect not in before or effect not in after:
            return EvidenceDelta()
        if scalar_token(before[cause]) == scalar_token(after[cause]):
            return EvidenceDelta()
        if scalar_token(before[effect]) != scalar_token(after[effect]):
            return EvidenceDelta(supports=1)
        return EvidenceDelta(contradicts=1)


def default_relation_registry() -> Mapping[str, RelationEvaluator]:
    return {"cochanges": CochangesEvaluator()}
