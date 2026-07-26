"""Lifecycle API for the deterministic transition evidence engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .atlas import parse_atlas
from .canonical import (
    canonical_json_bytes,
    genesis_chain_commitment,
    genesis_entry_commitment,
    is_json_scalar,
    post_chain_commitment,
    scalar_token,
    sha256_commitment,
)
from .errors import RuntimeValidationError
from .persistence import load_shelf, save_state
from .relations import default_relation_registry
from .shelf import canonical_state, initial_hypotheses, make_candidates, seal_shelf, validate_observation
from .types import (
    Atlas, CanonicalState, Experience, HypothesisState, JsonScalar, Observation,
    Prediction, Shelf, SnapshotPath, StepResponse,
)


@dataclass(frozen=True, slots=True)
class _PendingPrediction:
    prediction: Prediction
    before: Mapping[str, JsonScalar]
    prediction_commitment: str
    pre_chain_commitment: str


class TransitionEvidenceEngine:
    """In-memory deterministic engine; use :func:`create` or :func:`load`."""

    def __init__(self, atlas: Atlas, shelf: Shelf, *, bound: bool) -> None:
        self._atlas = atlas
        self._shelf = shelf
        self._bound = bound
        self._pending: dict[str, _PendingPrediction] = {}
        self._relations = default_relation_registry()

    def bind_observation(self, observation: Observation) -> None:
        normalised = validate_observation(self._atlas, observation)
        proposed = Shelf(
            version=self._shelf.version,
            current_observation=normalised,
            hypotheses=self._shelf.hypotheses,
            experiences=self._shelf.experiences,
            chain_commitment=self._shelf.chain_commitment,
            state_commitment="",
        )
        self._shelf = seal_shelf(self._atlas, proposed)
        self._bound = True
        self._pending.clear()

    def predict(self, action_id: str) -> Prediction:
        if not self._bound or self._shelf.current_observation is None:
            raise RuntimeValidationError("OBSERVATION_UNBOUND", "")
        action = self._atlas.actions.get(action_id)
        if action is None:
            raise RuntimeValidationError("ACTION_NOT_AVAILABLE", "/action_id")
        changed: set[str] = {action.target}
        unchanged: set[str] = set()
        for hypothesis in self._shelf.hypotheses.values():
            if (
                hypothesis.cause != action.target
                or hypothesis.effect == action.target
                or hypothesis.effect not in self._atlas.observable_ids
            ):
                continue
            status = hypothesis.status(self._atlas.policy)
            if status == "supported":
                changed.add(hypothesis.effect)
            elif status == "refuted":
                unchanged.add(hypothesis.effect)
            elif self._atlas.policy.proposed_prediction == "changed":
                changed.add(hypothesis.effect)
        unchanged.difference_update(changed)
        unknown = set(self._atlas.observable_ids).difference(changed, unchanged)
        before = dict(self._shelf.current_observation)
        prediction_id = self._prediction_id(action_id, before)
        prediction = Prediction(prediction_id, action_id, frozenset(changed), frozenset(unchanged), frozenset(unknown))
        commitment = sha256_commitment({
            "atlas_id": self._atlas.atlas_id,
            "pre_chain_commitment": self._shelf.chain_commitment,
            "prediction_id": prediction.id,
            "before": before,
            "action_id": action_id,
            "changed": sorted(prediction.changed),
            "unchanged": sorted(prediction.unchanged),
            "unknown": sorted(prediction.unknown),
        })
        self._pending[prediction_id] = _PendingPrediction(prediction, before, commitment, self._shelf.chain_commitment)
        return prediction

    def accept_transition(self, prediction_id: str, step_response: StepResponse) -> int:
        """Atomically accept one externally completed transition."""
        pending = self._pending.get(prediction_id)
        if pending is None:
            raise RuntimeValidationError("PREDICTION_REQUIRED", "/prediction_id")
        response = self._validate_step_response(step_response)
        if response["action_id"] != pending.prediction.action_id:
            raise RuntimeValidationError("PREDICTION_MISMATCH", "/action_id")
        if not self._bound or self._shelf.current_observation is None:
            raise RuntimeValidationError("OBSERVATION_UNBOUND", "")
        if dict(self._shelf.current_observation) != dict(pending.before) or self._shelf.chain_commitment != pending.pre_chain_commitment:
            raise RuntimeValidationError("PREDICTION_MISMATCH", "/before")

        # Everything through sealing is local; failure leaves shelf and pending IDs intact.
        before = dict(pending.before)
        after = validate_observation(self._atlas, response["obs_after"])
        action = self._atlas.actions[pending.prediction.action_id]
        experience_id = len(self._shelf.experiences) + 1
        informative = scalar_token(before[action.target]) != scalar_token(after[action.target])
        hypotheses = dict(self._shelf.hypotheses)
        if informative:
            hypotheses = make_candidates(self._atlas, hypotheses, action.target)
            hypotheses = self._apply_evidence(hypotheses, before, after, experience_id, action.target)
        comparison = self._comparison(pending.prediction, before, after)
        prediction_projection = self._prediction_projection(pending.prediction)
        previous_entry = (
            self._shelf.experiences[-1].entry_commitment
            if self._shelf.experiences
            else genesis_entry_commitment(self._atlas.atlas_id)
        )
        entry_payload = {
            "id": experience_id,
            "before": before,
            "prediction": prediction_projection,
            "action": action.as_mapping(),
            "outcome": response["outcome"],
            "after": after,
            "informative": informative,
            "comparison": comparison,
            "prediction_commitment": pending.prediction_commitment,
            "previous_entry_commitment": previous_entry,
            "pre_chain_commitment": pending.pre_chain_commitment,
        }
        entry_commitment = sha256_commitment(entry_payload)
        post_chain = post_chain_commitment(pending.pre_chain_commitment, entry_commitment)
        experience = Experience(
            id=experience_id, before=before, prediction=prediction_projection, action=action.as_mapping(),
            outcome=response["outcome"], after=after, informative=informative, comparison=comparison,
            prediction_commitment=pending.prediction_commitment,
            previous_entry_commitment=previous_entry,
            entry_commitment=entry_commitment,
            pre_chain_commitment=pending.pre_chain_commitment,
            post_chain_commitment=post_chain,
        )
        proposed = Shelf(
            version=self._shelf.version + 1,
            current_observation=after,
            hypotheses=hypotheses,
            experiences=(*self._shelf.experiences, experience),
            chain_commitment=post_chain,
            state_commitment="",
        )
        next_shelf = seal_shelf(self._atlas, proposed)
        # The only commit point: no validation or computation occurs after this.
        self._shelf = next_shelf
        self._pending = {}
        return experience_id

    def snapshot(self) -> CanonicalState:
        return canonical_state(self._atlas, self._shelf)

    def save(self, snapshot_path: SnapshotPath) -> None:
        save_state(snapshot_path, self.snapshot().to_json_bytes())

    def _prediction_id(self, action_id: str, before: Mapping[str, JsonScalar]) -> str:
        return sha256_commitment({"version": self._shelf.version, "before": dict(before), "action_id": action_id})

    def _validate_step_response(self, response: object) -> dict[str, Any]:
        if not isinstance(response, Mapping) or set(response) != {"action_id", "outcome", "obs_after"}:
            raise RuntimeValidationError("OBSERVATION_SHAPE", "/step_response")
        action_id = response["action_id"]
        if not isinstance(action_id, str):
            raise RuntimeValidationError("OBSERVATION_SHAPE", "/step_response/action_id")
        if not is_json_scalar(response["outcome"]):
            raise RuntimeValidationError("OBSERVATION_SHAPE", "/step_response/outcome")
        return {"action_id": action_id, "outcome": response["outcome"], "obs_after": response["obs_after"]}

    def _apply_evidence(self, hypotheses: Mapping[str, HypothesisState], before: Mapping[str, JsonScalar],
                        after: Mapping[str, JsonScalar], experience_id: int, action_target: str) -> dict[str, HypothesisState]:
        updated = dict(hypotheses)
        for hypothesis_id, hypothesis in hypotheses.items():
            # Causal eligibility is a lifecycle concern, not relation semantics.
            if hypothesis.cause != action_target:
                continue
            evaluator = self._relations.get(hypothesis.relation)
            if evaluator is None:
                # Loaded snapshots may contain a relation that v1 does not evaluate.
                # This fails under the normal local-computation boundary, before commit.
                raise RuntimeValidationError("UNKNOWN_RELATION", "/relation")
            delta = evaluator.evaluate(before, after, hypothesis.cause, hypothesis.effect)
            if not delta.supports and not delta.contradicts:
                continue
            updated[hypothesis_id] = HypothesisState(
                id=hypothesis.id, source_ref=hypothesis.source_ref, cause=hypothesis.cause,
                effect=hypothesis.effect, relation=hypothesis.relation, provenance=hypothesis.provenance,
                supports=hypothesis.supports + delta.supports, contradicts=hypothesis.contradicts + delta.contradicts,
                supporting_evidence=(*hypothesis.supporting_evidence, *((experience_id,) if delta.supports else ())),
                contradicting_evidence=(*hypothesis.contradicting_evidence, *((experience_id,) if delta.contradicts else ())),
            )
        return updated

    @staticmethod
    def _prediction_projection(prediction: Prediction) -> dict[str, object]:
        return {"id": prediction.id, "action_id": prediction.action_id, "changed": sorted(prediction.changed),
                "unchanged": sorted(prediction.unchanged), "unknown": sorted(prediction.unknown)}

    @staticmethod
    def _comparison(prediction: Prediction, before: Mapping[str, JsonScalar], after: Mapping[str, JsonScalar]) -> dict[str, object]:
        actual_changed = {variable_id for variable_id in before if scalar_token(before[variable_id]) != scalar_token(after[variable_id])}
        return {
            "actual_changed": sorted(actual_changed),
            "correct_changed": sorted(actual_changed.intersection(prediction.changed)),
            "correct_unchanged": sorted((set(before).difference(actual_changed)).intersection(prediction.unchanged)),
            "unknown": sorted(prediction.unknown),
        }


def create(atlas_bytes: bytes) -> TransitionEvidenceEngine:
    atlas = parse_atlas(atlas_bytes)
    initial = Shelf(0, None, initial_hypotheses(atlas), (), genesis_chain_commitment(atlas.atlas_id), "")
    return TransitionEvidenceEngine(atlas, seal_shelf(atlas, initial), bound=False)


def load(snapshot_path: SnapshotPath, atlas_bytes: bytes) -> TransitionEvidenceEngine:
    atlas = parse_atlas(atlas_bytes)
    return TransitionEvidenceEngine(atlas, load_shelf(snapshot_path, atlas), bound=False)
