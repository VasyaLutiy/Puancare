"""Pure shelf validation and canonical projection helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .canonical import canonical_json_bytes, content_id, pointer_join, scalar_token, sha256_commitment
from .errors import RuntimeValidationError
from .types import Atlas, CanonicalState, Experience, HypothesisState, JsonScalar, Observation, Shelf, readonly_mapping


def validate_observation(atlas: Atlas, observation: object, *, code: str = "OBSERVATION_SHAPE") -> dict[str, JsonScalar]:
    if not isinstance(observation, Mapping) or any(not isinstance(key, str) for key in observation):
        raise RuntimeValidationError(code, "", "observation must be an object")
    expected = set(atlas.observable_ids)
    supplied = set(observation)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise RuntimeValidationError(code, pointer_join("", missing[0]), "observable value missing")
    if unknown:
        raise RuntimeValidationError(code, pointer_join("", unknown[0]), "unknown observation variable")
    normalised: dict[str, JsonScalar] = {}
    for variable_id in atlas.observable_ids:
        value = observation[variable_id]
        try:
            token = scalar_token(value)  # rejects non-JSON values through json.dumps
        except (TypeError, ValueError):
            raise RuntimeValidationError("VALUE_OUT_OF_DOMAIN", pointer_join("", variable_id)) from None
        if token not in atlas.variables[variable_id].domain_tokens:
            raise RuntimeValidationError("VALUE_OUT_OF_DOMAIN", pointer_join("", variable_id))
        normalised[variable_id] = value  # type: ignore[assignment]
    return normalised


def hypothesis_projection(hypothesis: HypothesisState, atlas: Atlas) -> dict[str, object]:
    return {
        "id": hypothesis.id,
        "source_ref": hypothesis.source_ref,
        "cause": hypothesis.cause,
        "effect": hypothesis.effect,
        "relation": hypothesis.relation,
        "provenance": hypothesis.provenance,
        "status": hypothesis.status(atlas.policy),
        "evidence": {"supports": hypothesis.supports, "contradicts": hypothesis.contradicts},
        "supporting_evidence": list(sorted(hypothesis.supporting_evidence)),
        "contradicting_evidence": list(sorted(hypothesis.contradicting_evidence)),
    }


def experience_projection(experience: Experience) -> dict[str, object]:
    return {
        "id": experience.id,
        "before": dict(experience.before),
        "prediction": dict(experience.prediction),
        "action": dict(experience.action),
        "outcome": experience.outcome,
        "after": dict(experience.after),
        "informative": experience.informative,
        "comparison": dict(experience.comparison),
        "prediction_commitment": experience.prediction_commitment,
        "previous_entry_commitment": experience.previous_entry_commitment,
        "entry_commitment": experience.entry_commitment,
        "pre_chain_commitment": experience.pre_chain_commitment,
        "post_chain_commitment": experience.post_chain_commitment,
    }


def state_projection(atlas: Atlas, shelf: Shelf) -> dict[str, object]:
    """Canonical shelf fields bound by state commitment (without that commitment)."""
    return {
        "schema": "transition-shelf/v1",
        "version": shelf.version,
        "atlas_id": atlas.atlas_id,
        "current_observation": None if shelf.current_observation is None else dict(shelf.current_observation),
        "hypotheses": [
            hypothesis_projection(shelf.hypotheses[hypothesis_id], atlas)
            for hypothesis_id in sorted(shelf.hypotheses)
        ],
        "experiences": [experience_projection(experience) for experience in sorted(shelf.experiences, key=lambda item: item.id)],
        "chain_commitment": shelf.chain_commitment,
    }


def seal_shelf(atlas: Atlas, shelf: Shelf) -> Shelf:
    """Return an equivalent immutable shelf with its resolved state commitment."""
    return Shelf(
        version=shelf.version,
        current_observation=shelf.current_observation,
        hypotheses=shelf.hypotheses,
        experiences=shelf.experiences,
        chain_commitment=shelf.chain_commitment,
        state_commitment=sha256_commitment(state_projection(atlas, shelf)),
    )


def canonical_state(atlas: Atlas, shelf: Shelf) -> CanonicalState:
    projection = state_projection(atlas, shelf)
    projection["state_commitment"] = shelf.state_commitment
    return CanonicalState(readonly_mapping(projection), canonical_json_bytes(projection))


def initial_hypotheses(atlas: Atlas) -> dict[str, HypothesisState]:
    return {
        hypothesis.content_id: HypothesisState(
            id=hypothesis.content_id,
            source_ref=hypothesis.source_ref,
            cause=hypothesis.cause,
            effect=hypothesis.effect,
            relation=hypothesis.relation,
            provenance="atlas",
        )
        for hypothesis in atlas.hypotheses
    }


def make_candidates(atlas: Atlas, hypotheses: Mapping[str, HypothesisState], cause: str) -> dict[str, HypothesisState]:
    result = dict(hypotheses)
    for effect in atlas.observable_ids:
        if effect == cause:
            continue
        hypothesis_id = content_id(cause, effect, "cochanges")
        if hypothesis_id not in result:
            result[hypothesis_id] = HypothesisState(
                id=hypothesis_id,
                source_ref=None,
                cause=cause,
                effect=effect,
                relation="cochanges",
                provenance="experience",
            )
    return result
