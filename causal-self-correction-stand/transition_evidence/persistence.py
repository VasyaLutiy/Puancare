"""Snapshot integrity verification and crash-safe atomic replacement writes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
import tempfile
from typing import Any

from .canonical import canonical_json_bytes, content_id, genesis_chain_commitment, genesis_entry_commitment, is_json_scalar, post_chain_commitment, sha256_commitment
from .errors import PersistenceError, RuntimeValidationError, SnapshotValidationError
from .shelf import state_projection, validate_observation
from .types import Atlas, Experience, HypothesisState, JsonScalar, Shelf, SnapshotPath


_SHELF_FIELDS = {"schema", "version", "atlas_id", "current_observation", "hypotheses", "experiences", "chain_commitment", "state_commitment"}
_HYPOTHESIS_FIELDS = {"id", "source_ref", "cause", "effect", "relation", "provenance", "status", "evidence", "supporting_evidence", "contradicting_evidence"}
_EXPERIENCE_FIELDS = {"id", "before", "prediction", "action", "outcome", "after", "informative", "comparison", "prediction_commitment", "previous_entry_commitment", "entry_commitment", "pre_chain_commitment", "post_chain_commitment"}
_PREDICTION_FIELDS = {"id", "action_id", "changed", "unchanged", "unknown"}
_COMPARISON_FIELDS = {"actual_changed", "correct_changed", "correct_unchanged", "unknown"}


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)
    return value


def _exact(mapping: Mapping[str, Any], fields: set[str], path: str) -> None:
    if set(mapping) != fields:
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)
    return value


def _hex(value: object, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)
    return value


def _snapshot_observation(atlas: Atlas, value: object, path: str) -> dict[str, JsonScalar]:
    try:
        return validate_observation(atlas, value)
    except RuntimeValidationError as exc:
        raise SnapshotValidationError("SNAPSHOT_SHAPE", f"{path}{exc.path}") from exc


def _id_list(value: object, path: str, known: set[int]) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)
    result = tuple(_integer(item, f"{path}/{index}") for index, item in enumerate(value))
    if result != tuple(sorted(result)) or len(set(result)) != len(result) or any(item not in known for item in result):
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
    return result


def _id_set(value: object, path: str, observable: set[str]) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", path)
    if value != sorted(value) or len(set(value)) != len(value) or not set(value).issubset(observable):
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
    return frozenset(value)


def _prediction_shape(atlas: Atlas, value: object, path: str, version: int,
                      before: Mapping[str, JsonScalar], action_id: str) -> Mapping[str, object]:
    prediction = _mapping(value, path)
    _exact(prediction, _PREDICTION_FIELDS, path)
    prediction_id = _hex(prediction["id"], f"{path}/id")
    if prediction["action_id"] != action_id:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/action_id")
    expected_id = sha256_commitment({"version": version, "before": dict(before), "action_id": action_id})
    if prediction_id != expected_id:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/id")
    observable = set(atlas.observable_ids)
    changed = _id_set(prediction["changed"], f"{path}/changed", observable)
    unchanged = _id_set(prediction["unchanged"], f"{path}/unchanged", observable)
    unknown = _id_set(prediction["unknown"], f"{path}/unknown", observable)
    if changed & unchanged or changed & unknown or unchanged & unknown or changed | unchanged | unknown != observable:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
    if atlas.actions[action_id].target not in changed:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/changed")
    return prediction


def _comparison_shape(value: object, path: str, observable: set[str]) -> Mapping[str, object]:
    comparison = _mapping(value, path)
    _exact(comparison, _COMPARISON_FIELDS, path)
    for key in _COMPARISON_FIELDS:
        _id_set(comparison[key], f"{path}/{key}", observable)
    return comparison


def _experience_commitment_payload(experience: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": experience["id"], "before": experience["before"], "prediction": experience["prediction"],
        "action": experience["action"], "outcome": experience["outcome"], "after": experience["after"],
        "informative": experience["informative"], "comparison": experience["comparison"],
        "prediction_commitment": experience["prediction_commitment"],
        "previous_entry_commitment": experience["previous_entry_commitment"],
        "pre_chain_commitment": experience["pre_chain_commitment"],
    }


def _status(atlas: Atlas, supports: int, contradicts: int) -> str:
    if contradicts >= atlas.policy.refute_min:
        return "refuted"
    if supports >= atlas.policy.support_min and contradicts == 0:
        return "supported"
    return "proposed"


def _hypotheses(
    atlas: Atlas, raw: object, experience_transitions: Mapping[int, tuple[str, bool]]
) -> dict[str, HypothesisState]:
    if not isinstance(raw, list):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", "/hypotheses")
    source_by_id = {hypothesis.content_id: hypothesis for hypothesis in atlas.hypotheses}
    result: dict[str, HypothesisState] = {}
    previous_id = ""
    for index, raw_item in enumerate(raw):
        path = f"/hypotheses/{index}"
        item = _mapping(raw_item, path)
        _exact(item, _HYPOTHESIS_FIELDS, path)
        identifier, cause, effect, relation, source_ref = item["id"], item["cause"], item["effect"], item["relation"], item["source_ref"]
        if (
            not isinstance(identifier, str) or not isinstance(cause, str) or not isinstance(effect, str)
            or not isinstance(relation, str) or (source_ref is not None and not isinstance(source_ref, str))
            or identifier != content_id(cause, effect, relation)
            or cause not in atlas.variables or effect not in atlas.variables or cause == effect
            or identifier in result or (previous_id and identifier <= previous_id)
        ):
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
        previous_id = identifier
        source = source_by_id.get(identifier)
        if source is not None:
            if item["provenance"] != "atlas" or source_ref != source.source_ref:
                raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
        elif item["provenance"] != "experience" or source_ref is not None or cause not in atlas.observable_ids or effect not in atlas.observable_ids:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
        evidence = _mapping(item["evidence"], f"{path}/evidence")
        _exact(evidence, {"supports", "contradicts"}, f"{path}/evidence")
        supports = _integer(evidence["supports"], f"{path}/evidence/supports")
        contradicts = _integer(evidence["contradicts"], f"{path}/evidence/contradicts")
        if supports < 0 or contradicts < 0:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/evidence")
        supporting = _id_list(item["supporting_evidence"], f"{path}/supporting_evidence", set(experience_transitions))
        contradicting = _id_list(item["contradicting_evidence"], f"{path}/contradicting_evidence", set(experience_transitions))
        if (
            supports != len(supporting)
            or contradicts != len(contradicting)
            or set(supporting) & set(contradicting)
            or any(
                experience_transitions[experience_id][0] != cause
                or not experience_transitions[experience_id][1]
                for experience_id in (*supporting, *contradicting)
            )
        ):
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
        if item["status"] != _status(atlas, supports, contradicts):
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/status")
        if (cause not in atlas.observable_ids or effect not in atlas.observable_ids) and (supports or contradicts or supporting or contradicting or item["status"] != "proposed"):
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", path)
        result[identifier] = HypothesisState(identifier, source_ref, cause, effect, relation, item["provenance"], supports, contradicts, supporting, contradicting)
    return result


def load_shelf(snapshot_path: SnapshotPath, atlas: Atlas) -> Shelf:
    """Verify canonical committed data only; no causal/evaluator replay occurs."""
    try:
        raw_bytes = Path(snapshot_path).read_bytes()
        root_value = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError("SNAPSHOT_SHAPE", "", str(exc)) from exc
    root = _mapping(root_value, "")
    _exact(root, _SHELF_FIELDS, "")
    if root["schema"] != "transition-shelf/v1":
        raise SnapshotValidationError("SNAPSHOT_SHAPE", "/schema")
    if root["atlas_id"] != atlas.atlas_id:
        raise SnapshotValidationError("ATLAS_MISMATCH", "/atlas_id")
    version = _integer(root["version"], "/version")
    if version < 0:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", "/version")
    chain_commitment = _hex(root["chain_commitment"], "/chain_commitment")
    state_commitment = _hex(root["state_commitment"], "/state_commitment")
    current = None if root["current_observation"] is None else _snapshot_observation(atlas, root["current_observation"], "/current_observation")
    raw_experiences = root["experiences"]
    if not isinstance(raw_experiences, list):
        raise SnapshotValidationError("SNAPSHOT_SHAPE", "/experiences")
    seen_ids: set[int] = set()
    for index, raw_experience in enumerate(raw_experiences):
        candidate_id = _integer(_mapping(raw_experience, f"/experiences/{index}").get("id"), f"/experiences/{index}/id")
        if candidate_id in seen_ids:
            raise SnapshotValidationError("DUPLICATE_EXPERIENCE", f"/experiences/{index}/id")
        seen_ids.add(candidate_id)
    if version != len(raw_experiences):
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", "/version")

    experiences: list[Experience] = []
    experience_transitions: dict[int, tuple[str, bool]] = {}
    expected_pre_chain = genesis_chain_commitment(atlas.atlas_id)
    previous_entry = genesis_entry_commitment(atlas.atlas_id)
    observable = set(atlas.observable_ids)
    for index, raw_experience in enumerate(raw_experiences):
        path = f"/experiences/{index}"
        item = _mapping(raw_experience, path)
        _exact(item, _EXPERIENCE_FIELDS, path)
        experience_id = _integer(item["id"], f"{path}/id")
        if experience_id != index + 1:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/id")
        before = _snapshot_observation(atlas, item["before"], f"{path}/before")
        after = _snapshot_observation(atlas, item["after"], f"{path}/after")
        action = _mapping(item["action"], f"{path}/action")
        action_id = action.get("id")
        if not isinstance(action_id, str) or action_id not in atlas.actions or action != atlas.actions[action_id].as_mapping():
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/action")
        if not is_json_scalar(item["outcome"]):
            raise SnapshotValidationError("SNAPSHOT_SHAPE", f"{path}/outcome")
        if not isinstance(item["informative"], bool):
            raise SnapshotValidationError("SNAPSHOT_SHAPE", f"{path}/informative")
        prediction = _prediction_shape(atlas, item["prediction"], f"{path}/prediction", index, before, action_id)
        comparison = _comparison_shape(item["comparison"], f"{path}/comparison", observable)
        prediction_commitment = _hex(item["prediction_commitment"], f"{path}/prediction_commitment")
        expected_prediction_commitment = sha256_commitment({
            "atlas_id": atlas.atlas_id, "pre_chain_commitment": expected_pre_chain,
            "prediction_id": prediction["id"], "before": before, "action_id": action_id,
            "changed": prediction["changed"], "unchanged": prediction["unchanged"], "unknown": prediction["unknown"],
        })
        if prediction_commitment != expected_prediction_commitment:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/prediction_commitment")
        if _hex(item["pre_chain_commitment"], f"{path}/pre_chain_commitment") != expected_pre_chain:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/pre_chain_commitment")
        if _hex(item["previous_entry_commitment"], f"{path}/previous_entry_commitment") != previous_entry:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/previous_entry_commitment")
        entry_commitment = _hex(item["entry_commitment"], f"{path}/entry_commitment")
        if entry_commitment != sha256_commitment(_experience_commitment_payload(item)):
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/entry_commitment")
        expected_post_chain = post_chain_commitment(expected_pre_chain, entry_commitment)
        if _hex(item["post_chain_commitment"], f"{path}/post_chain_commitment") != expected_post_chain:
            raise SnapshotValidationError("SNAPSHOT_INVARIANT", f"{path}/post_chain_commitment")
        experiences.append(Experience(experience_id, before, prediction, action, item["outcome"], after,
                                      item["informative"], comparison, prediction_commitment, previous_entry,
                                      entry_commitment, expected_pre_chain, expected_post_chain))
        action_target = atlas.actions[action_id].target
        experience_transitions[experience_id] = (action_target, item["informative"])
        previous_entry, expected_pre_chain = entry_commitment, expected_post_chain
    if chain_commitment != expected_pre_chain:
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", "/chain_commitment")
    hypotheses = _hypotheses(atlas, root["hypotheses"], experience_transitions)
    shelf = Shelf(version, current, hypotheses, tuple(experiences), chain_commitment, state_commitment)
    if state_commitment != sha256_commitment(state_projection(atlas, shelf)):
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", "/state_commitment")
    projection = state_projection(atlas, shelf)
    projection["state_commitment"] = state_commitment
    if raw_bytes != canonical_json_bytes(projection):
        raise SnapshotValidationError("SNAPSHOT_INVARIANT", "")
    return shelf


def save_state(snapshot_path: SnapshotPath, payload: bytes) -> None:
    destination = Path(snapshot_path)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as exc:
        raise PersistenceError("PERSISTENCE_WRITE", str(destination), str(exc)) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
