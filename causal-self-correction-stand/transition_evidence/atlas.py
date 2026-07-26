"""Strict, safe parsing and canonical identification of transition atlases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

from .canonical import (
    atlas_digest,
    canonical_json_bytes,
    content_id,
    ensure_json_scalar,
    is_json_scalar,
    pointer_join,
    scalar_token,
)
from .errors import AtlasValidationError
from .relations import default_relation_registry
from .types import Action, Atlas, AtlasHypothesis, Policy, Variable


def _reject_cycles(value: object, path: str = "", active: set[int] | None = None) -> None:
    active = set() if active is None else active
    if not isinstance(value, (dict, list)):
        return
    identity = id(value)
    if identity in active:
        raise AtlasValidationError("CYCLIC_YAML", path)
    active.add(identity)
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_cycles(child, pointer_join(path, key), active)
    else:
        for index, child in enumerate(value):
            _reject_cycles(child, pointer_join(path, index), active)
    active.remove(identity)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AtlasValidationError("SCHEMA_SHAPE", path, "expected an object with string keys")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise AtlasValidationError("SCHEMA_SHAPE", path, "expected an array")
    return value


def _require_exact_fields(
    mapping: Mapping[str, object], path: str, fields: set[str], *, root: bool = False
) -> None:
    for key in mapping:
        if key not in fields:
            code = "INITIAL_STATE_FORBIDDEN" if root and key == "initial_state" else "UNKNOWN_FIELD"
            raise AtlasValidationError(code, pointer_join(path, key))
    missing = fields.difference(mapping)
    if missing:
        raise AtlasValidationError("SCHEMA_SHAPE", pointer_join(path, sorted(missing)[0]), "required field missing")


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise AtlasValidationError("SCHEMA_SHAPE", path, "expected an opaque string")
    return value


def parse_atlas(atlas_bytes: bytes) -> Atlas:
    """Parse exactly one safe ``transition-atlas/v1`` document."""

    try:
        loaded = yaml.safe_load(atlas_bytes)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise AtlasValidationError("UNSAFE_YAML", "", str(exc)) from exc
    _reject_cycles(loaded)
    root = _mapping(loaded, "")
    _require_exact_fields(root, "", {"schema", "variables", "actions", "hypotheses", "policy"}, root=True)
    if root["schema"] != "transition-atlas/v1":
        raise AtlasValidationError("SCHEMA_VERSION", "/schema")

    variables_list = _list(root["variables"], "/variables")
    if len(variables_list) < 2:
        raise AtlasValidationError("SCHEMA_SHAPE", "/variables", "at least two variables are required")
    variables: dict[str, Variable] = {}
    for index, raw_variable in enumerate(variables_list):
        path = f"/variables/{index}"
        variable = _mapping(raw_variable, path)
        _require_exact_fields(variable, path, {"id", "domain", "observable"})
        variable_id = _identifier(variable["id"], f"{path}/id")
        if variable_id in variables:
            raise AtlasValidationError("DUPLICATE_ID", f"{path}/id")
        if not isinstance(variable["observable"], bool):
            raise AtlasValidationError("SCHEMA_SHAPE", f"{path}/observable")
        raw_domain = _list(variable["domain"], f"{path}/domain")
        tokens: set[str] = set()
        domain: list[object] = []
        for domain_index, value in enumerate(raw_domain):
            domain_path = f"{path}/domain/{domain_index}"
            if not is_json_scalar(value):
                raise AtlasValidationError("BAD_DOMAIN", domain_path)
            token = scalar_token(value)
            if token in tokens:
                raise AtlasValidationError("BAD_DOMAIN", domain_path, "domain values must be unique")
            tokens.add(token)
            domain.append(value)
        if len(domain) < 2:
            raise AtlasValidationError("BAD_DOMAIN", f"{path}/domain")
        variables[variable_id] = Variable(
            id=variable_id,
            domain_tokens=frozenset(tokens),
            domain_values=tuple(domain),
            observable=variable["observable"],
        )

    actions_list = _list(root["actions"], "/actions")
    if not actions_list:
        raise AtlasValidationError("SCHEMA_SHAPE", "/actions", "at least one action is required")
    actions: dict[str, Action] = {}
    for index, raw_action in enumerate(actions_list):
        path = f"/actions/{index}"
        action = _mapping(raw_action, path)
        _require_exact_fields(action, path, {"id", "intervention"})
        action_id = _identifier(action["id"], f"{path}/id")
        if action_id in actions:
            raise AtlasValidationError("DUPLICATE_ID", f"{path}/id")
        intervention = _mapping(action["intervention"], f"{path}/intervention")
        _require_exact_fields(intervention, f"{path}/intervention", {"variable", "operation"})
        target = _identifier(intervention["variable"], f"{path}/intervention/variable")
        operation = _identifier(intervention["operation"], f"{path}/intervention/operation")
        if target not in variables or not variables[target].observable:
            raise AtlasValidationError("BAD_ACTION_TARGET", f"{path}/intervention/variable")
        actions[action_id] = Action(id=action_id, target=target, operation=operation)

    relation_registry = default_relation_registry()
    hypotheses_list = _list(root["hypotheses"], "/hypotheses")
    hypotheses: list[AtlasHypothesis] = []
    source_ids: set[str] = set()
    structural_ids: set[str] = set()
    for index, raw_hypothesis in enumerate(hypotheses_list):
        path = f"/hypotheses/{index}"
        hypothesis = _mapping(raw_hypothesis, path)
        _require_exact_fields(hypothesis, path, {"id", "cause", "effect", "relation", "provenance"})
        source_ref = _identifier(hypothesis["id"], f"{path}/id")
        if source_ref in source_ids:
            raise AtlasValidationError("DUPLICATE_ID", f"{path}/id")
        source_ids.add(source_ref)
        cause = _identifier(hypothesis["cause"], f"{path}/cause")
        effect = _identifier(hypothesis["effect"], f"{path}/effect")
        if cause not in variables or effect not in variables or cause == effect:
            raise AtlasValidationError("BAD_HYPOTHESIS_ENDPOINT", path)
        relation = _identifier(hypothesis["relation"], f"{path}/relation")
        if relation not in relation_registry:
            raise AtlasValidationError("UNKNOWN_RELATION", f"{path}/relation")
        if hypothesis["provenance"] != "atlas":
            raise AtlasValidationError("SCHEMA_SHAPE", f"{path}/provenance")
        structural_id = content_id(cause, effect, relation)
        if structural_id in structural_ids:
            raise AtlasValidationError("DUPLICATE_HYPOTHESIS", path)
        structural_ids.add(structural_id)
        hypotheses.append(
            AtlasHypothesis(structural_id, source_ref, cause, effect, relation)
        )

    policy_mapping = _mapping(root["policy"], "/policy")
    allowed_policy_fields = {"support_min", "refute_min", "candidate_generation", "proposed_prediction"}
    for key in policy_mapping:
        if key not in allowed_policy_fields:
            raise AtlasValidationError("UNKNOWN_FIELD", pointer_join("/policy", key))
    required_policy_fields = {"support_min", "refute_min", "candidate_generation"}
    missing_policy_fields = required_policy_fields.difference(policy_mapping)
    if missing_policy_fields:
        raise AtlasValidationError("BAD_POLICY", f"/policy/{sorted(missing_policy_fields)[0]}")
    support_min = policy_mapping["support_min"]
    refute_min = policy_mapping["refute_min"]
    candidate_generation = policy_mapping["candidate_generation"]
    proposed_prediction = policy_mapping.get("proposed_prediction", "unknown")
    if (
        isinstance(support_min, bool)
        or not isinstance(support_min, int)
        or support_min <= 0
        or isinstance(refute_min, bool)
        or not isinstance(refute_min, int)
        or refute_min <= 0
        or candidate_generation != "all_observable_effects"
    ):
        raise AtlasValidationError("BAD_POLICY", "/policy")
    if not isinstance(proposed_prediction, str) or proposed_prediction not in {"changed", "unknown"}:
        raise AtlasValidationError("BAD_POLICY", "/policy/proposed_prediction")
    policy = Policy(support_min, refute_min, candidate_generation, proposed_prediction)

    canonical_atlas = {
        "schema": "transition-atlas/v1",
        "variables": [
            {
                "id": variable.id,
                "domain": sorted(variable.domain_values, key=scalar_token),
                "observable": variable.observable,
            }
            for variable in sorted(variables.values(), key=lambda item: item.id)
        ],
        "actions": [action.as_mapping() for action in sorted(actions.values(), key=lambda item: item.id)],
        "hypotheses": [
            {
                "id": hypothesis.source_ref,
                "cause": hypothesis.cause,
                "effect": hypothesis.effect,
                "relation": hypothesis.relation,
                "provenance": "atlas",
            }
            for hypothesis in sorted(hypotheses, key=lambda item: item.content_id)
        ],
        "policy": {
            "support_min": policy.support_min,
            "refute_min": policy.refute_min,
            "candidate_generation": policy.candidate_generation,
            "proposed_prediction": policy.proposed_prediction,
        },
    }
    return Atlas(
        variables=variables,
        actions=actions,
        hypotheses=tuple(sorted(hypotheses, key=lambda item: item.content_id)),
        policy=policy,
        atlas_id=atlas_digest(canonical_atlas),
    )
