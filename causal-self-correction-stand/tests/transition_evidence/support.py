from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import yaml


def atlas_data(
    *,
    variable_count: int = 3,
    action_count: int = 1,
    domain_size: int = 2,
    hypotheses: list[dict[str, Any]] | None = None,
    support_min: int = 1,
    refute_min: int = 1,
) -> dict[str, Any]:
    """A deliberately generic atlas, with opaque IDs and values."""
    variables = [
        {"id": f"var-{index}", "domain": [f"value-{index}-{n}" for n in range(domain_size)], "observable": True}
        for index in range(variable_count)
    ]
    actions = [
        {
            "id": f"action-{index}",
            "intervention": {"variable": f"var-{index % variable_count}", "operation": f"opaque-op-{index}"},
        }
        for index in range(action_count)
    ]
    return {
        "schema": "transition-atlas/v1",
        "variables": variables,
        "actions": actions,
        "hypotheses": hypotheses or [],
        "policy": {
            "support_min": support_min,
            "refute_min": refute_min,
            "candidate_generation": "all_observable_effects",
        },
    }


def atlas_bytes(data: dict[str, Any]) -> bytes:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")


def observation(data: dict[str, Any], value_index: int = 0) -> dict[str, Any]:
    return {
        variable["id"]: variable["domain"][value_index]
        for variable in data["variables"]
    }


def changed_observation(
    data: dict[str, Any], before: dict[str, Any], changed: set[str]
) -> dict[str, Any]:
    result = dict(before)
    domains = {variable["id"]: variable["domain"] for variable in data["variables"]}
    for variable_id in changed:
        domain = domains[variable_id]
        result[variable_id] = next(value for value in domain if value != before[variable_id])
    return result


def accept(engine: Any, action_id: str, after: dict[str, Any], outcome: Any = "ok") -> int:
    prediction = engine.predict(action_id)
    return engine.accept_transition(
        prediction.id,
        {"action_id": action_id, "outcome": outcome, "obs_after": after},
    )


def canonical_content_id(cause: str, effect: str, relation: str = "cochanges") -> str:
    raw = json.dumps([cause, effect, relation], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def snapshot_mapping(engine: Any) -> dict[str, Any]:
    return dict(engine.snapshot().as_mapping())


def hypothesis_by_edge(engine: Any, cause: str, effect: str) -> dict[str, Any]:
    for hypothesis in snapshot_mapping(engine)["hypotheses"]:
        if hypothesis["cause"] == cause and hypothesis["effect"] == effect:
            return hypothesis
    raise AssertionError(f"missing hypothesis {cause!r} -> {effect!r}")


def three_handwritten_atlases() -> list[dict[str, Any]]:
    pair = atlas_data(variable_count=2, action_count=1, domain_size=2)
    triangle = atlas_data(
        variable_count=3,
        action_count=2,
        domain_size=3,
        hypotheses=[
            {"id": "local-source", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"}
        ],
        support_min=2,
        refute_min=3,
    )
    square = atlas_data(variable_count=4, action_count=3, domain_size=4)
    square["variables"][2]["observable"] = False
    # A non-observable variable is valid, but must never be selected as an action target.
    square["actions"][2]["intervention"]["variable"] = "var-3"
    return [pair, triangle, square]


def reverse_order(value: Any) -> Any:
    """Reverse mappings and collections without changing their logical contents."""
    if isinstance(value, dict):
        return {key: reverse_order(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reverse_order(item) for item in reversed(value)]
    return value


def renamed_atlas_and_observation(
    source: dict[str, Any], before: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[Any, Any], dict[str, str]]:
    """Produce a bijective rename, including source IDs and scalar values."""
    data = deepcopy(source)
    variable_map = {item["id"]: f"opaque-variable-{index}!" for index, item in enumerate(source["variables"])}
    action_map = {item["id"]: f"opaque-action-{index}!" for index, item in enumerate(source["actions"])}
    source_map = {item["id"]: f"opaque-source-{index}!" for index, item in enumerate(source["hypotheses"])}
    value_map: dict[Any, Any] = {}
    for item in source["variables"]:
        for index, value in enumerate(item["domain"]):
            value_map[value] = f"opaque-value-{item['id']}-{index}!"
    for item in data["variables"]:
        old_id = item["id"]
        item["id"] = variable_map[old_id]
        item["domain"] = [value_map[value] for value in item["domain"]]
    for item in data["actions"]:
        item["id"] = action_map[item["id"]]
        item["intervention"]["variable"] = variable_map[item["intervention"]["variable"]]
    for item in data["hypotheses"]:
        item["id"] = source_map[item["id"]]
        item["cause"] = variable_map[item["cause"]]
        item["effect"] = variable_map[item["effect"]]
    renamed_before = {variable_map[key]: value_map[value] for key, value in before.items()}
    return data, renamed_before, variable_map, value_map, action_map
