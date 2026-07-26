"""Environment-port helpers and strict action-space boundary validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .errors import RuntimeValidationError
from .types import ActionDescription, Atlas


def validate_action_space(atlas: Atlas, action_space: object) -> dict[str, ActionDescription]:
    if not isinstance(action_space, Sequence) or isinstance(action_space, (str, bytes)):
        raise RuntimeValidationError("ACTION_NOT_AVAILABLE", "/action_space")
    available: dict[str, ActionDescription] = {}
    for index, raw_action in enumerate(action_space):
        path = f"/action_space/{index}"
        if not isinstance(raw_action, Mapping) or set(raw_action) != {"id", "intervention"}:
            raise RuntimeValidationError("ACTION_NOT_AVAILABLE", path)
        action_id = raw_action["id"]
        intervention = raw_action["intervention"]
        if not isinstance(action_id, str) or not isinstance(intervention, Mapping) or set(intervention) != {"variable", "operation"}:
            raise RuntimeValidationError("ACTION_NOT_AVAILABLE", path)
        action = atlas.actions.get(action_id)
        if action is None or raw_action != action.as_mapping() or action_id in available:
            raise RuntimeValidationError("ACTION_NOT_AVAILABLE", path)
        available[action_id] = action.as_mapping()  # type: ignore[assignment]
    return available
