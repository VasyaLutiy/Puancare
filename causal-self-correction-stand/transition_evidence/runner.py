"""Small deterministic runner over the public :class:`EnvironmentPort`."""

from __future__ import annotations

from collections.abc import Sequence

from .api import TransitionEvidenceEngine
from .errors import RuntimeValidationError
from .ports import validate_action_space
from .shelf import validate_observation
from .types import EnvironmentPort


def run_schedule(
    engine: TransitionEvidenceEngine,
    environment: EnvironmentPort,
    schedule: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Run explicit actions, or currently available actions in stable ID order."""

    # The complete preflight is pure: invalid adapter input or a schedule entry
    # must not bind, clear pending predictions, or otherwise mutate the shelf.
    observation = validate_observation(engine._atlas, environment.observe())
    available = validate_action_space(engine._atlas, environment.action_space())
    actions = list(schedule) if schedule is not None else sorted(available)
    for action_id in actions:
        if not isinstance(action_id, str) or action_id not in available:
            raise RuntimeValidationError("ACTION_NOT_AVAILABLE", "/action_id")

    engine.bind_observation(observation)
    experience_ids: list[int] = []
    for action_id in actions:
        prediction = engine.predict(action_id)
        # Any adapter failure occurs before acceptance, leaving shelf untouched.
        response = environment.step(action_id)
        experience_ids.append(engine.accept_transition(prediction.id, response))
    return tuple(experience_ids)
