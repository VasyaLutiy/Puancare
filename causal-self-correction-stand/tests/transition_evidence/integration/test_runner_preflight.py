from __future__ import annotations

import unittest
from unittest import mock

from transition_evidence import create
from transition_evidence.errors import RuntimeValidationError
from transition_evidence.ports import validate_action_space
from transition_evidence.runner import run_schedule

from tests.transition_evidence.support import atlas_bytes, atlas_data, changed_observation, observation


class PreflightEnvironment:
    def __init__(self, before, actions):
        self.before = dict(before)
        self.actions = list(actions)
        self.events: list[str] = []
        self.observe_error: Exception | None = None
        self.action_space_error: Exception | None = None

    def observe(self):
        self.events.append("observe")
        if self.observe_error is not None:
            raise self.observe_error
        return dict(self.before)

    def action_space(self):
        self.events.append("action_space")
        if self.action_space_error is not None:
            raise self.action_space_error
        return list(self.actions)

    def step(self, action_id):
        self.events.append(f"step:{action_id}")
        raise AssertionError("preflight tests must not step")


class RunnerPreflightRegressionTests(unittest.TestCase):
    def _pending_engine(self):
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        prediction = engine.predict("action-0")
        after = changed_observation(data, before, {"var-0"})
        response = {"action_id": "action-0", "outcome": "ok", "obs_after": after}
        return data, engine, before, prediction, response

    def _assert_preflight_failure_preserves_pending(self, environment, schedule, expected=RuntimeValidationError):
        data, engine, _before, prediction, response = self._pending_engine()
        snapshot = engine.snapshot().to_json_bytes()
        with self.assertRaises(expected) as caught:
            run_schedule(engine, environment, schedule)
        self.assertEqual(snapshot, engine.snapshot().to_json_bytes())
        self.assertEqual(1, engine.accept_transition(prediction.id, response))
        self.assertFalse(any(event.startswith("step:") for event in environment.events))
        return caught.exception

    def test_observe_errors_and_invalid_observation_are_preflight_atomic(self) -> None:
        data, _engine, before, _prediction, _response = self._pending_engine()
        environment = PreflightEnvironment(before, data["actions"])
        environment.observe_error = RuntimeError("adapter unavailable")
        self._assert_preflight_failure_preserves_pending(environment, [], RuntimeError)

        environment = PreflightEnvironment({"var-0": before["var-0"]}, data["actions"])
        caught = self._assert_preflight_failure_preserves_pending(environment, [])
        self.assertEqual("OBSERVATION_SHAPE", caught.code)

    def test_action_space_and_schedule_preflight_failures_preserve_pending(self) -> None:
        data, _engine, before, _prediction, _response = self._pending_engine()
        valid = data["actions"]
        variants = (
            ("adapter-error", PreflightEnvironment(before, valid), [], RuntimeError),
            ("extra", PreflightEnvironment(before, [dict(valid[0], extra=True)]), [], RuntimeValidationError),
            ("duplicate", PreflightEnvironment(before, [valid[0], valid[0]]), [], RuntimeValidationError),
            ("foreign", PreflightEnvironment(before, [{"id": "foreign", "intervention": valid[0]["intervention"]}]), [], RuntimeValidationError),
            ("schedule", PreflightEnvironment(before, valid), ["foreign"], RuntimeValidationError),
        )
        variants[0][1].action_space_error = RuntimeError("adapter unavailable")
        for label, environment, schedule, error_type in variants:
            with self.subTest(case=label):
                self._assert_preflight_failure_preserves_pending(environment, schedule, error_type)

    def test_escaped_observation_pointer_is_preserved_during_preflight(self) -> None:
        data = atlas_data(variable_count=2)
        data["variables"][0]["id"] = "v/a~b"
        data["actions"][0]["id"] = "act/a~b"
        data["actions"][0]["intervention"]["variable"] = "v/a~b"
        before = observation(data)
        engine = create(atlas_bytes(data))
        engine.bind_observation(before)
        prediction = engine.predict("act/a~b")
        snapshot = engine.snapshot().to_json_bytes()
        invalid = dict(before)
        del invalid["v/a~b"]
        environment = PreflightEnvironment(invalid, data["actions"])
        with self.assertRaises(RuntimeValidationError) as caught:
            run_schedule(engine, environment, [])
        self.assertEqual("/v~1a~0b", caught.exception.path)
        self.assertEqual(snapshot, engine.snapshot().to_json_bytes())
        after = changed_observation(data, before, {"v/a~b"})
        self.assertEqual(1, engine.accept_transition(prediction.id, {"action_id": "act/a~b", "outcome": "ok", "obs_after": after}))

    def test_empty_schedule_validates_then_binds_after_all_preflight_work(self) -> None:
        data = atlas_data(variable_count=2)
        before = observation(data)
        engine = create(atlas_bytes(data))
        environment = PreflightEnvironment(before, data["actions"])
        events = environment.events
        original_bind = engine.bind_observation

        def tracked_bind(value):
            events.append("bind")
            return original_bind(value)

        def tracked_validation(atlas, action_space):
            events.append("validation")
            return validate_action_space(atlas, action_space)

        engine.bind_observation = tracked_bind
        with mock.patch("transition_evidence.runner.validate_action_space", side_effect=tracked_validation):
            self.assertEqual((), run_schedule(engine, environment, []))
        self.assertEqual(["observe", "action_space", "validation", "bind"], events)


if __name__ == "__main__":
    unittest.main()
