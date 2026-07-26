from __future__ import annotations

import unittest

from transition_evidence import create
from transition_evidence.runner import run_schedule

from tests.transition_evidence.support import atlas_bytes, atlas_data, changed_observation, observation


class PredictionGateEnvironment:
    """The adapter refuses a step unless the runner just predicted that action."""

    def __init__(self, atlas, start, responses):
        self.atlas = atlas
        self.current = dict(start)
        self.responses = list(responses)
        self.expected_step: str | None = None

    def observe(self):
        return dict(self.current)

    def action_space(self):
        return [dict(item) for item in self.atlas["actions"]]

    def step(self, action_id):
        if action_id != self.expected_step:
            raise AssertionError(f"step {action_id!r} was not preceded by its prediction")
        self.expected_step = None
        response = self.responses.pop(0)
        self.current = dict(response["obs_after"])
        return response


class PredictionBeforeStepIntegrationTests(unittest.TestCase):
    def test_each_step_requires_its_immediately_preceding_prediction(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        before = observation(data)
        after_one = changed_observation(data, before, {"var-0"})
        after_two = changed_observation(data, after_one, {"var-1"})
        environment = PredictionGateEnvironment(
            data,
            before,
            [
                {"action_id": "action-0", "outcome": "ok", "obs_after": after_one},
                {"action_id": "action-1", "outcome": "ok", "obs_after": after_two},
            ],
        )
        engine = create(atlas_bytes(data))
        predict = engine.predict

        def recording_predict(action_id):
            if environment.expected_step is not None:
                raise AssertionError("runner predicted another action before taking the pending step")
            prediction = predict(action_id)
            environment.expected_step = action_id
            return prediction

        engine.predict = recording_predict
        self.assertEqual((1, 2), run_schedule(engine, environment, ["action-0", "action-1"]))
        self.assertIsNone(environment.expected_step)

