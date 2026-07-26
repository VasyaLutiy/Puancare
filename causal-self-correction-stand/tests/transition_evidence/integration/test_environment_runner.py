from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create
from transition_evidence.runner import run_schedule

from tests.transition_evidence.support import atlas_bytes, atlas_data, changed_observation, observation, snapshot_mapping


class FakeEnvironment:
    """A port-only fake: no engine internals enter the adapter boundary."""

    def __init__(self, atlas: dict, start: dict, responses: list[dict]):
        self._atlas = atlas
        self._observation = dict(start)
        self._responses = list(responses)
        self.calls: list[str] = []

    def observe(self):
        self.calls.append("observe")
        return dict(self._observation)

    def action_space(self):
        self.calls.append("action_space")
        return [dict(action) for action in self._atlas["actions"]]

    def step(self, action_id):
        self.calls.append(f"step:{action_id}")
        response = self._responses.pop(0)
        assert response["action_id"] == action_id
        self._observation = dict(response["obs_after"])
        return response


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_predicts_before_each_step_for_multiple_targets_and_mixed_outcomes(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        start = observation(data)
        after_a = changed_observation(data, start, {"var-0", "var-2"})
        after_b = dict(after_a)
        after_a_again = changed_observation(data, after_b, {"var-0"})
        environment = FakeEnvironment(
            data,
            start,
            [
                {"action_id": "action-0", "outcome": True, "obs_after": after_a},
                {"action_id": "action-1", "outcome": "failed", "obs_after": after_b},
                {"action_id": "action-0", "outcome": 0, "obs_after": after_a_again},
            ],
        )
        engine = create(atlas_bytes(data))
        calls: list[str] = []
        original_predict = engine.predict

        def tracked_predict(action_id):
            calls.append(f"predict:{action_id}")
            return original_predict(action_id)

        engine.predict = tracked_predict
        ids = run_schedule(engine, environment, ["action-0", "action-1", "action-0"])
        self.assertEqual((1, 2, 3), ids)
        self.assertEqual(
            ["observe", "action_space", "step:action-0", "step:action-1", "step:action-0"],
            environment.calls,
        )
        self.assertEqual(["predict:action-0", "predict:action-1", "predict:action-0"], calls)
        self.assertEqual(3, len(snapshot_mapping(engine)["experiences"]))

    def test_subprocess_restart_loads_then_rebinds_a_new_environment_observation(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        start = observation(data)
        engine = create(atlas_bytes(data))
        engine.bind_observation(start)
        after = changed_observation(data, start, {"var-0", "var-1"})
        prediction = engine.predict("action-0")
        engine.accept_transition(prediction.id, {"action_id": "action-0", "outcome": "ok", "obs_after": after})
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            atlas_path = directory_path / "atlas.yaml"
            snapshot_path = directory_path / "snapshot.json"
            atlas_path.write_bytes(atlas_bytes(data))
            engine.save(snapshot_path)
            script = """
from pathlib import Path
from transition_evidence import load
from transition_evidence.errors import TransitionEvidenceError
atlas = Path(__import__('sys').argv[1]).read_bytes()
engine = load(__import__('sys').argv[2], atlas)
try:
    engine.predict('action-0')
except TransitionEvidenceError as error:
    assert error.code == 'OBSERVATION_UNBOUND'
engine.bind_observation({'var-0': 'value-0-1', 'var-1': 'value-1-1', 'var-2': 'value-2-0'})
prediction = engine.predict('action-1')
experience = engine.accept_transition(prediction.id, {
    'action_id': 'action-1', 'outcome': False,
    'obs_after': {'var-0': 'value-0-1', 'var-1': 'value-1-0', 'var-2': 'value-2-0'},
})
print(experience)
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, str(atlas_path), str(snapshot_path)],
                cwd=Path(__file__).parents[3],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual("2", completed.stdout.strip())

