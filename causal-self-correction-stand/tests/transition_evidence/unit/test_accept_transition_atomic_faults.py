from __future__ import annotations

import unittest
from unittest import mock

from transition_evidence import create
from transition_evidence.errors import RuntimeValidationError
from transition_evidence.types import HypothesisState

from tests.transition_evidence.support import atlas_bytes, atlas_data, changed_observation, observation


class AcceptTransitionAtomicFaultRegressionTests(unittest.TestCase):
    """Every pre-commit failure must preserve both shelf and pending prediction."""

    def _pending_transition(self):
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        prediction = engine.predict("action-0")
        response = {
            "action_id": "action-0",
            "outcome": "ok",
            "obs_after": changed_observation(data, before, {"var-0", "var-1"}),
        }
        return engine, prediction, response

    def test_each_precommit_phase_is_atomic_and_retryable(self) -> None:
        phases = (
            (
                "wire-validation",
                lambda engine: mock.patch.object(
                    engine, "_validate_step_response", side_effect=RuntimeError("injected validation")
                ),
            ),
            (
                "candidate-generation",
                lambda engine: mock.patch(
                    "transition_evidence.api.make_candidates", side_effect=RuntimeError("injected candidates")
                ),
            ),
            (
                "relation-evaluator",
                lambda engine: mock.patch.object(
                    engine._relations["cochanges"], "evaluate", side_effect=RuntimeError("injected evaluator")
                ),
            ),
            (
                "status-projection",
                lambda engine: mock.patch.object(
                    HypothesisState, "status", side_effect=RuntimeError("injected status")
                ),
            ),
            (
                "comparison",
                lambda engine: mock.patch.object(
                    engine, "_comparison", side_effect=RuntimeError("injected comparison")
                ),
            ),
            (
                "commitment-sealing",
                lambda engine: mock.patch(
                    "transition_evidence.api.seal_shelf", side_effect=RuntimeError("injected sealing")
                ),
            ),
        )
        for phase, patch_for in phases:
            with self.subTest(phase=phase):
                engine, prediction, response = self._pending_transition()
                before = engine.snapshot().to_json_bytes()
                with patch_for(engine):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        engine.accept_transition(prediction.id, response)
                self.assertEqual(before, engine.snapshot().to_json_bytes())
                self.assertEqual(1, engine.accept_transition(prediction.id, response))

    def test_action_mismatch_and_invalid_after_remain_retryable(self) -> None:
        engine, prediction, response = self._pending_transition()
        before = engine.snapshot().to_json_bytes()
        wrong_action = dict(response, action_id="wrong-action")
        with self.assertRaises(RuntimeValidationError) as caught:
            engine.accept_transition(prediction.id, wrong_action)
        self.assertEqual("PREDICTION_MISMATCH", caught.exception.code)
        self.assertEqual(before, engine.snapshot().to_json_bytes())

        malformed = dict(response, obs_after=dict(response["obs_after"], **{"var-0": "outside-domain"}))
        with self.assertRaises(RuntimeValidationError) as caught:
            engine.accept_transition(prediction.id, malformed)
        self.assertEqual("VALUE_OUT_OF_DOMAIN", caught.exception.code)
        self.assertEqual(before, engine.snapshot().to_json_bytes())
        self.assertEqual(1, engine.accept_transition(prediction.id, response))
