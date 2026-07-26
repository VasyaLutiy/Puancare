from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transition_evidence import create, load

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class MultiStepLoadNoRetrainingTests(unittest.TestCase):
    def test_valid_multistep_load_never_invokes_causal_or_learning_operations(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after_first = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after_first)
        after_second = changed_observation(data, after_first, {"var-1", "var-2"})
        accept(engine, "action-1", after_second)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            engine.save(source)
            forbidden = AssertionError("load must audit persisted facts, never retrain")
            with (
                patch("transition_evidence.api.TransitionEvidenceEngine.predict", side_effect=forbidden),
                patch("transition_evidence.api.make_candidates", side_effect=forbidden),
                patch("transition_evidence.api.TransitionEvidenceEngine._apply_evidence", side_effect=forbidden),
                patch("transition_evidence.relations.CochangesEvaluator.evaluate", side_effect=forbidden),
            ):
                restored = load(source, atlas_bytes(data))
            self.assertEqual(engine.snapshot().to_json_bytes(), restored.snapshot().to_json_bytes())

