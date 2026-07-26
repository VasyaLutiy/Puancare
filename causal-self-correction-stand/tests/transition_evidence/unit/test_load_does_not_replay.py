from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transition_evidence import create, load

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class DetachedLoadNoReplayRegressionTests(unittest.TestCase):
    def test_load_uses_persisted_evidence_without_replaying_old_transitions(self) -> None:
        """Loading is validation/deserialization, never a second learning pass."""
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            rewritten = Path(directory) / "rewritten.json"
            engine.save(source)
            source_bytes = source.read_bytes()

            forbidden = AssertionError("load must not replay persisted experiences")
            with (
                patch("transition_evidence.api.TransitionEvidenceEngine.predict", side_effect=forbidden),
                patch("transition_evidence.api.make_candidates", side_effect=forbidden),
                patch("transition_evidence.api.TransitionEvidenceEngine._apply_evidence", side_effect=forbidden),
                patch("transition_evidence.relations.CochangesEvaluator.evaluate", side_effect=forbidden),
            ):
                restored = load(source, atlas_bytes(data))

            # A detached load may be re-persisted exactly, without re-evaluation.
            restored.save(rewritten)
            self.assertEqual(source_bytes, rewritten.read_bytes())

