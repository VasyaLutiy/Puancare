from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.errors import SnapshotValidationError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, hypothesis_by_edge, observation


class HiddenHypothesisPersistenceRegressionTests(unittest.TestCase):
    def _hidden_atlas(self):
        data = atlas_data(
            variable_count=3,
            hypotheses=[
                {"id": "hidden-cause", "cause": "var-2", "effect": "var-0", "relation": "cochanges", "provenance": "atlas"},
                {"id": "hidden-effect", "cause": "var-0", "effect": "var-2", "relation": "cochanges", "provenance": "atlas"},
            ],
        )
        data["variables"][2]["observable"] = False
        return data

    def _accepted_engine(self):
        data = self._hidden_atlas()
        engine = create(atlas_bytes(data))
        before = observation(data)
        before.pop("var-2")
        engine.bind_observation(before)
        prediction = engine.predict("action-0")
        self.assertEqual({"var-0", "var-1"}, set().union(prediction.changed, prediction.unchanged, prediction.unknown))
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)
        return data, engine

    def test_hidden_endpoint_hypotheses_survive_lifecycle_and_byte_stable_reload(self) -> None:
        data, engine = self._accepted_engine()
        for cause, effect in (("var-2", "var-0"), ("var-0", "var-2")):
            with self.subTest(cause=cause, effect=effect):
                hypothesis = hypothesis_by_edge(engine, cause, effect)
                self.assertEqual("proposed", hypothesis["status"])
                self.assertEqual({"supports": 0, "contradicts": 0}, hypothesis["evidence"])
                self.assertEqual((), hypothesis["supporting_evidence"])
                self.assertEqual((), hypothesis["contradicting_evidence"])

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            rewritten = Path(directory) / "rewritten.json"
            engine.save(source)
            restored = load(source, atlas_bytes(data))
            restored.save(rewritten)
            self.assertEqual(source.read_bytes(), rewritten.read_bytes())

    def test_corrupted_hidden_endpoint_evidence_is_snapshot_error_not_key_error(self) -> None:
        data, engine = self._accepted_engine()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            forged_path = Path(directory) / "forged.json"
            engine.save(source)
            source_bytes = source.read_bytes()
            state = json.loads(source_bytes)
            hidden_edges = (("var-2", "var-0"), ("var-0", "var-2"))
            for cause, effect in hidden_edges:
                for evidence, references, status, field in (
                    ({"supports": 1, "contradicts": 0}, [1], "supported", "supporting_evidence"),
                    ({"supports": 0, "contradicts": 1}, [1], "refuted", "contradicting_evidence"),
                ):
                    with self.subTest(cause=cause, effect=effect, evidence=evidence):
                        forged = copy.deepcopy(state)
                        hypothesis = next(
                            item for item in forged["hypotheses"]
                            if item["cause"] == cause and item["effect"] == effect
                        )
                        hypothesis["evidence"] = evidence
                        hypothesis[field] = references
                        hypothesis["status"] = status
                        forged_path.write_text(json.dumps(forged), encoding="utf-8")
                        with self.assertRaises(SnapshotValidationError) as caught:
                            load(forged_path, atlas_bytes(data))
                        self.assertEqual("SNAPSHOT_INVARIANT", caught.exception.code)
                        self.assertEqual(source_bytes, source.read_bytes())

