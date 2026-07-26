from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transition_evidence import create, load
from transition_evidence.canonical import canonical_json_bytes, content_id, sha256_commitment
from transition_evidence.errors import SnapshotValidationError, TransitionEvidenceError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class RelationAgnosticLoadRegressionTests(unittest.TestCase):
    def _saved_snapshot(self, directory):
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)
        source = Path(directory) / "source.json"
        engine.save(source)
        return data, before, after, source, source.read_bytes()

    @staticmethod
    def _rehashed_relation_snapshot(raw, relation):
        forged = json.loads(raw)
        hypothesis = next(
            item for item in forged["hypotheses"]
            if item["provenance"] == "experience" and item["cause"] == "var-0" and item["effect"] == "var-1"
        )
        hypothesis["relation"] = relation
        hypothesis["id"] = content_id(hypothesis["cause"], hypothesis["effect"], relation)
        forged["hypotheses"].sort(key=lambda item: item["id"])
        forged["state_commitment"] = sha256_commitment(
            {key: value for key, value in forged.items() if key != "state_commitment"}
        )
        return forged

    def test_coherent_unknown_relation_loads_without_evaluation_or_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, _before, after, _source, raw = self._saved_snapshot(directory)
            forged_path = Path(directory) / "relation.json"
            forged_path.write_bytes(canonical_json_bytes(self._rehashed_relation_snapshot(raw, "opaque/relation~v2")))
            with (
                mock.patch("transition_evidence.api.TransitionEvidenceEngine.predict", side_effect=AssertionError("load must not predict")),
                mock.patch("transition_evidence.api.make_candidates", side_effect=AssertionError("load must not generate")),
                mock.patch("transition_evidence.api.TransitionEvidenceEngine._apply_evidence", side_effect=AssertionError("load must not apply")),
                mock.patch("transition_evidence.relations.CochangesEvaluator.evaluate", side_effect=AssertionError("load must not evaluate")),
            ):
                restored = load(forged_path, atlas_bytes(data))
            rewritten = Path(directory) / "rewritten.json"
            restored.save(rewritten)
            self.assertEqual(forged_path.read_bytes(), rewritten.read_bytes())

            restored.bind_observation(after)
            prediction = restored.predict("action-0")
            baseline = restored.snapshot().to_json_bytes()
            after_retry = changed_observation(data, after, {"var-0"})
            with self.assertRaises(TransitionEvidenceError) as caught:
                restored.accept_transition(prediction.id, {"action_id": "action-0", "outcome": "ok", "obs_after": after_retry})
            self.assertNotIsInstance(caught.exception, KeyError)
            self.assertEqual(baseline, restored.snapshot().to_json_bytes())

    def test_non_string_relation_and_bad_content_id_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, _before, _after, _source, raw = self._saved_snapshot(directory)
            variants = []
            non_string = self._rehashed_relation_snapshot(raw, "cochanges")
            non_string["hypotheses"][0]["relation"] = 7
            variants.append(non_string)
            bad_id = self._rehashed_relation_snapshot(raw, "cochanges")
            bad_id["hypotheses"][0]["id"] = "0" * 64
            variants.append(bad_id)
            for index, forged in enumerate(variants):
                with self.subTest(case=index):
                    path = Path(directory) / f"invalid-{index}.json"
                    path.write_text(json.dumps(forged), encoding="utf-8")
                    with self.assertRaises(SnapshotValidationError):
                        load(path, atlas_bytes(data))


if __name__ == "__main__":
    unittest.main()
