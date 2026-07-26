from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transition_evidence import create, load
from transition_evidence.errors import TransitionEvidenceError

from tests.transition_evidence.support import (
    accept,
    atlas_bytes,
    atlas_data,
    canonical_content_id,
    changed_observation,
    hypothesis_by_edge,
    observation,
    snapshot_mapping,
    three_handwritten_atlases,
)


class SemanticsTests(unittest.TestCase):
    def _engine(self, **kwargs):
        data = atlas_data(**kwargs)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        return data, engine, before

    def test_three_handwritten_structurally_distinct_atlases_construct(self) -> None:
        states = [create(atlas_bytes(data)).snapshot().to_json_bytes() for data in three_handwritten_atlases()]
        self.assertEqual(3, len(set(states)))

    def test_cochanges_support_contradiction_and_noninformative(self) -> None:
        data, engine, before = self._engine(variable_count=3)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)
        supported = hypothesis_by_edge(engine, "var-0", "var-1")
        contradicted = hypothesis_by_edge(engine, "var-0", "var-2")
        self.assertEqual({"supports": 1, "contradicts": 0}, supported["evidence"])
        self.assertEqual({"supports": 0, "contradicts": 1}, contradicted["evidence"])

        # The selected action did not alter its target: no evidence changes at all.
        unchanged_target = dict(after)
        unchanged_target["var-1"] = before["var-1"]
        accept(engine, "action-0", unchanged_target, outcome="machine-declined")
        self.assertEqual({"supports": 1, "contradicts": 0}, hypothesis_by_edge(engine, "var-0", "var-1")["evidence"])
        self.assertEqual({"supports": 0, "contradicts": 1}, hypothesis_by_edge(engine, "var-0", "var-2")["evidence"])
        state = snapshot_mapping(engine)
        self.assertEqual(2, len(state["experiences"]))
        self.assertEqual(2, state["version"])

    def test_thresholds_one_two_and_three_and_refuted_retention(self) -> None:
        for threshold in (1, 2, 3):
            with self.subTest(threshold=threshold):
                data, engine, current = self._engine(variable_count=3, support_min=threshold, refute_min=threshold)
                for _ in range(threshold):
                    next_observation = changed_observation(data, current, {"var-0", "var-1"})
                    accept(engine, "action-0", next_observation)
                    current = next_observation
                self.assertEqual("supported", hypothesis_by_edge(engine, "var-0", "var-1")["status"])
                self.assertEqual("refuted", hypothesis_by_edge(engine, "var-0", "var-2")["status"])
                refuted = hypothesis_by_edge(engine, "var-0", "var-2")
                self.assertEqual(threshold, refuted["evidence"]["contradicts"])
                self.assertEqual(threshold, len(refuted["contradicting_evidence"]))
                self.assertIn(refuted, snapshot_mapping(engine)["hypotheses"])

    def test_candidate_birth_is_atomic_and_reuses_atlas_hypothesis(self) -> None:
        initial = [{"id": "source-local-id", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"}]
        data, engine, before = self._engine(variable_count=4, hypotheses=initial)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)
        hypotheses = [item for item in snapshot_mapping(engine)["hypotheses"] if item["cause"] == "var-0"]
        self.assertEqual(3, len(hypotheses))
        self.assertEqual(3, len({item["id"] for item in hypotheses}))
        reused = hypothesis_by_edge(engine, "var-0", "var-1")
        self.assertEqual("source-local-id", reused["source_ref"])
        self.assertEqual({"supports": 1, "contradicts": 0}, reused["evidence"])
        self.assertEqual(1, snapshot_mapping(engine)["version"])

    def test_prediction_is_immutable_disjoint_complete_partition(self) -> None:
        data, engine, current = self._engine(variable_count=4, support_min=2, refute_min=2)
        # b reaches support=2; c reaches refutation; d has conflicting counts and remains proposed.
        first = changed_observation(data, current, {"var-0", "var-1", "var-3"})
        accept(engine, "action-0", first)
        second = changed_observation(data, first, {"var-0", "var-1"})
        accept(engine, "action-0", second)
        prediction = engine.predict("action-0")
        self.assertEqual(frozenset({"var-0", "var-1"}), prediction.changed)
        self.assertEqual(frozenset({"var-2"}), prediction.unchanged)
        self.assertEqual(frozenset({"var-3"}), prediction.unknown)
        self.assertFalse(prediction.changed & prediction.unchanged)
        self.assertFalse(prediction.changed & prediction.unknown)
        self.assertFalse(prediction.unchanged & prediction.unknown)
        self.assertEqual({"var-0", "var-1", "var-2", "var-3"}, set().union(prediction.changed, prediction.unchanged, prediction.unknown))
        with self.assertRaises(AttributeError):
            prediction.changed.add("var-2")

    def test_content_id_and_atlas_id_are_canonical(self) -> None:
        initial = [{"id": "different-local-name", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"}]
        data, engine, _ = self._engine(hypotheses=initial)
        item = hypothesis_by_edge(engine, "var-0", "var-1")
        self.assertEqual(canonical_content_id("var-0", "var-1"), item["id"])
        permuted = json.loads(json.dumps(data))
        permuted["variables"].reverse()
        permuted["actions"].reverse()
        permuted["hypotheses"].reverse()
        other = create(atlas_bytes(permuted))
        other.bind_observation(observation(permuted))
        self.assertEqual(snapshot_mapping(engine)["atlas_id"], snapshot_mapping(other)["atlas_id"])
        self.assertEqual(engine.snapshot().to_json_bytes(), other.snapshot().to_json_bytes())


class PersistenceTests(unittest.TestCase):
    def test_reload_is_stale_but_snapshot_is_byte_stable(self) -> None:
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            engine.save(first)
            restored = load(first, atlas_bytes(data))
            with self.assertRaises(TransitionEvidenceError) as caught:
                restored.predict("action-0")
            self.assertEqual("OBSERVATION_UNBOUND", caught.exception.code)
            restored.save(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            restored_again = load(second, atlas_bytes(data))
            self.assertEqual(second.read_bytes(), restored_again.snapshot().to_json_bytes())

    def test_failed_atomic_write_preserves_last_valid_file(self) -> None:
        data = atlas_data()
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            engine.save(path)
            last_valid = path.read_bytes()
            after = changed_observation(data, before, {"var-0"})
            accept(engine, "action-0", after)
            with patch("transition_evidence.persistence.os.replace", side_effect=OSError("simulated replacement failure")):
                with self.assertRaises(TransitionEvidenceError) as caught:
                    engine.save(path)
            self.assertEqual("PERSISTENCE_WRITE", caught.exception.code)
            self.assertEqual(last_valid, path.read_bytes())
            self.assertEqual(last_valid, load(path, atlas_bytes(data)).snapshot().to_json_bytes())

