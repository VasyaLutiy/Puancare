from __future__ import annotations

import copy
import json
import math
from unittest import mock
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.canonical import canonical_json_bytes, sha256_commitment
from transition_evidence.errors import SnapshotValidationError, TransitionEvidenceError

from tests.transition_evidence.support import (
    accept,
    atlas_bytes,
    atlas_data,
    changed_observation,
    hypothesis_by_edge,
    observation,
    snapshot_mapping,
)


class OutcomeScalarRegressionTests(unittest.TestCase):
    def _bound_engine(self):
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        return data, engine, before

    def test_each_json_scalar_outcome_is_accepted(self) -> None:
        for outcome in (None, False, 17, 1.25, "opaque-machine-result"):
            with self.subTest(outcome=repr(outcome)):
                data, engine, before = self._bound_engine()
                after = changed_observation(data, before, {"var-0"})
                prediction = engine.predict("action-0")
                self.assertEqual(
                    1,
                    engine.accept_transition(
                        prediction.id,
                        {"action_id": "action-0", "outcome": outcome, "obs_after": after},
                    ),
                )
                self.assertEqual(outcome, snapshot_mapping(engine)["experiences"][0]["outcome"])

    def test_list_object_and_nan_outcomes_are_rejected_atomically(self) -> None:
        for outcome in (["not", "scalar"], {"not": "scalar"}, math.nan):
            with self.subTest(outcome=repr(outcome)):
                data, engine, before = self._bound_engine()
                after = changed_observation(data, before, {"var-0"})
                prediction = engine.predict("action-0")
                unchanged = engine.snapshot().to_json_bytes()
                with self.assertRaises(TransitionEvidenceError) as caught:
                    engine.accept_transition(
                        prediction.id,
                        {"action_id": "action-0", "outcome": outcome, "obs_after": after},
                    )
                self.assertEqual("OBSERVATION_SHAPE", caught.exception.code)
                self.assertEqual(unchanged, engine.snapshot().to_json_bytes())
                # The still-pending prediction proves the rejected response did not partially commit.
                self.assertEqual(
                    1,
                    engine.accept_transition(
                        prediction.id,
                        {"action_id": "action-0", "outcome": None, "obs_after": after},
                    ),
                )


class SnapshotForgeryRegressionTests(unittest.TestCase):
    def _saved_informative_snapshot(self, directory: str):
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after, outcome="recorded")
        source = Path(directory) / "source.json"
        engine.save(source)
        raw = source.read_bytes()
        return data, source, raw, json.loads(raw)

    def _assert_forgery_rejected(self, data, source: Path, raw: bytes, forged: dict) -> None:
        forged_path = source.with_name("forged.json")
        forged_path.write_text(json.dumps(forged, allow_nan=True), encoding="utf-8")
        with self.assertRaises(TransitionEvidenceError) as caught:
            load(forged_path, atlas_bytes(data))
        self.assertIn(caught.exception.code, {"SNAPSHOT_SHAPE", "SNAPSHOT_INVARIANT"})
        self.assertEqual(raw, source.read_bytes(), "load must never mutate the source snapshot")

    def test_load_rejects_forged_action_id_and_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._saved_informative_snapshot(directory)
            variants = []
            forged = copy.deepcopy(state)
            forged["experiences"][0]["action"]["id"] = "forged-action"
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["action"]["intervention"]["variable"] = "var-2"
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["action"]["intervention"]["operation"] = "forged-operation"
            variants.append(forged)
            for forged in variants:
                with self.subTest(action=forged["experiences"][0]["action"]):
                    self._assert_forgery_rejected(data, source, raw, forged)

    def test_load_rejects_malformed_or_extra_prediction_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._saved_informative_snapshot(directory)
            variants = []
            forged = copy.deepcopy(state)
            del forged["experiences"][0]["prediction"]["changed"]
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["prediction"]["unexpected"] = []
            variants.append(forged)
            forged = copy.deepcopy(state)
            del forged["experiences"][0]["comparison"]["actual_changed"]
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["comparison"]["unexpected"] = []
            variants.append(forged)
            for forged in variants:
                self._assert_forgery_rejected(data, source, raw, forged)

    def test_load_rejects_non_scalar_outcome_and_invalid_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._saved_informative_snapshot(directory)
            variants = []
            forged = copy.deepcopy(state)
            forged["experiences"][0]["outcome"] = {"not": "a scalar"}
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["before"]["var-0"] = "outside-domain"
            variants.append(forged)
            forged = copy.deepcopy(state)
            forged["current_observation"]["var-1"] = "outside-domain"
            variants.append(forged)
            for forged in variants:
                self._assert_forgery_rejected(data, source, raw, forged)

    def test_load_rejects_evidence_with_wrong_support_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._saved_informative_snapshot(directory)
            forged = copy.deepcopy(state)
            hypothesis = next(item for item in forged["hypotheses"] if item["cause"] == "var-0" and item["effect"] == "var-1")
            hypothesis["evidence"] = {"supports": 0, "contradicts": 1}
            hypothesis["supporting_evidence"] = []
            hypothesis["contradicting_evidence"] = [1]
            hypothesis["status"] = "refuted"
            self._assert_forgery_rejected(data, source, raw, forged)

    def test_load_rejects_evidence_references_to_noninformative_or_wrong_cause_experiences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = atlas_data(variable_count=3, action_count=2, support_min=2, refute_min=2)
            engine = create(atlas_bytes(data))
            start = observation(data)
            engine.bind_observation(start)
            after_first = changed_observation(data, start, {"var-0", "var-1"})
            accept(engine, "action-0", after_first)
            # action-0 now has an unchanged target, so this recorded experience is non-informative.
            noninformative_after = changed_observation(data, after_first, {"var-1"})
            accept(engine, "action-0", noninformative_after)
            source = Path(directory) / "source.json"
            engine.save(source)
            raw = source.read_bytes()
            forged = json.loads(raw)
            hypothesis = next(item for item in forged["hypotheses"] if item["cause"] == "var-0" and item["effect"] == "var-1")
            hypothesis["evidence"] = {"supports": 2, "contradicts": 0}
            hypothesis["supporting_evidence"] = [1, 2]
            hypothesis["status"] = "supported"
            self._assert_forgery_rejected(data, source, raw, forged)

            # A different action may be informative while the forged hypothesis cause stays unchanged.
            engine = create(atlas_bytes(data))
            engine.bind_observation(start)
            accept(engine, "action-0", after_first)
            wrong_cause_after = changed_observation(data, after_first, {"var-1"})
            accept(engine, "action-1", wrong_cause_after)
            source = Path(directory) / "wrong-cause-source.json"
            engine.save(source)
            raw = source.read_bytes()
            forged = json.loads(raw)
            hypothesis = next(item for item in forged["hypotheses"] if item["cause"] == "var-0" and item["effect"] == "var-1")
            hypothesis["evidence"] = {"supports": 2, "contradicts": 0}
            hypothesis["supporting_evidence"] = [1, 2]
            hypothesis["status"] = "supported"
            self._assert_forgery_rejected(data, source, raw, forged)



class CoherentSnapshotCausalEvidenceRegressionTests(unittest.TestCase):
    def test_rehashed_wrong_cause_evidence_is_rejected_without_replay(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        start = observation(data)
        engine.bind_observation(start)
        after_first = changed_observation(data, start, {"var-0", "var-1"})
        accept(engine, "action-0", after_first)
        after_second = changed_observation(data, after_first, {"var-1", "var-2"})
        accept(engine, "action-1", after_second)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            forged_path = Path(directory) / "forged.json"
            engine.save(source)
            raw = source.read_bytes()
            forged = json.loads(raw)
            hypothesis = next(
                item for item in forged["hypotheses"]
                if item["cause"] == "var-0" and item["effect"] == "var-1"
            )
            hypothesis["supporting_evidence"] = [2]
            forged["state_commitment"] = sha256_commitment(
                {key: value for key, value in forged.items() if key != "state_commitment"}
            )
            forged_path.write_bytes(canonical_json_bytes(forged))

            with (
                mock.patch(
                    "transition_evidence.api.TransitionEvidenceEngine.predict",
                    side_effect=AssertionError("load must not predict"),
                ),
                mock.patch(
                    "transition_evidence.api.make_candidates",
                    side_effect=AssertionError("load must not generate candidates"),
                ),
                mock.patch(
                    "transition_evidence.api.TransitionEvidenceEngine._apply_evidence",
                    side_effect=AssertionError("load must not apply evidence"),
                ),
                mock.patch(
                    "transition_evidence.relations.CochangesEvaluator.evaluate",
                    side_effect=AssertionError("load must not evaluate"),
                ),
            ):
                with self.assertRaises(SnapshotValidationError) as caught:
                    load(forged_path, atlas_bytes(data))
            self.assertEqual("SNAPSHOT_INVARIANT", caught.exception.code)
            self.assertEqual(raw, source.read_bytes(), "load must not mutate the source snapshot")


    def test_rehashed_evidence_cannot_reference_same_target_but_noninformative_step(self) -> None:
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        start = observation(data)
        engine.bind_observation(start)
        after_first = changed_observation(data, start, {"var-0", "var-1"})
        accept(engine, "action-0", after_first)
        # action-0 still targets var-0, but its target is unchanged here.
        after_second = changed_observation(data, after_first, {"var-1"})
        accept(engine, "action-0", after_second)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            forged_path = Path(directory) / "forged.json"
            engine.save(source)
            raw = source.read_bytes()
            forged = json.loads(raw)
            hypothesis = next(
                item for item in forged["hypotheses"]
                if item["cause"] == "var-0" and item["effect"] == "var-1"
            )
            hypothesis["evidence"] = {"supports": 2, "contradicts": 0}
            hypothesis["supporting_evidence"] = [1, 2]
            hypothesis["status"] = "supported"
            forged["state_commitment"] = sha256_commitment(
                {key: value for key, value in forged.items() if key != "state_commitment"}
            )
            forged_path.write_bytes(canonical_json_bytes(forged))

            with (
                mock.patch(
                    "transition_evidence.api.TransitionEvidenceEngine.predict",
                    side_effect=AssertionError("load must not predict"),
                ),
                mock.patch(
                    "transition_evidence.api.make_candidates",
                    side_effect=AssertionError("load must not generate candidates"),
                ),
                mock.patch(
                    "transition_evidence.api.TransitionEvidenceEngine._apply_evidence",
                    side_effect=AssertionError("load must not apply evidence"),
                ),
                mock.patch(
                    "transition_evidence.relations.CochangesEvaluator.evaluate",
                    side_effect=AssertionError("load must not evaluate"),
                ),
            ):
                with self.assertRaises(SnapshotValidationError) as caught:
                    load(forged_path, atlas_bytes(data))
            self.assertEqual("SNAPSHOT_INVARIANT", caught.exception.code)
            self.assertEqual(raw, source.read_bytes(), "load must not mutate the source snapshot")
