from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.errors import TransitionEvidenceError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation, reverse_order, snapshot_mapping


class ProposedPredictionPolicyTests(unittest.TestCase):
    def _trained_engine(self, proposed_prediction: str | None):
        data = atlas_data(variable_count=4, support_min=2, refute_min=2)
        if proposed_prediction is not None:
            data["policy"]["proposed_prediction"] = proposed_prediction
        engine = create(atlas_bytes(data))
        current = observation(data)
        engine.bind_observation(current)
        # b: support/support; c: contradict/contradict; d: support/contradict => proposed conflict.
        first = changed_observation(data, current, {"var-0", "var-1", "var-3"})
        accept(engine, "action-0", first)
        second = changed_observation(data, first, {"var-0", "var-1"})
        accept(engine, "action-0", second)
        return data, engine

    def test_absent_and_explicit_unknown_preserve_backward_unknown_prediction(self) -> None:
        for mode in (None, "unknown"):
            with self.subTest(mode=mode):
                _, engine = self._trained_engine(mode)
                prediction = engine.predict("action-0")
                self.assertEqual(frozenset({"var-0", "var-1"}), prediction.changed)
                self.assertEqual(frozenset({"var-2"}), prediction.unchanged)
                self.assertEqual(frozenset({"var-3"}), prediction.unknown)

    def test_changed_mode_places_proposed_and_conflicted_effects_in_changed(self) -> None:
        _, engine = self._trained_engine("changed")
        prediction = engine.predict("action-0")
        self.assertEqual(frozenset({"var-0", "var-1", "var-3"}), prediction.changed)
        self.assertEqual(frozenset({"var-2"}), prediction.unchanged)
        self.assertEqual(frozenset(), prediction.unknown)

    def test_invalid_proposed_prediction_has_stable_policy_code_and_path(self) -> None:
        for value in ("other", 7, None, True, ["changed"]):
            with self.subTest(value=repr(value)):
                data = atlas_data()
                data["policy"]["proposed_prediction"] = value
                with self.assertRaises(TransitionEvidenceError) as caught:
                    create(atlas_bytes(data))
                self.assertEqual("BAD_POLICY", caught.exception.code)
                self.assertEqual("/policy/proposed_prediction", caught.exception.path)

    def test_unknown_policy_field_is_rejected(self) -> None:
        data = atlas_data()
        data["policy"]["unrecognised"] = "value"
        with self.assertRaises(TransitionEvidenceError) as caught:
            create(atlas_bytes(data))
        self.assertEqual("UNKNOWN_FIELD", caught.exception.code)
        self.assertEqual("/policy/unrecognised", caught.exception.path)

    def test_absent_default_policy_is_permutation_canonical(self) -> None:
        data = atlas_data(
            variable_count=3,
            hypotheses=[{"id": "source", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"}],
        )
        original = create(atlas_bytes(data))
        permuted = create(atlas_bytes(reverse_order(copy.deepcopy(data))))
        self.assertEqual(snapshot_mapping(original)["atlas_id"], snapshot_mapping(permuted)["atlas_id"])
        self.assertEqual(original.snapshot().to_json_bytes(), permuted.snapshot().to_json_bytes())


    def test_changed_mode_never_predicts_a_nonobservable_atlas_effect(self) -> None:
        data = atlas_data(
            variable_count=3,
            hypotheses=[{"id": "hidden", "cause": "var-0", "effect": "var-2", "relation": "cochanges", "provenance": "atlas"}],
        )
        data["variables"][2]["observable"] = False
        data["policy"]["proposed_prediction"] = "changed"
        before = {key: value for key, value in observation(data).items() if key != "var-2"}
        engine = create(atlas_bytes(data))
        engine.bind_observation(before)
        prediction = engine.predict("action-0")
        self.assertEqual(frozenset({"var-0"}), prediction.changed)
        self.assertEqual(frozenset(), prediction.unchanged)
        self.assertEqual(frozenset({"var-1"}), prediction.unknown)
        self.assertEqual(frozenset(before), prediction.changed | prediction.unchanged | prediction.unknown)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "state.json"
            rewritten = Path(directory) / "rewritten.json"
            engine.save(source)
            restored = load(source, atlas_bytes(data))
            restored.save(rewritten)
            self.assertEqual(source.read_bytes(), rewritten.read_bytes())
            restored.bind_observation(before)
            self.assertEqual(prediction, restored.predict("action-0"))
