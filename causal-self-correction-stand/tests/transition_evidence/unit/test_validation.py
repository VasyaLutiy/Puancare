from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from transition_evidence import create, load
from transition_evidence.errors import TransitionEvidenceError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class ErrorAssertions(unittest.TestCase):
    def assert_code(self, code: str, operation) -> TransitionEvidenceError:
        with self.assertRaises(TransitionEvidenceError) as caught:
            operation()
        self.assertEqual(code, caught.exception.code)
        self.assertIsInstance(caught.exception.path, str)
        return caught.exception


class SchemaValidationTests(ErrorAssertions):
    def invalid(self, mutate) -> None:
        data = atlas_data()
        mutate(data)
        return create(atlas_bytes(data))

    def test_schema_version(self) -> None:
        self.assert_code("SCHEMA_VERSION", lambda: self.invalid(lambda d: d.__setitem__("schema", "other/v9")))

    def test_duplicate_id(self) -> None:
        self.assert_code("DUPLICATE_ID", lambda: self.invalid(lambda d: d["variables"].append(dict(d["variables"][0]))))

    def test_bad_domain(self) -> None:
        self.assert_code("BAD_DOMAIN", lambda: self.invalid(lambda d: d["variables"][0].__setitem__("domain", ["same", "same"])))

    def test_bad_action_target(self) -> None:
        self.assert_code("BAD_ACTION_TARGET", lambda: self.invalid(lambda d: d["actions"][0]["intervention"].__setitem__("variable", "missing")))

    def test_bad_hypothesis_endpoint(self) -> None:
        def mutate(data):
            data["hypotheses"] = [{"id": "h", "cause": "var-0", "effect": "var-0", "relation": "cochanges", "provenance": "atlas"}]
        self.assert_code("BAD_HYPOTHESIS_ENDPOINT", lambda: self.invalid(mutate))

    def test_duplicate_hypothesis(self) -> None:
        def mutate(data):
            data["hypotheses"] = [
                {"id": "one", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"},
                {"id": "two", "cause": "var-0", "effect": "var-1", "relation": "cochanges", "provenance": "atlas"},
            ]
        self.assert_code("DUPLICATE_HYPOTHESIS", lambda: self.invalid(mutate))

    def test_unknown_relation(self) -> None:
        def mutate(data):
            data["hypotheses"] = [{"id": "h", "cause": "var-0", "effect": "var-1", "relation": "implies", "provenance": "atlas"}]
        self.assert_code("UNKNOWN_RELATION", lambda: self.invalid(mutate))

    def test_unknown_field(self) -> None:
        self.assert_code("UNKNOWN_FIELD", lambda: self.invalid(lambda d: d.__setitem__("unrecognised", True)))

    def test_initial_state_forbidden(self) -> None:
        self.assert_code("INITIAL_STATE_FORBIDDEN", lambda: self.invalid(lambda d: d.__setitem__("initial_state", {})))

    def test_bad_policy(self) -> None:
        self.assert_code("BAD_POLICY", lambda: self.invalid(lambda d: d["policy"].__setitem__("support_min", 0)))

    def test_cyclic_yaml(self) -> None:
        cycle = b"schema: transition-atlas/v1\nvariables: &loop [*loop]\nactions: []\nhypotheses: []\npolicy: {}\n"
        self.assert_code("CYCLIC_YAML", lambda: create(cycle))


class RuntimeValidationTests(ErrorAssertions):
    def setUp(self) -> None:
        self.data = atlas_data()
        self.engine = create(atlas_bytes(self.data))
        self.before = observation(self.data)

    def test_observation_shape_missing_and_unknown_key(self) -> None:
        missing = {"var-0": self.before["var-0"]}
        self.assert_code("OBSERVATION_SHAPE", lambda: self.engine.bind_observation(missing))
        extra = dict(self.before, extra="not-a-variable")
        self.assert_code("OBSERVATION_SHAPE", lambda: self.engine.bind_observation(extra))

    def test_value_out_of_domain_before_and_after(self) -> None:
        malformed = dict(self.before, **{"var-0": "outside"})
        self.assert_code("VALUE_OUT_OF_DOMAIN", lambda: self.engine.bind_observation(malformed))
        self.engine.bind_observation(self.before)
        prediction = self.engine.predict("action-0")
        after = dict(self.before, **{"var-0": "outside"})
        self.assert_code("VALUE_OUT_OF_DOMAIN", lambda: self.engine.accept_transition(prediction.id, {"action_id": "action-0", "outcome": "ok", "obs_after": after}))

    def test_action_not_available(self) -> None:
        self.engine.bind_observation(self.before)
        self.assert_code("ACTION_NOT_AVAILABLE", lambda: self.engine.predict("absent-action"))

    def test_observation_unbound_before_bind_and_after_reload(self) -> None:
        self.assert_code("OBSERVATION_UNBOUND", lambda: self.engine.predict("action-0"))
        self.engine.bind_observation(self.before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.engine.save(path)
            restored = load(path, atlas_bytes(self.data))
        self.assert_code("OBSERVATION_UNBOUND", lambda: restored.predict("action-0"))

    def test_prediction_required_for_missing_or_consumed_id(self) -> None:
        self.engine.bind_observation(self.before)
        after = changed_observation(self.data, self.before, {"var-0"})
        self.assert_code("PREDICTION_REQUIRED", lambda: self.engine.accept_transition("missing", {"action_id": "action-0", "outcome": "ok", "obs_after": after}))
        prediction = self.engine.predict("action-0")
        self.engine.accept_transition(prediction.id, {"action_id": "action-0", "outcome": "ok", "obs_after": after})
        self.assert_code("PREDICTION_REQUIRED", lambda: self.engine.accept_transition(prediction.id, {"action_id": "action-0", "outcome": "ok", "obs_after": after}))

    def test_prediction_mismatch_does_not_mutate(self) -> None:
        self.engine.bind_observation(self.before)
        prediction = self.engine.predict("action-0")
        before_state = self.engine.snapshot().to_json_bytes()
        after = changed_observation(self.data, self.before, {"var-0"})
        self.assert_code("PREDICTION_MISMATCH", lambda: self.engine.accept_transition(prediction.id, {"action_id": "another", "outcome": "ok", "obs_after": after}))
        self.assertEqual(before_state, self.engine.snapshot().to_json_bytes())

    def test_duplicate_experience_in_corrupted_snapshot(self) -> None:
        self.engine.bind_observation(self.before)
        accept(self.engine, "action-0", changed_observation(self.data, self.before, {"var-0"}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.engine.save(path)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["experiences"].append(dict(payload["experiences"][0]))
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            self.assert_code("DUPLICATE_EXPERIENCE", lambda: load(path, atlas_bytes(self.data)))

