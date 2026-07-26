from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.errors import SnapshotValidationError, TransitionEvidenceError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class JsonPointerEscapingRegressionTests(unittest.TestCase):
    def _escaped_atlas(self):
        data = atlas_data(
            variable_count=2,
            hypotheses=[{"id": "source/a~b", "cause": "v/a~b", "effect": "other", "relation": "cochanges", "provenance": "atlas"}],
        )
        data["variables"][0]["id"] = "v/a~b"
        data["variables"][1]["id"] = "other"
        data["actions"][0]["id"] = "act/a~b"
        data["actions"][0]["intervention"]["variable"] = "v/a~b"
        return data

    def test_runtime_and_snapshot_observation_paths_escape_each_token(self) -> None:
        data = self._escaped_atlas()
        before = observation(data)
        engine = create(atlas_bytes(data))
        cases = (
            ("invalid", dict(before, **{"v/a~b": "outside"}), "VALUE_OUT_OF_DOMAIN", "/v~1a~0b"),
            ("missing", {"other": before["other"]}, "OBSERVATION_SHAPE", "/v~1a~0b"),
            ("unknown", dict(before, **{"x/y~z": "opaque"}), "OBSERVATION_SHAPE", "/x~1y~0z"),
        )
        for label, malformed, code, path in cases:
            with self.subTest(boundary=label):
                with self.assertRaises(TransitionEvidenceError) as caught:
                    engine.bind_observation(malformed)
                self.assertEqual(code, caught.exception.code)
                self.assertEqual(path, caught.exception.path)

        engine.bind_observation(before)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "state.json"
            engine.save(source)
            original = json.loads(source.read_bytes())
            for label, malformed, _code, path in cases:
                with self.subTest(snapshot=label):
                    forged = copy.deepcopy(original)
                    forged["current_observation"] = malformed
                    path_to_forge = Path(directory) / f"{label}.json"
                    path_to_forge.write_text(json.dumps(forged), encoding="utf-8")
                    with self.assertRaises(SnapshotValidationError) as caught:
                        load(path_to_forge, atlas_bytes(data))
                    self.assertEqual("SNAPSHOT_SHAPE", caught.exception.code)
                    self.assertEqual(f"/current_observation{path}", caught.exception.path)

    def test_policy_unknown_field_is_escaped(self) -> None:
        data = self._escaped_atlas()
        data["policy"]["x/y~z"] = True
        with self.assertRaises(TransitionEvidenceError) as caught:
            create(atlas_bytes(data))
        self.assertEqual("UNKNOWN_FIELD", caught.exception.code)
        self.assertEqual("/policy/x~1y~0z", caught.exception.path)

    def test_opaque_action_and_source_ids_survive_lifecycle_and_canonical_restart(self) -> None:
        data = self._escaped_atlas()
        before = observation(data)
        engine = create(atlas_bytes(data))
        engine.bind_observation(before)
        after = changed_observation(data, before, {"v/a~b", "other"})
        accept(engine, "act/a~b", after)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "state.json"
            rewritten = Path(directory) / "rewritten.json"
            engine.save(source)
            load(source, atlas_bytes(data)).save(rewritten)
            self.assertEqual(source.read_bytes(), rewritten.read_bytes())


if __name__ == "__main__":
    unittest.main()
