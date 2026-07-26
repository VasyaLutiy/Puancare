from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.canonical import genesis_entry_commitment
from transition_evidence.errors import SnapshotValidationError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class CommitmentChainRegressionTests(unittest.TestCase):
    def _multi_step_snapshot(self, directory: str):
        data = atlas_data(variable_count=3, action_count=2)
        engine = create(atlas_bytes(data))
        start = observation(data)
        engine.bind_observation(start)
        after_first = changed_observation(data, start, {"var-0", "var-1"})
        accept(engine, "action-0", after_first, outcome="first")
        after_second = changed_observation(data, after_first, {"var-1", "var-2"})
        accept(engine, "action-1", after_second, outcome="second")
        source = Path(directory) / "source.json"
        engine.save(source)
        raw = source.read_bytes()
        return data, source, raw, json.loads(raw)

    def _assert_rejected(self, data, source: Path, raw: bytes, forged: dict) -> None:
        path = source.with_name("forged.json")
        path.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with self.assertRaises(SnapshotValidationError):
            load(path, atlas_bytes(data))
        self.assertEqual(raw, source.read_bytes(), "loading a forge must not mutate the source")

    def test_valid_multi_step_snapshot_load_save_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._multi_step_snapshot(directory)
            self.assertEqual(
                genesis_entry_commitment(state["atlas_id"]),
                state["experiences"][0]["previous_entry_commitment"],
                "the first entry must carry the deterministic genesis link",
            )
            rewritten = Path(directory) / "rewritten.json"
            load(source, atlas_bytes(data)).save(rewritten)
            self.assertEqual(raw, rewritten.read_bytes())

    def test_all_payload_and_chain_commitment_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, raw, state = self._multi_step_snapshot(directory)
            mutations = []

            def changed_scalar(value):
                return "forged" if value != "forged" else "forged-2"

            def add(label, mutate):
                forged = copy.deepcopy(state)
                mutate(forged)
                mutations.append((label, forged))

            add("full-prediction", lambda s: s["experiences"][0]["prediction"].__setitem__("id", "forged"))
            add("action", lambda s: s["experiences"][0]["action"]["intervention"].__setitem__("operation", "forged"))
            add("outcome", lambda s: s["experiences"][0].__setitem__("outcome", "forged"))
            add("before", lambda s: s["experiences"][0]["before"].__setitem__("var-0", "value-0-1"))
            add("after", lambda s: s["experiences"][0]["after"].__setitem__("var-0", "value-0-0"))
            add("comparison", lambda s: s["experiences"][0]["comparison"].__setitem__("actual_changed", []))
            add("evidence", lambda s: s["hypotheses"][0]["evidence"].__setitem__("supports", 99))
            add("current", lambda s: s["current_observation"].__setitem__("var-0", "value-0-0"))
            add("version", lambda s: s.__setitem__("version", 99))
            add("reorder", lambda s: s.__setitem__("experiences", list(reversed(s["experiences"]))))
            add("backdate", lambda s: s["experiences"][1].__setitem__("id", 1))
            add("top-chain", lambda s: s.__setitem__("chain_commitment", changed_scalar(s["chain_commitment"])))
            add("top-state", lambda s: s.__setitem__("state_commitment", changed_scalar(s["state_commitment"])))
            for field in (
                "prediction_commitment",
                "previous_entry_commitment",
                "pre_chain_commitment",
                "entry_commitment",
                "post_chain_commitment",
            ):
                add(field, lambda s, field=field: s["experiences"][1].__setitem__(field, changed_scalar(s["experiences"][1][field])))

            for label, forged in mutations:
                with self.subTest(tamper=label):
                    self._assert_rejected(data, source, raw, forged)

