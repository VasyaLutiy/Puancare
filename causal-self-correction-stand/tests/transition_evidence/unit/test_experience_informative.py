from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load
from transition_evidence.canonical import canonical_json_bytes, genesis_chain_commitment, post_chain_commitment, sha256_commitment
from transition_evidence.errors import SnapshotValidationError

from tests.transition_evidence.support import accept, atlas_bytes, atlas_data, changed_observation, observation


class ExperienceInformativeRegressionTests(unittest.TestCase):
    def _two_step_snapshot(self, directory):
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after_first = changed_observation(data, before, {"var-0", "var-1"})
        accept(engine, "action-0", after_first)
        # The target var-0 is unchanged, even though a different observed value moves.
        after_second = changed_observation(data, after_first, {"var-1"})
        accept(engine, "action-0", after_second)
        source = Path(directory) / "source.json"
        engine.save(source)
        return data, source, source.read_bytes(), json.loads(source.read_bytes())

    @staticmethod
    def _entry_payload(experience):
        return {
            "id": experience["id"],
            "before": experience["before"],
            "prediction": experience["prediction"],
            "action": experience["action"],
            "outcome": experience["outcome"],
            "after": experience["after"],
            "comparison": experience["comparison"],
            "informative": experience["informative"],
            "prediction_commitment": experience["prediction_commitment"],
            "previous_entry_commitment": experience["previous_entry_commitment"],
            "pre_chain_commitment": experience["pre_chain_commitment"],
        }

    @staticmethod
    def _rehash_state(state):
        state["state_commitment"] = sha256_commitment(
            {key: value for key, value in state.items() if key != "state_commitment"}
        )

    def test_informative_is_exact_boolean_and_bound_by_entry_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _data, _source, _raw, state = self._two_step_snapshot(directory)
            first, second = state["experiences"]
            self.assertIs(first["informative"], True)
            self.assertIs(second["informative"], False)
            for experience in state["experiences"]:
                self.assertEqual(
                    sha256_commitment(self._entry_payload(experience)),
                    experience["entry_commitment"],
                )

    def test_informative_is_required_and_accepts_only_a_real_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, _raw, state = self._two_step_snapshot(directory)
            variants = []
            missing = copy.deepcopy(state)
            del missing["experiences"][0]["informative"]
            variants.append(missing)
            for invalid in (0, 1, "true"):
                forged = copy.deepcopy(state)
                forged["experiences"][0]["informative"] = invalid
                variants.append(forged)
            for index, forged in enumerate(variants):
                with self.subTest(value=index):
                    path = source.with_name(f"invalid-{index}.json")
                    path.write_text(json.dumps(forged), encoding="utf-8")
                    with self.assertRaises(SnapshotValidationError) as caught:
                        load(path, atlas_bytes(data))
                    self.assertEqual("SNAPSHOT_SHAPE", caught.exception.code)

    def test_unrehashed_informative_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, _raw, state = self._two_step_snapshot(directory)
            forged = copy.deepcopy(state)
            forged["experiences"][0]["informative"] = False
            path = source.with_name("unrehashed.json")
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(SnapshotValidationError) as caught:
                load(path, atlas_bytes(data))
            self.assertEqual("SNAPSHOT_INVARIANT", caught.exception.code)

    def test_rehashed_false_experience_cannot_carry_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, source, _raw, state = self._two_step_snapshot(directory)
            forged = copy.deepcopy(state)
            hypothesis = next(
                item for item in forged["hypotheses"]
                if item["cause"] == "var-0" and item["effect"] == "var-1"
            )
            hypothesis["evidence"] = {"supports": 2, "contradicts": 0}
            hypothesis["supporting_evidence"] = [1, 2]
            hypothesis["status"] = "supported"
            self._rehash_state(forged)
            path = source.with_name("false-with-evidence.json")
            path.write_bytes(canonical_json_bytes(forged))
            with self.assertRaises(SnapshotValidationError) as caught:
                load(path, atlas_bytes(data))
            self.assertEqual("SNAPSHOT_INVARIANT", caught.exception.code)

    def test_coherently_rehashed_informative_field_is_not_recomputed_when_no_refs_exist(self) -> None:
        data = atlas_data(variable_count=3)
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        after = changed_observation(data, before, {"var-0"})
        accept(engine, "action-0", after)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            forged_path = Path(directory) / "coherent.json"
            rewritten = Path(directory) / "rewritten.json"
            engine.save(source)
            forged = json.loads(source.read_bytes())
            experience = forged["experiences"][0]
            self.assertNotEqual(experience["before"]["var-0"], experience["after"]["var-0"])
            experience["informative"] = False
            forged["hypotheses"] = []
            experience["entry_commitment"] = sha256_commitment(self._entry_payload(experience))
            experience["post_chain_commitment"] = post_chain_commitment(
                genesis_chain_commitment(forged["atlas_id"]), experience["entry_commitment"]
            )
            forged["chain_commitment"] = experience["post_chain_commitment"]
            self._rehash_state(forged)
            forged_path.write_bytes(canonical_json_bytes(forged))
            load(forged_path, atlas_bytes(data)).save(rewritten)
            self.assertEqual(forged_path.read_bytes(), rewritten.read_bytes())


if __name__ == "__main__":
    unittest.main()
