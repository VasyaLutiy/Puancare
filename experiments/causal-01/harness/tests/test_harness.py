from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
HARNESS = ROOT / "experiments" / "causal-01" / "harness" / "harness.py"
SOURCE_PILOT = ROOT / "experiments" / "causal-01" / "pilot-001"
sys.path.insert(0, str(ROOT / "experiments" / "causal-01" / "harness"))
from harness import (  # noqa: E402
    PARENT_PROBE,
    PARENT_PROBE_SUCCESS,
    Learner,
    HarnessError,
    assert_e1_prediction,
    assert_parent_probe_result,
    create,
    create_private_examiner_module,
    secure_regular_read,
    sandbox_command,
)


class HarnessTest(unittest.TestCase):
    def copied_pilot(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        destination = Path(temporary.name) / "pilot-001"
        shutil.copytree(SOURCE_PILOT / "public", destination / "public")
        self.write_synthetic_oracle(destination)
        return temporary, destination

    def write_synthetic_oracle(self, pilot: Path) -> Path:
        """Build a unique private fixture from the public atlas, never from pilot data."""
        atlas_file = pilot / "public" / "atlas.yaml"
        atlas_bytes = atlas_file.read_bytes()
        atlas = yaml.safe_load(atlas_bytes)
        self.assertIsInstance(atlas, dict)
        variables = atlas["variables"]
        actions = atlas["actions"]
        prior = atlas["hypotheses"][0]
        domains = {variable["id"]: variable["domain"] for variable in variables}
        cause = prior["cause"]
        false_effect = prior["effect"]
        cause_action = next(action["id"] for action in actions if action["intervention"]["variable"] == cause)
        control_action = next(action["id"] for action in actions if action["id"] != cause_action)
        control = next(action["intervention"]["variable"] for action in actions if action["id"] == control_action)
        true_effect = next(variable for variable in domains if variable not in {cause, false_effect, control})
        nonce = uuid.uuid4().hex
        initial = {variable: domain[0] for variable, domain in domains.items()}
        oracle = {
            "synthetic_test_nonce": nonce,
            "public_atlas": {
                "file_sha256": hashlib.sha256(atlas_bytes).hexdigest(),
                "engine_atlas_id": create(atlas_bytes).snapshot().as_mapping()["atlas_id"],
            },
            "action_rules": {
                cause_action: {"operation": "toggle", "changed_variables": [cause, true_effect], "outcome": f"synthetic-{nonce}"},
                control_action: {"operation": "toggle", "changed_variables": [control], "outcome": f"synthetic-{nonce}"},
            },
            "training_schedule": [cause_action, control_action, cause_action, control_action],
            "initial_observation": initial,
            "limits": {"max_steps": 5},
            "restart": {"after_training_step": 4},
            "held_out": {"initial_observation": {variable: domain[1] for variable, domain in domains.items()}, "action_id": cause_action},
            "expected_after_training": {
                "supported_edges": [[cause, true_effect, "cochanges"]],
                "refuted_edges": [[cause, false_effect, "cochanges"]],
            },
            "expected_held_out_prediction": {
                "changed": [cause, true_effect],
                "unchanged": sorted(set(domains) - {cause, true_effect}),
                "unknown": [],
            },
        }
        oracle_file = pilot / "examiner-private" / "oracle.yaml"
        oracle_file.parent.mkdir()
        oracle_file.write_text(yaml.safe_dump(oracle, sort_keys=False))
        self.assertIn(nonce, oracle_file.read_text())
        return oracle_file

    def run_harness(self, pilot: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["python3", str(HARNESS), "--pilot-root", str(pilot)], text=True, capture_output=True,
            check=False, timeout=60)

    def test_happy_path_writes_public_artifacts_and_all_gates(self) -> None:
        temporary, pilot = self.copied_pilot()
        with temporary:
            completed = self.run_harness(pilot)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifacts = pilot / "artifacts"
            result = json.loads((artifacts / "result.json").read_text())
            self.assertEqual(result["schema"], "causal-01-result/v2")
            self.assertEqual(result["verdict"], "PASS")
            self.assertEqual(result["gates"], {f"E{item}": True for item in range(6)})
            self.assertFalse((artifacts / "oracle.yaml").exists())
            events = [json.loads(line) for line in (artifacts / "transcript.jsonl").read_text().splitlines()]
            self.assertTrue(any(event["event"] == "predict" and event["reply"] == {"ok": False, "code": "OBSERVATION_UNBOUND"} for event in events))
            probes = [event for event in events if event["event"] == "parent_sandbox_probe"]
            self.assertEqual(len(probes), 3)
            for event in probes:
                self.assertEqual(event["proof"], "parent direct, proc-self-root, namespace-pid-root, and examiner-import denials")
                self.assertNotIn("/", event["proof"])
                self.assertEqual({key: event[key] for key in ("filesystem_isolated", "pid_isolated", "network_isolated")},
                    {"filesystem_isolated": True, "pid_isolated": True, "network_isolated": False})
            self.assertEqual(sum(event["event"] == "sandbox_confirmed" and event["filesystem_isolated"] is True and event["pid_isolated"] is True and event["network_isolated"] is False for event in events), 3)
            self.assertTrue(all(event["network_isolated"] is False for event in events if event["event"] == "learner_start"))
            for position, event in enumerate(events):
                if event["event"] == "learner_start":
                    self.assertEqual(events[position - 1]["event"], "parent_sandbox_probe")
            for position, event in enumerate(events):
                if event["event"] == "step":
                    self.assertEqual(events[position - 1]["event"], "prediction_recorded")

    def test_tampered_public_hash_fails_before_pass(self) -> None:
        temporary, pilot = self.copied_pilot()
        with temporary:
            oracle = pilot / "examiner-private" / "oracle.yaml"
            data = yaml.safe_load(oracle.read_text())
            data["public_atlas"]["file_sha256"] = "0" * 64
            oracle.write_text(yaml.safe_dump(data, sort_keys=False))
            completed = self.run_harness(pilot)
            self.assertEqual(completed.returncode, 1)
            result = json.loads((pilot / "artifacts" / "result.json").read_text())
            self.assertEqual(result["verdict"], "FAIL")
            self.assertFalse(result["gates"]["E1"])

    def test_e1_rejects_true_effect_overprediction(self) -> None:
        with self.assertRaises(HarnessError):
            assert_e1_prediction({"changed": ["false", "true"]}, "false", "true")

    def test_parent_probe_requires_exact_success_proof(self) -> None:
        with self.assertRaises(HarnessError):
            assert_parent_probe_result(subprocess.CompletedProcess([], 0, PARENT_PROBE_SUCCESS + "forged\n", ""))
        with self.assertRaises(HarnessError):
            assert_parent_probe_result(subprocess.CompletedProcess([], 1, PARENT_PROBE_SUCCESS, ""))
        assert_parent_probe_result(subprocess.CompletedProcess([], 0, PARENT_PROBE_SUCCESS, ""))

    def test_parent_bwrap_probe_denies_synthetic_oracle_canary_and_private_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, work = root / "runtime", root / "work"
            runtime.mkdir(); work.mkdir()
            private = root / "examiner-visible"
            private.mkdir()
            nonce = uuid.uuid4().hex
            oracle, canary = private / "oracle.yaml", private / "canary.secret"
            oracle.write_bytes(f"synthetic-oracle-{nonce}".encode())
            canary.write_bytes(f"synthetic-canary-{nonce}".encode())
            module = create_private_examiner_module(private, marker=f"synthetic-module-{nonce}")
            self.assertEqual(oracle.read_bytes(), f"synthetic-oracle-{nonce}".encode())
            self.assertEqual(canary.read_bytes(), f"synthetic-canary-{nonce}".encode())
            self.assertIn(nonce, module.read_text())
            python = Path(sys.executable).resolve()
            targets = [
                str(oracle), str(canary),
                f"/proc/self/root{oracle}", f"/proc/self/root{canary}",
                f"/proc/1/root{oracle}", f"/proc/1/root{canary}",
            ]
            completed = subprocess.run(sandbox_command(runtime, work, python, [str(python), "-I", "-c", PARENT_PROBE, *targets]),
                text=True, capture_output=True, check=False, timeout=10)
            assert_parent_probe_result(completed)

    def test_parent_probe_failure_forbids_learner_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "examiner-visible"
            private.mkdir()
            nonce = uuid.uuid4().hex
            oracle, canary = private / "oracle.yaml", private / "canary.secret"
            oracle.write_bytes(f"synthetic-oracle-{nonce}".encode())
            canary.write_bytes(f"synthetic-canary-{nonce}".encode())
            module = create_private_examiner_module(private, marker=f"synthetic-module-{nonce}")
            self.assertIn(nonce, module.read_text())
            atlas = (SOURCE_PILOT / "public" / "atlas.yaml").read_bytes()
            events: list[dict[str, object]] = []
            failed_probe = subprocess.CompletedProcess([], 3, "", "read unexpectedly succeeded")
            with patch("harness.subprocess.run", return_value=failed_probe):
                with self.assertRaises(HarnessError):
                    Learner(atlas, root / "published.json", None, oracle, canary, events, "test")
            self.assertFalse(any(event["event"] == "learner_start" for event in events))

    def test_secure_shelf_reader_rejects_a_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            work.mkdir()
            secret = Path(directory) / "examiner-private.secret"
            secret.write_bytes(b"must-not-publish")
            (work / "shelf.json").symlink_to(secret)
            descriptor = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                with self.assertRaises(HarnessError):
                    secure_regular_read(descriptor, "shelf.json")
            finally:
                os.close(descriptor)

    def test_tampered_held_out_oracle_expectation_fails_cross_check(self) -> None:
        temporary, pilot = self.copied_pilot()
        with temporary:
            oracle = pilot / "examiner-private" / "oracle.yaml"
            data = yaml.safe_load(oracle.read_text())
            data["expected_held_out_prediction"]["unknown"] = [data["expected_held_out_prediction"]["unchanged"].pop()]
            oracle.write_text(yaml.safe_dump(data, sort_keys=False))
            completed = self.run_harness(pilot)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(json.loads((pilot / "artifacts" / "result.json").read_text())["verdict"], "FAIL")

    def test_invalid_profile_overwrites_stale_pass_artifacts(self) -> None:
        temporary, pilot = self.copied_pilot()
        with temporary:
            artifacts = pilot / "artifacts"
            artifacts.mkdir(exist_ok=True)
            (artifacts / "result.json").write_text('{"verdict":"PASS"}')
            (artifacts / "training-shelf.json").write_text("stale")
            atlas = pilot / "public" / "atlas.yaml"
            data = yaml.safe_load(atlas.read_text())
            data["variables"].pop()
            atlas.write_text(yaml.safe_dump(data, sort_keys=False))
            completed = self.run_harness(pilot)
            self.assertEqual(completed.returncode, 1)
            result = json.loads((artifacts / "result.json").read_text())
            self.assertEqual(result["verdict"], "FAIL")
            self.assertFalse((artifacts / "training-shelf.json").exists())


if __name__ == "__main__":
    unittest.main()
