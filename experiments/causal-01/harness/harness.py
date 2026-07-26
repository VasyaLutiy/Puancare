#!/usr/bin/env python3
"""Fail-safe CAUSAL-01 examiner with a mandatory bubblewrap namespace."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import json
import os
import shutil
import select
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = ROOT / "causal-self-correction-stand"
BWRAP = Path("/usr/bin/bwrap")
SHELF_NAME = "shelf.json"
MAX_SHELF_BYTES = 4 * 1024 * 1024
PARENT_PROBE_SUCCESS = "PARENT_BWRAP_PROBE_OK\n"
PRIVATE_EXAMINER_MODULE = "causal_examiner_private_oracle"
PROBE_TIMEOUT_SECONDS = 10
LEARNER_RESPONSE_TIMEOUT_SECONDS = 10
PARENT_PROBE = """\
import importlib
import sys
from pathlib import Path

targets = sys.argv[1:]
if len(targets) != 6:
    raise SystemExit(2)
for target in targets:
    try:
        Path(target).read_bytes()
    except OSError:
        continue
    raise SystemExit(3)
try:
    importlib.import_module("causal_examiner_private_oracle")
except ImportError:
    pass
else:
    raise SystemExit(4)
print("PARENT_BWRAP_PROBE_OK")
"""
sys.path.insert(0, str(ENGINE_ROOT))
from transition_evidence import TransitionEvidenceError, create, load  # noqa: E402


class HarnessError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise HarnessError(message)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() 


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Durably replace one artifact without exposing a partial official file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def secure_regular_read(directory_fd: int, name: str) -> bytes:
    """Read exactly one regular child without resolving any untrusted pathname."""
    check(name == SHELF_NAME, "unexpected learner shelf name")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        fail(f"learner shelf unavailable: {error.strerror}")
    try:
        metadata = os.fstat(descriptor)
        check(stat.S_ISREG(metadata.st_mode), "learner shelf is not a regular file")
        check(metadata.st_size <= MAX_SHELF_BYTES, "learner shelf exceeds size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_SHELF_BYTES + 1)
        check(len(data) <= MAX_SHELF_BYTES, "learner shelf exceeds size limit")
        return data
    finally:
        os.close(descriptor)


def validate_shelf_bytes(data: bytes, atlas_bytes: bytes) -> dict[str, Any]:
    """Accept only a canonical engine snapshot before publishing it as official."""
    check(data, "learner shelf is empty")
    with tempfile.TemporaryDirectory(prefix="causal-shelf-validate-") as directory:
        candidate = Path(directory) / SHELF_NAME
        atomic_write(candidate, data)
        loaded = load(candidate, atlas_bytes).snapshot().as_mapping()
    parsed = json.loads(data)
    check(isinstance(parsed, dict) and parsed.get("atlas_id") == loaded.get("atlas_id"), "learner shelf validation mismatch")
    return parsed


def sandbox_command(runtime: Path, work: Path, python: Path, program: list[str]) -> list[str]:
    """One fixed sandbox configuration shared by the parent probe and learner."""
    system_binds = [Path("/usr"), Path("/lib"), Path("/lib64"), Path("/bin")]
    if python.parent.parent not in system_binds:
        system_binds.append(python.parent.parent)
    command = [str(BWRAP), "--die-with-parent", "--new-session", "--unshare-pid", "--proc", "/proc",
        "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/run", "--ro-bind", str(runtime), "/app", "--bind", str(work), "/work", "--chdir", "/work",
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONIOENCODING", "utf-8"]
    for system_path in system_binds:
        if system_path.exists():
            command.extend(["--ro-bind", str(system_path), str(system_path)])
    return [*command, *program]


def assert_parent_probe_result(completed: subprocess.CompletedProcess[str]) -> None:
    check(completed.returncode == 0 and completed.stdout == PARENT_PROBE_SUCCESS and completed.stderr == "", "parent-controlled sandbox probe")


def create_private_examiner_module(directory: Path) -> Path:
    """Create and positively load a private module before proving sandbox denial."""
    module = directory / f"{PRIVATE_EXAMINER_MODULE}.py"
    atomic_write(module, b"EXAMINER_PRIVATE_MARKER = 'parent-only'\n")
    sys.path.insert(0, str(directory))
    try:
        sys.modules.pop(PRIVATE_EXAMINER_MODULE, None)
        loaded = importlib.import_module(PRIVATE_EXAMINER_MODULE)
        check(getattr(loaded, "EXAMINER_PRIVATE_MARKER", None) == "parent-only", "private examiner module import")
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop(PRIVATE_EXAMINER_MODULE, None)
    return module


def yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_bytes())
    check(isinstance(value, dict), "input YAML must be an object")
    return value


def profile(atlas: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    variables, actions, hypotheses, policy = (atlas.get(key) for key in ("variables", "actions", "hypotheses", "policy"))
    check(atlas.get("schema") == "transition-atlas/v1", "BAD_EXPERIMENT_SHAPE schema")
    check(isinstance(variables, list) and len(variables) == 5, "BAD_EXPERIMENT_SHAPE variables")
    check(isinstance(actions, list) and len(actions) == 2, "BAD_EXPERIMENT_SHAPE actions")
    check(isinstance(hypotheses, list) and len(hypotheses) == 1 and isinstance(policy, dict), "BAD_EXPERIMENT_SHAPE prior/policy")
    domains: dict[str, list[Any]] = {}
    for variable in variables:
        check(isinstance(variable, dict) and isinstance(variable.get("id"), str), "BAD_EXPERIMENT_SHAPE variable id")
        domain = variable.get("domain")
        check(variable.get("observable") is True and isinstance(domain, list) and len(domain) == 2 and domain[0] != domain[1], "BAD_EXPERIMENT_SHAPE binary observable")
        domains[variable["id"]] = domain
    check(len(domains) == 5, "BAD_EXPERIMENT_SHAPE duplicate variable")
    indexed: dict[str, dict[str, Any]] = {}
    targets: list[str] = []
    for action in actions:
        intervention = action.get("intervention") if isinstance(action, dict) else None
        check(isinstance(action, dict) and isinstance(action.get("id"), str) and isinstance(intervention, dict), "BAD_EXPERIMENT_SHAPE action")
        target = intervention.get("variable")
        check(intervention.get("operation") == "toggle" and target in domains, "BAD_EXPERIMENT_SHAPE toggle")
        indexed[action["id"]] = action
        targets.append(target)
    check(len(indexed) == 2 and len(set(targets)) == 2, "BAD_EXPERIMENT_SHAPE action targets")
    prior = hypotheses[0]
    check(isinstance(prior, dict) and isinstance(prior.get("id"), str) and prior.get("relation") == "cochanges", "BAD_EXPERIMENT_SHAPE cochanges")
    check(prior.get("cause") in domains and prior.get("effect") in domains and prior["cause"] != prior["effect"], "BAD_EXPERIMENT_SHAPE prior endpoints")
    check(targets.count(prior["cause"]) == 1, "BAD_EXPERIMENT_SHAPE prior cause target")
    check(policy == {"support_min": 2, "refute_min": 2, "candidate_generation": "all_observable_effects", "proposed_prediction": "changed"}, "BAD_EXPERIMENT_SHAPE policy")
    return domains, indexed, prior


def reordered(atlas: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(atlas)
    value["variables"] = list(reversed(value["variables"]))
    value["actions"] = list(reversed(value["actions"]))
    value["hypotheses"] = list(reversed(value["hypotheses"]))
    for item in value["variables"]:
        item["domain"] = list(reversed(item["domain"]))
    return value


def e0_fixture_matrix(atlas: dict[str, Any]) -> list[str]:
    def invalid(code: str, mutate: Callable[[dict[str, Any]], None]) -> str:
        value = copy.deepcopy(atlas)
        mutate(value)
        try:
            create(yaml.safe_dump(value, sort_keys=False).encode())
        except TransitionEvidenceError as error:
            check(error.code == code, f"E0 fixture expected {code}, got {error.code}")
            return code
        fail(f"E0 fixture accepted: {code}")
    checks = [
        invalid("SCHEMA_VERSION", lambda v: v.__setitem__("schema", "bad/v1")),
        invalid("DUPLICATE_ID", lambda v: v["variables"].append(copy.deepcopy(v["variables"][0]))),
        invalid("BAD_DOMAIN", lambda v: v["variables"][0].__setitem__("domain", ["same", "same"])),
        invalid("BAD_ACTION_TARGET", lambda v: v["actions"][0]["intervention"].__setitem__("variable", "missing")),
        invalid("BAD_HYPOTHESIS_ENDPOINT", lambda v: v["hypotheses"][0].__setitem__("effect", v["hypotheses"][0]["cause"])),
        invalid("UNKNOWN_RELATION", lambda v: v["hypotheses"][0].__setitem__("relation", "nope")),
        invalid("INITIAL_STATE_FORBIDDEN", lambda v: v.__setitem__("initial_state", {})),
        invalid("UNKNOWN_FIELD", lambda v: v.__setitem__("unexpected", True)),
    ]
    bad = copy.deepcopy(atlas)
    bad["variables"].pop()
    try:
        profile(bad)
    except HarnessError as error:
        check(str(error).startswith("BAD_EXPERIMENT_SHAPE"), "E0 wrapper code")
        checks.append("BAD_EXPERIMENT_SHAPE")
    else:
        fail("E0 wrapper fixture accepted")
    return checks


def validate_e0(atlas_bytes: bytes, atlas: dict[str, Any]) -> list[str]:
    profile(atlas)
    baseline, permuted = create(atlas_bytes), create(yaml.safe_dump(reordered(atlas), sort_keys=False).encode())
    check(baseline.snapshot().as_mapping()["atlas_id"] == permuted.snapshot().as_mapping()["atlas_id"], "E0 permutation identity")
    with tempfile.TemporaryDirectory() as directory:
        one, two = Path(directory) / "one.json", Path(directory) / "two.json"
        baseline.save(one)
        load(one, atlas_bytes).save(two)
        check(one.read_bytes() == two.read_bytes(), "E0 empty roundtrip")
    return e0_fixture_matrix(atlas)


@dataclass
class Learner:
    atlas_bytes: bytes
    output: Path
    snapshot_bytes: bytes | None
    oracle_path: Path
    canary_path: Path
    events: list[dict[str, Any]]
    label: str

    def __post_init__(self) -> None:
        check(BWRAP.is_file() and os.access(BWRAP, os.X_OK), "bubblewrap unavailable")
        self.temp = tempfile.TemporaryDirectory(prefix="causal-learner-")
        root = Path(self.temp.name)
        runtime, work = root / "runtime", root / "work"
        runtime.mkdir()
        work.mkdir()
        create_private_examiner_module(root / "examiner-private")
        shutil.copy2(Path(__file__).with_name("learner.py"), runtime / "learner.py")
        shutil.copytree(ENGINE_ROOT / "transition_evidence", runtime / "transition_evidence", ignore=shutil.ignore_patterns("__pycache__"))
        self.work_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        python = Path(sys.executable).resolve()
        probe_targets = [
            str(self.oracle_path), str(self.canary_path),
            f"/proc/self/root{self.oracle_path}", f"/proc/self/root{self.canary_path}",
            f"/proc/1/root{self.oracle_path}", f"/proc/1/root{self.canary_path}",
        ]
        probe_command = sandbox_command(runtime, work, python, [str(python), "-I", "-c", PARENT_PROBE, *probe_targets])
        try:
            try:
                probe = subprocess.run(probe_command, text=True, capture_output=True, check=False, timeout=PROBE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                fail(f"parent-controlled sandbox probe timed out after {error.timeout} seconds")
            assert_parent_probe_result(probe)
        except Exception:
            os.close(self.work_fd)
            self.work_fd = -1
            self.temp.cleanup()
            raise
        self.event("parent_sandbox_probe", filesystem_isolated=True, pid_isolated=True, network_isolated=False,
            proof="parent direct, proc-self-root, namespace-pid-root, and examiner-import denials")
        command = sandbox_command(runtime, work, python, [str(python), "-I", "/app/learner.py", "--shelf", f"/work/{SHELF_NAME}"])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=work, env={"PATH": os.environ.get("PATH", "")})
        self.event("learner_start", pid=self.process.pid, filesystem_isolated=True, pid_isolated=True,
            network_isolated=False, sandbox="bwrap-filesystem-pid")
        try:
            ready = self.call({"bootstrap": True, "atlas_b64": base64.b64encode(self.atlas_bytes).decode(),
                "snapshot_b64": None if self.snapshot_bytes is None else base64.b64encode(self.snapshot_bytes).decode()}, raw=True)
            check(ready.get("ok") is True and isinstance(ready.get("pid"), int) and ready["pid"] > 0, "learner bootstrap")
        except Exception:
            self.abort()
            raise
        self.event("learner_namespace_pid", host_pid=self.process.pid, namespace_pid=ready["pid"])
        self.event("load", source="snapshot_bytes" if self.snapshot_bytes is not None else "empty_shelf")
        self.event("sandbox_confirmed", filesystem_isolated=True, pid_isolated=True, network_isolated=False,
            proof="parent-controlled filesystem and PID probe completed before learner startup")

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"sequence": len(self.events) + 1, "phase": self.label, "event": kind, **data})

    def call(self, request: dict[str, Any], *, raw: bool = False) -> dict[str, Any]:
        check(self.process.stdin is not None and self.process.stdout is not None, "learner pipes")
        self.process.stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        readable, _, _ = select.select([self.process.stdout], [], [], LEARNER_RESPONSE_TIMEOUT_SECONDS)
        if not readable:
            self.abort()
            fail(f"learner response timed out after {LEARNER_RESPONSE_TIMEOUT_SECONDS} seconds")
        line = self.process.stdout.readline()
        if line == "":
            stderr = self.process.stderr.read() if self.process.stderr else ""
            self.abort()
            fail(f"learner/bwrap exited: {stderr.strip() or 'no response'}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            self.abort()
            fail("learner response is not JSON")
        check(isinstance(reply, dict), "learner response")
        if not raw:
            self.event(request.get("command", "bootstrap"), reply=reply)
        return reply

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=PROBE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=PROBE_TIMEOUT_SECONDS)
        if self.work_fd >= 0:
            os.close(self.work_fd)
            self.work_fd = -1
        self.temp.cleanup()

    def save_to(self) -> bytes:
        reply = self.call({"command": "save"})
        check(reply.get("ok") is True, "learner save")
        data = secure_regular_read(self.work_fd, SHELF_NAME)
        snapshot = validate_shelf_bytes(data, self.atlas_bytes)
        self.event("commitments", chain=snapshot["chain_commitment"], state=snapshot["state_commitment"])
        return data

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            status = self.process.wait(timeout=PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.abort()
            fail("learner exit timed out")
        stderr = self.process.stderr.read() if self.process.stderr else ""
        self.event("learner_exit", pid=self.process.pid, status=status)
        if self.work_fd >= 0:
            os.close(self.work_fd)
            self.work_fd = -1
        self.temp.cleanup()
        check(status == 0 and not stderr, "learner exit")


def append(events: list[dict[str, Any]], phase: str, event: str, **data: Any) -> None:
    events.append({"sequence": len(events) + 1, "phase": phase, "event": event, **data})


def prediction(reply: dict[str, Any], variables: set[str]) -> dict[str, Any]:
    check(reply.get("ok") is True and isinstance(reply.get("prediction"), dict), "prediction failed")
    value = reply["prediction"]
    buckets = [set(value[name]) for name in ("changed", "unchanged", "unknown")]
    check(not (buckets[0] & buckets[1] or buckets[0] & buckets[2] or buckets[1] & buckets[2]), "partition overlap")
    check(set.union(*buckets) == variables, "partition incomplete")
    return value


def assert_e1_prediction(value: dict[str, Any], false_effect: str, true_effect: str) -> None:
    check(false_effect in value["changed"] and true_effect not in value["changed"], "E1 false-prior prediction / true overprediction")


def flip(domains: dict[str, list[Any]], variable: str, value: Any) -> Any:
    check(value in domains[variable], "oracle value outside domain")
    return domains[variable][1] if value == domains[variable][0] else domains[variable][0]


def step(before: dict[str, Any], action: str, rules: dict[str, Any], domains: dict[str, list[Any]]) -> dict[str, Any]:
    rule = rules.get(action)
    check(isinstance(rule, dict) and rule.get("operation") == "toggle", "oracle rule")
    changed = rule.get("changed_variables")
    check(isinstance(changed, list) and changed and all(item in domains for item in changed), "oracle changed_variables")
    after = dict(before)
    for item in changed:
        after[item] = flip(domains, item, after[item])
    return {"action_id": action, "outcome": rule.get("outcome"), "obs_after": after}


def actual_changed(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {key for key in before if before[key] != after[key]}


def commitments(snapshot: dict[str, Any]) -> bool:
    valid = lambda item: isinstance(item, str) and len(item) == 64 and set(item) <= set("0123456789abcdef")
    return valid(snapshot.get("chain_commitment")) and valid(snapshot.get("state_commitment")) and all(
        all(valid(experience.get(name)) for name in ("prediction_commitment", "previous_entry_commitment", "entry_commitment", "pre_chain_commitment", "post_chain_commitment"))
        for experience in snapshot.get("experiences", []))


def replay_e3(snapshot: dict[str, Any], domains: dict[str, list[Any]], actions: dict[str, dict[str, Any]], prior: dict[str, Any]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str]]:
    targets = {action["id"]: action["intervention"]["variable"] for action in actions.values()}
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for action_id, cause in targets.items():
        for effect in set(domains) - {cause}:
            expected[(cause, effect, "cochanges")] = {"supports": [], "contradicts": []}
    for experience in snapshot["experiences"]:
        cause = experience["action"]["intervention"]["variable"]
        if experience["before"][cause] == experience["after"][cause]:
            continue
        for effect in set(domains) - {cause}:
            key = (cause, effect, "cochanges")
            bucket = "supports" if experience["before"][effect] != experience["after"][effect] else "contradicts"
            expected[key][bucket].append(experience["id"])
    actual = {(item["cause"], item["effect"], item["relation"]): item for item in snapshot["hypotheses"]}
    check(set(actual) == set(expected) and len(actual) == 8, "E3 exact eight-edge set")
    statuses: dict[str, str] = {}
    for key, evidence in expected.items():
        item = actual[key]
        supports, contradicts = len(evidence["supports"]), len(evidence["contradicts"])
        status = "refuted" if contradicts >= 2 else "supported" if supports >= 2 and not contradicts else "proposed"
        check(item["evidence"] == {"supports": supports, "contradicts": contradicts} and item["supporting_evidence"] == evidence["supports"] and item["contradicting_evidence"] == evidence["contradicts"] and item["status"] == status, "E3 replay mismatch")
        expected_provenance = "atlas" if key == (prior["cause"], prior["effect"], "cochanges") else "experience"
        check(item["provenance"] == expected_provenance, "E3 provenance")
        if expected_provenance == "atlas":
            check(item["source_ref"] == prior["id"], "E3 prior retained")
        else:
            check(item["source_ref"] is None, "E3 candidate source")
        statuses["|".join(key)] = status
    return actual, statuses


def check_oracle_crosscheck(oracle: dict[str, Any], true_effect: str, prior: dict[str, Any], control: str, held_diff: set[str], variables: set[str]) -> None:
    expected = oracle.get("expected_after_training", {})
    check(set(map(tuple, expected.get("supported_edges", []))) == {(prior["cause"], true_effect, "cochanges")}, "oracle supported cross-check")
    check(set(map(tuple, expected.get("refuted_edges", []))) == {(prior["cause"], prior["effect"], "cochanges")}, "oracle refuted cross-check")
    held = oracle.get("expected_held_out_prediction", {})
    check({key: set(held.get(key, [])) for key in ("changed", "unchanged", "unknown")} == {"changed": held_diff, "unchanged": variables - held_diff, "unknown": set()}, "oracle held-out cross-check")


def run(pilot: Path) -> tuple[dict[str, bool], list[dict[str, Any]], dict[str, Any]]:
    gates, events, summary = {f"E{number}": False for number in range(6)}, [], {
        "training_steps": 0,
        "experience_count": 0,
        "isolation": {"filesystem_isolated": True, "pid_isolated": True, "network_isolated": False},
    }
    canary_path: Path | None = None
    try:
        public, oracle_file = pilot / "public" / "atlas.yaml", pilot / "examiner-private" / "oracle.yaml"
        atlas_bytes, atlas, oracle = public.read_bytes(), yaml_object(public), yaml_object(oracle_file)
        domains, actions, prior = profile(atlas)
        e0_codes = validate_e0(atlas_bytes, atlas)
        gates["E0"], summary["e0_fixture_codes"] = True, e0_codes
        info = oracle.get("public_atlas", {})
        check(digest(atlas_bytes) == info.get("file_sha256") and create(atlas_bytes).snapshot().as_mapping()["atlas_id"] == info.get("engine_atlas_id"), "public atlas integrity")
        rules, schedule, initial, max_steps = oracle.get("action_rules"), oracle.get("training_schedule"), oracle.get("initial_observation"), oracle.get("limits", {}).get("max_steps")
        check(isinstance(rules, dict) and isinstance(schedule, list) and isinstance(initial, dict) and isinstance(max_steps, int), "oracle schedule")
        check(len(schedule) + 1 <= max_steps and all(action in actions for action in schedule) and oracle.get("restart", {}).get("after_training_step") == len(schedule), "step budget/restart")
        cause_action = next(key for key, value in actions.items() if value["intervention"]["variable"] == prior["cause"])
        control_action = next(key for key in actions if key != cause_action)
        control = actions[control_action]["intervention"]["variable"]
        check(schedule and schedule[0] == cause_action, "E1 starts with prior cause")
        # The first world response independently identifies B, then all remaining role labels.
        first_step = step(dict(initial), cause_action, rules, domains)
        first_diff = actual_changed(initial, first_step["obs_after"])
        true_candidates = first_diff - {prior["cause"]}
        check(len(true_candidates) == 1 and prior["effect"] not in true_candidates and control not in first_diff, "world role separation")
        true_effect = next(iter(true_candidates))
        decoys = set(domains) - {prior["cause"], prior["effect"], control, true_effect}
        check(len(decoys) == 1, "world decoy")
        summary["roles_derived"] = {"cause": prior["cause"], "false": prior["effect"], "control": control, "true": true_effect, "decoy": next(iter(decoys))}
        artifacts = pilot / "artifacts"
        descriptor, raw_canary = tempfile.mkstemp(prefix="examiner-private-canary-", suffix=".secret")
        canary_path = Path(raw_canary)
        with os.fdopen(descriptor, "wb") as canary_file:
            canary_file.write(os.urandom(32))
            canary_file.flush()
            os.fsync(canary_file.fileno())
        check(canary_path.read_bytes(), "examiner canary creation")
        training_path, final_path = artifacts / "training-shelf.json", artifacts / "final-shelf.json"
        learner = Learner(atlas_bytes, training_path, None, oracle_file.resolve(), canary_path.resolve(), events, "training")
        check(learner.call({"command": "bind", "observation": initial}).get("ok") is True, "bind")
        observation, training_records = dict(initial), []
        for number, action in enumerate(schedule):
            pre = prediction(learner.call({"command": "predict", "action_id": action}), set(domains))
            append(events, "training", "prediction_recorded", action_id=action, prediction=pre)
            response = step(observation, action, rules, domains)
            append(events, "training", "step", action_id=action, response=response)
            if number == 0:
                assert_e1_prediction(pre, prior["effect"], true_effect)
            actual = actual_changed(observation, response["obs_after"])
            append(events, "training", "score", action_id=action, actual_changed=sorted(actual))
            check(learner.call({"command": "accept", "prediction_id": pre["id"], "step_response": response}).get("ok") is True, "accept")
            training_records.append((dict(observation), response))
            observation = response["obs_after"]
        training_bytes = learner.save_to()
        learner.close()
        training = json.loads(training_bytes)
        check(len(training["experiences"]) == len(schedule) and commitments(training), "training shelf")
        for index, event in enumerate(events):
            if event["event"] == "step": check(index > 0 and events[index - 1]["event"] == "prediction_recorded", "E2 prediction-before-step")
        gates["E1"], gates["E2"] = True, True
        hypotheses, statuses = replay_e3(training, domains, actions, prior)
        for target in (prior["cause"], control):
            states = {entry["before"][target] for entry in training["experiences"] if entry["action"]["intervention"]["variable"] == target}
            check(states == set(domains[target]), "E3 opposite cause states")
        gates["E3"], summary["e3_statuses"] = True, statuses
        atomic_write(training_path, training_bytes)
        restarted = Learner(atlas_bytes, final_path, training_bytes, oracle_file.resolve(), canary_path.resolve(), events, "restart")
        unbound = restarted.call({"command": "predict", "action_id": cause_action})
        check(unbound == {"ok": False, "code": "OBSERVATION_UNBOUND"}, "E4 unbound")
        detached = Learner(atlas_bytes, artifacts / ".detached.json", training_bytes, oracle_file.resolve(), canary_path.resolve(), events, "detached")
        detached_bytes = detached.save_to(); detached.close()
        (artifacts / ".detached.json").unlink(missing_ok=True)
        check(detached_bytes == training_bytes, "E4 detached byte identity")
        held = oracle.get("held_out", {})
        before, held_action = held.get("initial_observation"), held.get("action_id")
        check(isinstance(before, dict) and held_action in actions, "held-out shape")
        check(restarted.call({"command": "bind", "observation": before}).get("ok") is True, "held-out bind")
        held_prediction = prediction(restarted.call({"command": "predict", "action_id": held_action}), set(domains))
        append(events, "held_out", "prediction_recorded", action_id=held_action, prediction=held_prediction)
        held_response = step(dict(before), held_action, rules, domains)
        held_diff = actual_changed(before, held_response["obs_after"])
        expected_partition = {"changed": held_diff, "unchanged": set(domains) - held_diff, "unknown": set()}
        check({key: set(held_prediction[key]) for key in expected_partition} == expected_partition, "E5 independently derived prediction")
        check_oracle_crosscheck(oracle, true_effect, prior, control, held_diff, set(domains))
        append(events, "held_out", "step", action_id=held_action, response=held_response)
        append(events, "held_out", "score", actual_changed=sorted(held_diff))
        check(restarted.call({"command": "accept", "prediction_id": held_prediction["id"], "step_response": held_response}).get("ok") is True, "held-out accept")
        final_bytes = restarted.save_to(); restarted.close()
        final = json.loads(final_bytes)
        check(commitments(final) and final["atlas_id"] == info["engine_atlas_id"] and final["experiences"][:len(schedule)] == training["experiences"], "final shelf/load")
        gates["E4"], gates["E5"] = True, True
        atomic_write(final_path, final_bytes)
        summary.update({"training_steps": len(schedule), "experience_count": len(final["experiences"]), "max_steps": max_steps})
    except Exception as error:  # Fail closed, including malformed YAML/filesystem input.
        append(events, "examiner", "failure", error=type(error).__name__, message=str(error))
    finally:
        if canary_path is not None:
            canary_path.unlink(missing_ok=True)
    return gates, events, summary


def clean_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("transcript.jsonl", "result.json", "training-shelf.json", "final-shelf.json", ".detached.json"):
        (path / name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=Path(__file__).resolve().parents[1] / "pilot-001")
    pilot = parser.parse_args().pilot_root.resolve()
    artifacts = pilot / "artifacts"
    clean_artifacts(artifacts)
    gates, events, summary = run(pilot)
    transcript = b"".join(canon(event) + b"\n" for event in events)
    atomic_write(artifacts / "transcript.jsonl", transcript)
    hashes = {"transcript.jsonl": digest(transcript)}
    for name in ("training-shelf.json", "final-shelf.json"):
        target = artifacts / name
        if target.exists(): hashes[name] = digest(target.read_bytes())
    per_gate = {
        "E0": {"fixture_codes": summary.get("e0_fixture_codes", []), "permutation_and_empty_roundtrip": gates["E0"]},
        "E1": {"first_prediction_checked": gates["E1"]},
        "E2": {"experience_count": summary.get("experience_count", 0), "prediction_before_step": gates["E2"]},
        "E3": {"edge_statuses": summary.get("e3_statuses", {}), "exact_edge_count": len(summary.get("e3_statuses", {}))},
        "E4": {"restart_and_detached_load": gates["E4"]},
        "E5": {"held_out_independent_diff": gates["E5"]},
    }
    result = {"schema": "causal-01-result/v2", "verdict": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "evidence": {**summary, "per_gate": per_gate, "artifact_sha256": hashes}}
    atomic_write(artifacts / "result.json", canon(result) + b"\n")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
