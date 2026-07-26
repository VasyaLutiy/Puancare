# Transition Evidence Engine v1

An isolated, deterministic Python component for collecting evidence about
observable state transitions.  It reads a declarative `transition-atlas/v1`,
requires an explicit `predict → step → accept_transition` lifecycle, and
persists canonical `transition-shelf/v1` snapshots.

The only runtime dependency is PyYAML.  There are no network calls, databases,
or dependencies on the surrounding project.

## Install and check

```bash
python3 -m pip install -e .
python3 -m transition_evidence validate examples/switch-atlas.yaml
python3 -m unittest discover -s tests -t .
```

The final command is the documented complete test-suite command.  The package
itself can also be checked without tests with the first `validate` command.

## Lifecycle API

```python
from transition_evidence import create, load

engine = create(open("atlas.yaml", "rb").read())
engine.bind_observation({"power": False, "light": False, "alarm": False})
prediction = engine.predict("set_power")

# The environment is external; the engine does not call it itself.
experience_id = engine.accept_transition(prediction.id, {
    "action_id": "set_power",
    "outcome": "ok",
    "obs_after": {"power": True, "light": True, "alarm": False},
})
engine.save("shelf.json")

restarted = load("shelf.json", open("atlas.yaml", "rb").read())
# A loaded engine is deliberately unbound; bind a fresh environment observation
# before its next prediction.
```

`Prediction.changed`, `.unchanged`, and `.unknown` are `frozenset`s and
partition every observable variable.  `snapshot()` returns an immutable
`CanonicalState`; `to_json_bytes()` is canonical UTF-8 JSON.

Evidence is limited to the action target: only hypotheses caused by the target
receive an evidence delta, even if another observed variable changed
incidentally. If the target itself is unchanged, the engine still records an
experience/version/current observation but makes no candidates or evidence
changes. Optional atlas policy `proposed_prediction` is `unknown` by default;
set it to `changed` to forecast proposed/conflicted effects as changed.

Snapshots include a strict SHA-256 commitment chain for each prediction and
experience plus committed final shelf state. Each experience also records a
required boolean `informative` (whether its action target changed); it is bound
into the entry/state commitments. Loading verifies those structural and
cryptographic-consistency links without replaying learning, and old snapshots
without this field are intentionally rejected. The hashes are
unkeyed, so they detect accidental corruption but do not prevent a full-snapshot
rewrite by an attacker who can recompute them; see
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

`TransitionEvidenceError` and its subclasses expose a stable `code` and RFC
6901 JSON-Pointer `path`.  See [SCHEMAS.md](SCHEMAS.md) for input/output
schemas and [API.md](API.md) for the complete public interface.

## Environment integration

Implement `EnvironmentPort` with `observe()`, `action_space()`, and
`step(action_id)`.  `run_schedule(engine, environment, schedule)` binds an
observation and performs the required lifecycle.  It validates raw observation, action-space descriptions, and an entire explicit
schedule before binding the engine, so a preflight failure preserves state and
pending predictions. If no schedule is supplied, it uses stable lexicographic
action-ID order solely for reproducibility.

## Reproducible demo

```bash
python3 demo/run_demo.py
```

It recreates `demo/demo.jsonl` and `demo/demo-shelf.json` from
`examples/switch-atlas.yaml`.  The JSONL contains one canonical event record
per accepted external transition.
