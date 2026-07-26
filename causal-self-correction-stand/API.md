# Public Python API

```python
create(atlas_bytes: bytes) -> TransitionEvidenceEngine
load(snapshot_path: str | os.PathLike[str], atlas_bytes: bytes) -> TransitionEvidenceEngine

engine.bind_observation(observation: Mapping[str, JsonScalar]) -> None
engine.predict(action_id: str) -> Prediction
engine.accept_transition(prediction_id: str, step_response: StepResponse) -> int
engine.snapshot() -> CanonicalState
engine.save(snapshot_path: str | os.PathLike[str]) -> None
```

`StepResponse` is exactly `{"action_id": str, "outcome": JsonScalar,
"obs_after": Observation}`. Observations contain exactly the atlas's
observable variable IDs and values from their declared domains.

`accept_transition` accepts only a pending prediction created in the same
bound state and for the same action. It computes and validates the next shelf
locally, then makes one commit; on failure the shelf and pending prediction are
unchanged, so it can be retried. It returns the next positive monotonic
experience ID. A successful acceptance invalidates all pending predictions.

Evidence is action-target scoped. When the action target changes, the engine
creates missing candidates only from that target and applies evidence only to
hypotheses whose cause is that target. Incidental source-variable changes do
not receive evidence. When the target does not change, an experience/version
and current observation are still recorded, but there are no candidates or
evidence updates.

The optional atlas policy `proposed_prediction` accepts `"unknown"`
(default) or `"changed"`. Supported effects predict changed and refuted
effects predict unchanged. Proposed/conflicted effects predict unknown by
default, or changed under the `changed` policy. The resolved policy value is
part of the atlas identity.

`EnvironmentPort` has `observe()`, `action_space()`, and `step(action_id)`.
`run_schedule(engine, port, schedule=None)` validates raw observation, action
space, and every explicit schedule item before calling `bind_observation`; a
preflight error leaves shelf state and existing pending predictions untouched.

## Persistence and integrity

Every new `transition-shelf/v1` snapshot has shelf `chain_commitment` and
`state_commitment` fields. Every experience has `prediction_commitment`,
`previous_entry_commitment`, `entry_commitment`, `pre_chain_commitment`, and
`post_chain_commitment`, plus required boolean `informative`; commitment fields
are 64-character lowercase SHA-256 digests. `informative` records whether the
action target changed and is included in the entry and final-state commitments,
not in the prediction commitment. The first predecessor is an atlas-bound
deterministic entry-genesis digest, never
`null`. `load()` verifies canonical shape, atlas/action identity, ordering and
these links/hashes only; it does not replay relation evaluation, prediction,
candidate creation, or evidence application. It uses the recorded `informative`
value (rather than recomputing a transition diff) when structurally validating
evidence references. A loaded engine remains unbound until `bind_observation()`
is called. Snapshots created before this required field are intentionally invalid
under the strict current `transition-shelf/v1` format.

These are unkeyed integrity commitments, not authentication. A writer able to
replace the entire snapshot can recompute them; use an external signature or
trusted-key HMAC where adversarial tampering matters.

## Errors

All public engine errors inherit `TransitionEvidenceError` and carry `code`
and `path`. Required schema/runtime codes include `BAD_POLICY`,
`UNKNOWN_FIELD`, `OBSERVATION_SHAPE`, `VALUE_OUT_OF_DOMAIN`,
`ACTION_NOT_AVAILABLE`, `OBSERVATION_UNBOUND`, `PREDICTION_REQUIRED`,
`PREDICTION_MISMATCH`, and `DUPLICATE_EXPERIENCE`. Snapshot failures use
`SNAPSHOT_SHAPE`, `SNAPSHOT_INVARIANT`, or `ATLAS_MISMATCH`.

Snapshots are relation-agnostic at load time. Atlas input still admits only
registered relations; a stored relation the runtime cannot evaluate is retained
until a later evidence update; that update fails with controlled
`UNKNOWN_RELATION` before commit, not an uncontrolled lookup.
