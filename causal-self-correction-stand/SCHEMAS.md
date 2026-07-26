# Schemas and canonicalization

## Atlas input

The YAML root has exactly `schema`, `variables`, `actions`, `hypotheses`, and
`policy`. `schema` is `transition-atlas/v1`. Variables, actions, and
hypotheses retain their v1 constraints: actions target observable variables and
hypotheses have distinct existing endpoints using `cochanges`/`atlas`.

Policy requires positive `support_min` and `refute_min` plus
`candidate_generation: all_observable_effects`. It may additionally contain
`proposed_prediction: changed|unknown`; omission resolves to `unknown`. An
invalid supplied value raises `BAD_POLICY` at `/policy/proposed_prediction`;
another policy key raises `UNKNOWN_FIELD`. The resolved field participates in
the canonical atlas digest.

YAML is loaded only with `yaml.safe_load`; unsafe tags, cyclic aliases,
non-JSON scalars, unknown fields, and `initial_state` are rejected.

## Shelf snapshot

`snapshot()` and `save()` emit canonical `transition-shelf/v1` JSON with
`schema`, `version`, `atlas_id`, `current_observation`, `hypotheses`,
`experiences`, `chain_commitment`, and `state_commitment`. Each experience
requires `id`, `before`, `prediction`, `action`, `outcome`, `after`, required
boolean `informative`, `comparison`, `prediction_commitment`, `previous_entry_commitment`,
`entry_commitment`, `pre_chain_commitment`, and `post_chain_commitment`. New
snapshots missing this field or any commitment are rejected. This is a strict
breaking revision within the current v1 snapshot label. Hypothesis IDs are
`sha256(canonical_json([cause, effect, relation]))`; source-local IDs are kept
only as `source_ref`.

Canonical JSON is UTF-8, sorted-key, compact-separator JSON. Variables/actions
are canonicalized by ID, hypotheses by content ID, and experiences/evidence
links by numeric experience ID. `load()` requires byte-for-byte canonical
encoding after verification. Snapshot writes use a same-directory temporary
file followed by `os.replace`.

## Commitment chain

All digests are SHA-256 of canonical JSON. Initial chain: `sha256({"atlas_id":
A, "kind": "transition-shelf/genesis/v1"})`; first entry predecessor:
`sha256({"atlas_id": A, "kind": "transition-shelf/entry-genesis/v1"})`.
The prediction payload is `{atlas_id, pre_chain_commitment, prediction_id,
before, action_id, changed, unchanged, unknown}`. The entry payload is `{id,
before, prediction, action, outcome, after, informative, comparison,
prediction_commitment, previous_entry_commitment, pre_chain_commitment}`.
`post_chain_commitment = sha256([pre_chain_commitment, entry_commitment])`;
subsequent `previous_entry_commitment` values equal the prior entry hash.
`state_commitment` hashes the entire canonical shelf projection excluding that
field itself.

An accepted transition always appends one experience and increments version
once. If the action target is unchanged, no candidate or evidence update is
made. A detached loaded engine retains its last observation only for audit and
must be rebound before `predict()`.

`informative` is exactly a JSON boolean recorded by acceptance when the action
target changed. Loader evidence-reference validation uses only the recorded
boolean and committed action target; it does not recompute differences or call
a relation evaluator. A snapshot may structurally contain an otherwise unknown
relation string; only atlas ingress applies the v1 relation allowlist. If a
later runtime transition would evaluate an unavailable stored relation, it
raises controlled `UNKNOWN_RELATION` before the atomic commit.
