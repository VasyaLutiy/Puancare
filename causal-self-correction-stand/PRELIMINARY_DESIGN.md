# Preliminary design: Transition Evidence Engine v1

## Scope and package layout

The implementation will be an isolated `transition_evidence/` Python package
(no imports from the existing application) and will have no runtime dependency
other than `PyYAML`:

```text
transition_evidence/
  __init__.py       # deliberately small public re-export surface
  api.py            # TransitionEvidenceEngine and create/load factories
  types.py          # frozen public value types and wire TypedDicts
  errors.py         # public exception hierarchy and stable codes
  atlas.py          # safe YAML loading, strict schema validation, Atlas model
  relations.py      # RelationEvaluator protocol and cochanges registry entry
  shelf.py          # atomic domain transition and invariant checks
  canonical.py      # typed JSON-scalar handling, canonical JSON and content IDs
  persistence.py    # snapshot decoding/encoding and atomic replacement write
  ports.py          # EnvironmentPort protocol and adapter-wire validation
  runner.py         # minimal deterministic environment runner
tests/
  transition_evidence/{unit,property,integration}/
```

`atlas`, `shelf`, and `persistence` are deliberately separate: relation
evaluation is injected through a registry, so adding a relation does not change
the shelf representation or snapshot format.

## Public API

`transition_evidence.__init__` will export only the following domain surface
(aliases such as `PathLike` are documented types, not additional factories):

```python
from os import PathLike
from typing import Mapping, Sequence, TypeAlias, TypedDict

JsonScalar: TypeAlias = None | bool | int | float | str
Observation: TypeAlias = Mapping[str, JsonScalar]
SnapshotPath: TypeAlias = str | PathLike[str]

class ActionDescription(TypedDict):
    id: str
    intervention: InterventionDescription

class StepResponse(TypedDict):
    action_id: str
    outcome: JsonScalar
    obs_after: Observation

@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    action_id: str
    changed: frozenset[str]
    unchanged: frozenset[str]
    unknown: frozenset[str]

@dataclass(frozen=True, slots=True)
class CanonicalState:
    """Read-only canonical shelf projection; `to_json_bytes()` is canonical UTF-8 JSON."""
    def to_json_bytes(self) -> bytes: ...
    def as_mapping(self) -> Mapping[str, object]: ...

def create(atlas_bytes: bytes) -> TransitionEvidenceEngine: ...
def load(snapshot_path: SnapshotPath, atlas_bytes: bytes) -> TransitionEvidenceEngine: ...

class TransitionEvidenceEngine:
    def bind_observation(self, observation: Observation) -> None: ...
    def predict(self, action_id: str) -> Prediction: ...
    def accept_transition(
        self, prediction_id: str, step_response: StepResponse
    ) -> int: ...
    def snapshot(self) -> CanonicalState: ...
    def save(self, snapshot_path: SnapshotPath) -> None: ...
```

All collections exposed by `Prediction` and `CanonicalState` are immutable;
`snapshot()` never exposes mutable shelf internals.  A prediction ID is a
canonical digest of the state version, complete bound `before` observation and
action ID.  It identifies that precise prediction, is held only as pending
runtime state, and is consumed after a successful acceptance.  Repeating
`predict` in the same unchanged state returns the same immutable value; after
`bind_observation` or a successful transition, prior pending predictions are
invalid.

The engine allocates positive, monotonically increasing integer experience IDs
itself.  The environment wire response has no experience ID, so a caller cannot
inject one.  `DUPLICATE_EXPERIENCE` is used when a snapshot contains a repeated
experience ID; a repeated/consumed prediction is rejected as
`PREDICTION_REQUIRED` (and an ID for a different action or `before` as
`PREDICTION_MISMATCH`).

## Errors and validation boundary

```python
class TransitionEvidenceError(Exception):
    code: str       # stable machine code
    path: str       # RFC 6901 JSON Pointer, with "" denoting the document root

class AtlasValidationError(TransitionEvidenceError): ...
class RuntimeValidationError(TransitionEvidenceError): ...
class SnapshotValidationError(TransitionEvidenceError): ...
class PersistenceError(TransitionEvidenceError): ...
```

The required schema and runtime codes are preserved verbatim:
`SCHEMA_VERSION`, `DUPLICATE_ID`, `BAD_DOMAIN`, `BAD_ACTION_TARGET`,
`BAD_HYPOTHESIS_ENDPOINT`, `DUPLICATE_HYPOTHESIS`, `UNKNOWN_RELATION`,
`UNKNOWN_FIELD`, `INITIAL_STATE_FORBIDDEN`, `BAD_POLICY`, `CYCLIC_YAML`,
`OBSERVATION_SHAPE`, `VALUE_OUT_OF_DOMAIN`, `ACTION_NOT_AVAILABLE`,
`OBSERVATION_UNBOUND`, `PREDICTION_REQUIRED`, `PREDICTION_MISMATCH`, and
`DUPLICATE_EXPERIENCE`.  Invalid/unconstructable YAML and invalid/foreign
snapshots use additional documented codes (`UNSAFE_YAML`, `SNAPSHOT_SHAPE`,
`SNAPSHOT_INVARIANT`, `ATLAS_MISMATCH`, `PERSISTENCE_WRITE`) with the same
`code`/`path` fields.  Message wording is intentionally non-contractual.

YAML enters only through `yaml.safe_load`, followed by a recursive check that
rejects aliases producing a cyclic object graph and rejects every non-JSON
scalar.  IDs and operation strings are opaque strings; their contents,
lexical form, and source order are never given semantic meaning.  JSON numbers
must be finite.  Domain membership and change detection use canonical JSON
tokens, rather than Python equality, so JSON types such as `true` and `1` are
not accidentally conflated.

## Relations and atomic shelf update

`relations.py` defines:

```python
class RelationEvaluator(Protocol):
    name: str
    def evaluate(self, before: Observation, after: Observation,
                 cause: str, effect: str) -> EvidenceDelta: ...

RelationRegistry: Mapping[str, RelationEvaluator]
```

The v1 registry is fixed at construction to `{"cochanges": CochangesEvaluator()}`
and atlas validation rejects any other name.  Its evaluator emits support when
both values change, contradiction when only the cause changes, and no delta
when the cause does not change.  It neither sees action IDs nor interprets
values.

On a successful `accept_transition`, the engine validates the exact response
shape and its action ID before mutation.  A transition is informative iff the
declared target of its action changed.  For an informative transition it performs one copy-on-write commit: materialise
all missing `target -> every-other-observable / cochanges` candidates (reusing
an atlas hypothesis with the same content ID), then evaluate the transition
exactly once only for hypotheses whose *cause is that action target*. Incidental
changes to other causes never add evidence.  It appends evidence references,
recomputes statuses, appends one experience, sets current observation to
`after`, and increments version once.  A non-informative transition (the target
is unchanged) still appends one experience and updates current observation and
version, but makes no candidates and changes no hypothesis counters or evidence
links.

Status is derived, never independently mutable: `supported` requires enough
supports and zero contradictions; `refuted` requires enough contradictions;
otherwise it is `proposed`.  The specification's “conflict” is represented by
the latter case with both counters nonzero (there is no fourth persisted
status), so it remains in prediction `unknown` until refutation.

An exception at any validation, relation, or append step discards the proposed
copy; no version, experience, evidence counter, reference, current observation
or pending-prediction state changes partially.

## Prediction, environment port, and runner

```python
@runtime_checkable
class EnvironmentPort(Protocol):
    def observe(self) -> Observation: ...
    def action_space(self) -> Sequence[ActionDescription]: ...
    def step(self, action_id: str) -> StepResponse: ...

def run_schedule(
    engine: TransitionEvidenceEngine,
    environment: EnvironmentPort,
    schedule: Sequence[str] | None = None,
) -> tuple[int, ...]: ...
```

Before it changes engine state, the runner obtains raw `observe()` and raw
`action_space()` values, validates both purely against the atlas, and validates
every explicit schedule ID against that action space. Only then does it bind the
normalised observation and perform `predict(action_id) -> environment.step(action_id)
-> accept_transition(...)`. Thus a failing observation, action space, or explicit
schedule preserves the shelf and all pre-existing pending predictions. With
`schedule=None`, it traverses currently available IDs in lexicographic order
only for reproducibility. Adapter failures after this preflight are not rolled
back beyond the normal atomicity of `accept_transition`.

`predict` requires a currently bound observation.  Its three sets are frozen,
disjoint, and partition all observable variable IDs: the action target plus
supported effects are `changed`, refuted effects are `unchanged`, and all remaining variables are `unknown`. The atlas policy optional
`proposed_prediction` resolves to `"unknown"` when omitted; when explicitly
`"changed"`, proposed/conflicted effects are included in `changed` instead.

## Canonical state and persistence

Canonical JSON is UTF-8 with `sort_keys=True` and compact separators.  Atlas
identity is computed after sorting variables/actions by ID and hypotheses by
their `sha256(canonical_json([cause, effect, relation]))` content ID.  The
same content ID is used for atlas and experience-created hypotheses; atlas
local `id` becomes `source_ref` only.  Snapshot hypotheses use content-ID
order; experiences and each evidence-reference list use numeric experience-ID
order.  No timestamps, paths, PIDs, platform fields, source ordering, or
pending prediction state is serialized.

`save()` serializes `snapshot().to_json_bytes()` to a same-directory temporary
file, flushes and fsyncs it, then uses `os.replace`. `transition-shelf/v1` now
requires SHA-256 commitment fields on the shelf (`chain_commitment`,
`state_commitment`) and every experience (`prediction_commitment`,
`previous_entry_commitment`, `entry_commitment`, `pre_chain_commitment`,
`post_chain_commitment`, and required boolean `informative`). `load()` validates the atlas identity, canonical
structure, exact action descriptions, IDs, commitment links, and state hash;
it never reconstructs predictions, evaluates relations, materialises candidates,
or reapplies evidence. It retains the persisted last
`current_observation` in the snapshot but marks it runtime-stale/unbound;
therefore `predict` fails with `OBSERVATION_UNBOUND` until the new environment
is explicitly rebound.  Save/load without a new transition consequently
preserves byte-identical canonical JSON.

## Tests and one command

Tests will use the standard-library `unittest` runner plus deterministic
seeded generators for property/metamorphic cases, so no test dependency is
needed beyond PyYAML.  The three directories above cover every prescribed
validation code, relation/status/candidate/prediction/persistence atomicity
unit case; generated 2–9-variable, 1–4-action fixtures for rename/permutation
and save-load properties; and a port-only fake-environment lifecycle including
restart/rebind and mixed outcomes.  At least three structurally distinct
handwritten atlases will accompany the generated cases.

The documented single local command will be:

```bash
python3 -m unittest discover -s tests -t .
```

## Integrity commitments

All commitment inputs use canonical UTF-8 JSON (sorted keys and compact
separators) and unkeyed SHA-256. Let `A` be the atlas ID. The initial chain is
`sha256({"atlas_id": A, "kind": "transition-shelf/genesis/v1"})`; the first
entry predecessor is separately
`sha256({"atlas_id": A, "kind": "transition-shelf/entry-genesis/v1"})`.
A prediction commits `{atlas_id, pre_chain_commitment, prediction_id, before,
action_id, changed, unchanged, unknown}`. An entry commits `{id, before,
prediction, action, outcome, after, informative, comparison,
prediction_commitment, previous_entry_commitment, pre_chain_commitment}`. Its
post-chain commitment
is `sha256([pre_chain_commitment, entry_commitment])`; every later entry uses
the previous entry hash as its predecessor. Finally `state_commitment` hashes
the canonical shelf projection (including chain, hypotheses, and experiences,
but excluding `state_commitment` itself). These commitments detect accidental
corruption and inconsistent partial edits. They are not authentication: anyone
who can rewrite the entire snapshot can recompute unkeyed SHA-256 values. A
trusted external signing or HMAC key is required for adversarial tamper or
backdating protection.

## Persisted relation compatibility

Atlas ingress accepts only registered v1 relations. Snapshot loading is relation
agnostic: it checks a relation is a string and that its committed content ID and
provenance are structurally coherent, but never imports or invokes a registry or
evaluator. A relation unknown to the runtime is preserved on load; a later attempt
to evaluate it raises controlled `UNKNOWN_RELATION` before the atomic commit,
rather than an uncontrolled lookup error.
