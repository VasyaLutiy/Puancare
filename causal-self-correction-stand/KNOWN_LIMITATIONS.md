# Known limitations and intentional v1 boundaries

- v1 implements only the `cochanges` relation.  The registry is extensible,
  but accepting new relation names requires a schema/version decision.
- The engine accepts JSON-scalar outcomes to ensure persistence remains
  canonical JSON.  Structured external outcome payloads need an adapter or a
  future schema revision.
- The bundled runner executes an explicit schedule or a single lexicographic
  pass over the current action space.  It deliberately contains no learning or
  action-selection policy.
- Source hypotheses whose endpoints are non-observable are retained (the atlas
  rule only requires endpoints to exist), but cannot accrue evidence because
  observations contain observable variables only.
- `current_observation` is retained in snapshots for auditability, but a loaded
  engine is always runtime-unbound and must be rebound to a live environment.

- Snapshot SHA-256 commitments are deliberately unkeyed. They provide
  corruption/self-consistency detection, not origin authentication: an attacker
  able to rewrite the whole snapshot can recalculate every digest. Deployments
  requiring tamper or backdating resistance must wrap snapshots in a trusted
  signature or key-held HMAC outside this v1 format.
- `load()` intentionally verifies committed structure rather than replaying
  causal learning. It cannot establish that a correctly re-committed snapshot
  was produced by a particular external environment.

- The snapshot loader is deliberately relation-agnostic and does not replay an
  evaluator. Atlas input remains restricted to registered v1 relations. A
  snapshot relation unknown to the current runtime is retained, but a later
  evidence update fails atomically with controlled `UNKNOWN_RELATION`.
- The required `experiences[].informative` field is a strict current-v1 format
  requirement. Older snapshots lacking it must be migrated externally; loader
  does not derive it from before/after during loading.
