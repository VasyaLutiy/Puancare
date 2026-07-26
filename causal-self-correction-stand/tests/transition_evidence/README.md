# Transition Evidence test suite

Run the full isolated regression suite from the repository root:

```bash
python -m unittest discover -s tests -t .
```

`unit/` covers the public schema/runtime error codes and domain/persistence
invariants. `property/` uses a deterministic seed (`20260726`) across the
required fixture cardinality ranges. `integration/` drives the public
`EnvironmentPort` runner boundary, including detached process reload/rebind.
