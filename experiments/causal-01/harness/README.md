# CAUSAL-01 pilot harness

The examiner reads `pilot-001/examiner-private/oracle.yaml`; every learner is
run under mandatory `/usr/bin/bwrap` with a separate mount and PID namespace.
Network isolation is deliberately not claimed because this host policy rejects
`--unshare-net`; the transcript and result record `network_isolated: false`.
It sees a copied runtime, readonly interpreter libraries, a writable
temporary shelf, and JSONL stdin/stdout only. Atlas/snapshot bytes travel in a
bootstrap JSONL record; normal lifecycle messages are `bind`, `predict`,
`accept`, and `save`.

The harness creates real examiner-readable oracle/canary/module fixtures and
runs a parent-controlled probe in the production-equivalent bwrap filesystem/PID
namespace. It denies direct oracle/canary reads, `/proc/self/root` and
`/proc/1/root` variants, and private-module imports before any learner starts.
Any probe failure fails closed. There is no unsandboxed fallback: missing or
failed `bwrap` produces a FAIL result.

Run the pilot from the repository root:

```bash
python3 experiments/causal-01/harness/harness.py
```

It writes only these examiner artifacts in `experiments/causal-01/pilot-001/artifacts/`:
`transcript.jsonl`, `result.json`, `training-shelf.json`, and `final-shelf.json`.
The oracle is never copied there.  The command exits successfully only when
E0 through E5 pass.

Run harness tests:

```bash
python3 -m unittest discover -s experiments/causal-01/harness/tests -t .
```
