"""Эталонный модуль мира для conformance.py: двухосевая ОС (из exp9).

Ровно та форма, которую обязан порождать worldkit ветки:
GLUE + EXAMS, никакого кода верхнего уровня.
"""

from world import World

QUARANTINE = [
    {("owner", "user"), ("dir", "/etc")},
    {("ftype", "text"), ("exec_bit", 1)},
]
FILE_ACTIONS = ("read", "write", "delete", "chmod", "exec", "copy")


def _ctx(obs, action, target, kw):
    if action in FILE_ACTIONS:
        f = obs["files"][target]
        return frozenset(("x" if k == "exec_bit" else k, v)
                         for k, v in f.items() if k != "name")
    if action in ("kill", "start"):
        return frozenset({("status", obs["procs"][target]["status"])})
    return frozenset({("dir", kw.get("dir")), ("ftype", kw.get("ftype")),
                      ("x", int(kw.get("exec_bit", 0)))})


GLUE = dict(
    object_actions=FILE_ACTIONS, ctx_fn=_ctx,
    truth_fn=lambda w, t: (
        f"{w.files[t].owner}-{'sealed' if w.files[t].sealed else 'open'}"
        if t in w.files else None),
    entities_fn=lambda obs: {n: p["status"]
                             for n, p in obs["procs"].items()},
    factory=lambda ep: World(seed=42 + ep, quarantine=QUARANTINE,
                             hidden=("owner", "sealed")),
    link_truth=lambda ep, o, e: (
        World(seed=42 + ep, quarantine=QUARANTINE).procs[e].config == o),
)

EXAMS = [("права", "chmod", "write", True),
         ("печать", "read", "copy", False)]
