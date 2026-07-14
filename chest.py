"""chest — сундук: выписать выученное организмом в файл и поднять обратно.

Шаг «пережить собственную смерть». Один сундук = снимок ОДНОЙ прожитой
жизни (один мир): hist, records, оси, R, problem_actions, steps. Правила
(mem) не хранятся — они выводимы из records+axes при подъёме (_relabel),
так что сундук держит источник, а не производное.

НЕ трогает organism2.py (заморозка цела) — только читает его поля.
НЕ переносит библиотеку в другой мир (объекты/слова там чужие) — это
следующий шаг, recognition. Здесь только верный снимок + продолжение жизни.

Формат — читаемый JSON. Кортежи/множества/оси помечены явными тегами,
чтобы подъём был точным, а файл — открываемым глазами.
"""

from __future__ import annotations

import json
from collections import defaultdict

from organism2 import Axis, Organism

# поля оси, которые несут знание (кэш _tpow не храним — пересчитается)
_AXIS_FIELDS = ("actions", "k", "res_a", "emis", "haz", "tgt", "pi",
                "n_obs", "n_pairs", "score", "pen_dyn")


# ---------------------------------------------------------------- кодек

def enc(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Axis):
        return {"__axis__": {f: enc(getattr(x, f)) for f in _AXIS_FIELDS}}
    if isinstance(x, tuple):
        return {"__tuple__": [enc(v) for v in x]}
    if isinstance(x, frozenset):
        return {"__frozenset__": [enc(v) for v in x]}
    if isinstance(x, set):
        return {"__set__": [enc(v) for v in x]}
    if isinstance(x, dict):
        if all(isinstance(k, str) for k in x):
            return {k: enc(v) for k, v in x.items()}       # обычный объект
        return {"__dict__": [[enc(k), enc(v)] for k, v in x.items()]}
    if isinstance(x, list):
        return [enc(v) for v in x]
    raise TypeError(f"нечего кодировать: {type(x)}")


def dec(x):
    if isinstance(x, list):
        return [dec(v) for v in x]
    if not isinstance(x, dict):
        return x
    if "__tuple__" in x:
        return tuple(dec(v) for v in x["__tuple__"])
    if "__frozenset__" in x:
        return frozenset(dec(v) for v in x["__frozenset__"])
    if "__set__" in x:
        return set(dec(v) for v in x["__set__"])
    if "__dict__" in x:
        return {dec(k): dec(v) for k, v in x["__dict__"]}
    if "__axis__" in x:
        d = {f: dec(v) for f, v in x["__axis__"].items()}   # декодируем ВСЁ
        ax = Axis(d["actions"], d["k"],
                  {a: list(rs) for a, rs in d["res_a"].items()})
        ax.emis = d["emis"]
        ax.haz = list(d["haz"])
        ax.tgt = [list(row) for row in d["tgt"]]
        ax.pi = list(d["pi"])
        ax.n_obs, ax.n_pairs = d["n_obs"], d["n_pairs"]
        ax.score, ax.pen_dyn = d["score"], d["pen_dyn"]
        return ax
    return {k: dec(v) for k, v in x.items()}


# ---------------------------------------------------------------- API

def dump(org, path):
    """Снимок выученного организмом → JSON-файл."""
    state = {
        "curious": org.curious,
        "steps": org.steps,
        "records": org.records,
        "hist": {o: dict(by) for o, by in org.hist.items()},
        "axes": org.axes,
        "R": org.R,
        "problem_actions": org.problem_actions,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(enc(state), fh, ensure_ascii=False, indent=1)


def load(path, glue):
    """Поднять организм из сундука в мир glue. Правила выводятся заново
    из records+axes — источник хранился, производное восстанавливается."""
    with open(path, encoding="utf-8") as fh:
        st = dec(json.load(fh))
    org = Organism(st["curious"], glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue.get("entities_fn"))
    org.steps = st["steps"]
    org.records = st["records"]
    org.hist = defaultdict(lambda: defaultdict(list))
    for o, by in st["hist"].items():
        for a, seq in by.items():
            org.hist[o][a] = [tuple(p) for p in seq]
    org.axes = st["axes"]
    org.R = st["R"]
    org.problem_actions = st["problem_actions"]
    org.mem = org._relabel(org.axes)   # правила — из источника, тождественно
    org._bel = {}
    return org
