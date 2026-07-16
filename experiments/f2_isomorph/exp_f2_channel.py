# Пригвоздить канал пробоя F2 на donor@600: рестарт-1 (rng-раздача эмиссий
# по sorted(res_a), organism2:549) vs порядок float-суммирования в рестарте 0.
# Метод: monkeypatch fit_axis с restarts=1 (остаётся только кластерный старт).

import sys, json
sys.path.insert(0, "/Users/kyrylo/Documents/Pers/ccc")
import os
os.chdir("/Users/kyrylo/Documents/Pers/ccc")

from worldkit import load_spec, make_glue
import organism2, fastpath, koopman

SCRATCH = os.path.dirname(os.path.abspath(__file__))
ORIG = "worlds/comp/donor.yaml"
ISO = os.path.join(SCRATCH, "donor_iso.yaml")

_orig_fit = organism2.fit_axis
def fit1(actions, timelines, k, restarts=2, iters=5):
    return _orig_fit(actions, timelines, k, restarts=1, iters=iters)


def run(path, budget):
    glue = make_glue(load_spec(path))
    sig, fit = fastpath.quick_signature(glue, budget)
    return sig


for label, patch in (("restarts=2 (как есть)", _orig_fit),
                     ("restarts=1 (только кластерный старт)", fit1)):
    fastpath.fit_axis = patch
    sigA = run(ORIG, 600)
    sigB = run(ISO, 600)
    print(f"{label}: бит-в-бит={sigA == sigB}")
    if sigA != sigB:
        print(f"  noise A: {sigA['noise']}")
        print(f"  noise B: {sigB['noise']}")
