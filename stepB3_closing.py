#!/usr/bin/env python3
"""
Step B3 — ЗАМЫКАЮЩИЙ тест: спасает ли атомная ось то, на чём пал 3b?

В 3b абстрактный h_ic-diff НЕ предсказал соблазнительность ошибки на cousin-парах
(partial|freq = +0.06, мёртв). Теперь у нас атомы. Ключевая величина:

    incl(A,B) = |A∩B| / |B|  = доля определяющих атомов B, которыми обладает A
              = насколько A "проходит" под определение B = соблазн сказать "A это B".

Гипотеза: frontier_asym (соблазн A→B минус B→A) коррелирует с incl(A,B)-incl(B,A),
И ВЫЖИВАЕТ под контролем частоты — там, где h_ic умер.
Сравниваем атомный предиктор vs старый h_ic-diff на ОДНИХ парах.
"""

import sys
import json
import os
import math
import random
from concurrent.futures import ThreadPoolExecutor

from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic, resolve
from step3_asymmetry import lemma_freq, spearman, partial_spearman
from step3b_azure_confirm import query_one
from utils_azure import AzureJSON

A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json")))
C = {k: set(v) for k, v in A["concepts"].items()}

# конкретные животные (не абстрактные узлы) — среди них почти нет is-a, все cousins
SPECIFIC = ["carp", "goldfish", "pike", "salmon", "shark", "tuna", "eel",
            "whale", "dolphin", "seal", "bat", "dog", "cat", "horse", "cow",
            "mouse", "elephant", "eagle", "penguin", "ostrich", "sparrow",
            "owl", "duck", "frog", "snake", "lizard", "turtle", "crocodile",
            "bee", "ant", "butterfly", "spider", "octopus", "crab"]


def incl(x, y):
    return len(C[x] & C[y]) / len(C[y]) if C[y] else 0.0


def sim(x, y):
    return len(C[x] & C[y]) / len(C[x] | C[y]) if (C[x] | C[y]) else 0.0


def main():
    ensure_wordnet()
    root, nodes, edges, parents = collect_subtree("animal.n.01")
    h_ic, _ = build_h_ic(root, nodes, parents)

    rng = random.Random(0)
    allp = [(a, b) for i, a in enumerate(SPECIFIC) for b in SPECIFIC[i + 1:]]
    rng.shuffle(allp)
    pairs = allp[:120]
    print(f"[data] cousin-пар среди понятий atoms.json: {len(pairs)}")

    az = AzureJSON()
    print("[azure] фронтир live (соблазн ошибки)...", file=sys.stderr)

    def work(p):
        a, b = p
        try:
            r = query_one(az, a, b)
            return a, b, float(r.get("a_is_b", 0)), float(r.get("b_is_a", 0))
        except Exception:
            return a, b, None, None

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for a, b, ab, ba in ex.map(work, pairs):
            if ab is None:
                continue
            sa, sb = resolve(a, nodes), resolve(b, nodes)
            hd = (h_ic[sa] - h_ic[sb]) if (sa and sb) else 0.0
            fa = math.log(lemma_freq(sa)) if sa else 0.0
            fb = math.log(lemma_freq(sb)) if sb else 0.0
            rows.append(dict(a=a, b=b, fasym=ab - ba, sal=max(ab, ba),
                             incl_asym=incl(a, b) - incl(b, a),
                             sim=sim(a, b), hdiff=hd, freq=fa - fb))

    Fr = [r["fasym"] for r in rows]
    IA = [r["incl_asym"] for r in rows]
    HD = [r["hdiff"] for r in rows]
    FQ = [r["freq"] for r in rows]
    SM = [r["sim"] for r in rows]
    SAL = [r["sal"] for r in rows]

    def signmatch(pred):
        nz = sum(1 for f in Fr if f != 0)
        m = sum(1 for p, f in zip(pred, Fr) if (p > 0) == (f > 0) and f != 0)
        return m / nz if nz else float("nan")

    print("=" * 72)
    print(f"  ЗАМЫКАЮЩИЙ ТЕСТ — N={len(rows)} cousin-пар, сигнал=фронтир соблазн-асимметрия")
    print("=" * 72)
    print("  Spearman с frontier_asym (направление соблазна):")
    print(f"    АТОМЫ incl-асимметрия : {spearman(IA, Fr):+.3f}")
    print(f"    старый h_ic-diff      : {spearman(HD, Fr):+.3f}   <- что было в 3b")
    print(f"    freq_diff             : {spearman(FQ, Fr):+.3f}")
    print(f"    sign-match АТОМЫ      : {signmatch(IA):.1%}  (h_ic было ~45%)")
    print("-" * 72)
    print("  *** КРИТЕРИЙ УБИЙСТВА (partial | freq) ***")
    print(f"    АТОМЫ incl-асим : {partial_spearman(IA, Fr, FQ):+.3f}")
    print(f"    старый h_ic     : {partial_spearman(HD, Fr, FQ):+.3f}   (в 3b было ~+0.06, мёртв)")
    pa = partial_spearman(IA, Fr, FQ)
    print("    -> АТОМЫ СПАСЛИ ✅" if abs(pa) >= 0.2 else
          ("    -> атомы лучше, но скромно" if abs(pa) >= 0.1 else "    -> тоже слабо ⚰️"))
    print("-" * 72)
    print(f"  Ось сходства как ГЕЙТ соблазна: Spearman(sim, salience) = {spearman(SM, SAL):+.3f}")
    print(f"    (высокий sim => есть соблазн вообще? проверка двухфакторной модели)")
    print("=" * 72)
    print("  Сильнейшие соблазны по фронтиру (где asym велик):")
    print(f"    {'A':9s} {'B':9s} {'fr_asym':>8s} {'incl_as':>8s} {'sim':>5s}")
    for r in sorted(rows, key=lambda x: -abs(x["fasym"]))[:10]:
        print(f"    {r['a']:9s} {r['b']:9s} {r['fasym']:8.0f} {r['incl_asym']:+8.2f} {r['sim']:5.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
