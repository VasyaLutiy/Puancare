#!/usr/bin/env python3
"""
Step A — диагностика: провал 3b — это теория или измерение?

Гипотеза измерения: большинство случайных cousin-пар НЕ соблазнительны ни в
какую сторону (обе правдоподобности ~0) => асимметрия = шум => sign-match 50%.

Берём кэш фронтира (step3b_cache.json, без новых вызовов), считаем:
  salience = max(a_is_b, b_is_a)   — есть ли соблазн ВООБЩЕ
  asym     = a_is_b - b_is_a
Пере-тестим Spearman(наш, asym) и partial|freq на подмножествах salience>=T
(пред-регистрируемый порог). Если на соблазнительных парах сигнал оживает —
провал был от шума, а не от теории.
"""

import sys
import json
import math
import os

from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic
from step3_asymmetry import build_pairs, lemma_freq, spearman, partial_spearman

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step3b_cache.json")


def main():
    ensure_wordnet()
    root, nodes, edges, parents = collect_subtree("animal.n.01")
    h_ic, _ = build_h_ic(root, nodes, parents)
    isa, cousins, words = build_pairs(nodes, parents)

    if not os.path.exists(CACHE):
        print("НЕТ кэша step3b_cache.json — сначала прогони step3b на VPS."); return
    cache = json.load(open(CACHE))

    rows = []
    for a, b in cousins:
        wa, wb = words[a], words[b]
        r = cache.get(f"{wa}|{wb}")
        if not r:
            continue
        ab, ba = float(r.get("a_is_b", 0)), float(r.get("b_is_a", 0))
        rows.append(dict(a=a, b=b, wa=wa, wb=wb, ab=ab, ba=ba,
                         sal=max(ab, ba), asym=ab - ba,
                         our=h_ic[a] - h_ic[b],
                         freq=math.log(lemma_freq(a)) - math.log(lemma_freq(b))))

    print("=" * 70)
    print(f"  ДИАГНОСТИКА SALIENCE  —  cousin-пар с ответом: {len(rows)}")
    print("=" * 70)
    bins = [(0, 5), (5, 20), (20, 40), (40, 70), (70, 101)]
    print("  Распределение salience = max(правдоподобность обеих сторон):")
    for lo, hi in bins:
        c = sum(1 for r in rows if lo <= r["sal"] < hi)
        bar = "#" * (c * 50 // max(1, len(rows)))
        print(f"    [{lo:3d},{hi:3d}) : {c:3d}  {bar}")
    noise = sum(1 for r in rows if r["sal"] < 20)
    print(f"  Пар без реального соблазна (salience<20): {noise}/{len(rows)} "
          f"= {noise/len(rows):.0%}  <- кандидаты в шум")
    print("-" * 70)

    print("  Пере-тест на подмножествах salience>=T (пред-регистрируемый порог):")
    print(f"    {'T':>4s} {'n':>4s} {'Spear(наш,asym)':>16s} {'partial|freq':>14s} {'sign-match':>11s}")
    for T in (0, 10, 20, 30, 40, 50):
        sub = [r for r in rows if r["sal"] >= T]
        if len(sub) < 8:
            print(f"    {T:>4d} {len(sub):>4d}   (мало данных)"); continue
        O = [r["our"] for r in sub]; A = [r["asym"] for r in sub]; F = [r["freq"] for r in sub]
        sp = spearman(O, A)
        pp = partial_spearman(O, A, F)
        nz = sum(1 for r in sub if r["asym"] != 0)
        sm = sum(1 for r in sub if (r["our"] > 0) == (r["asym"] > 0) and r["asym"] != 0)
        smr = sm / nz if nz else float("nan")
        print(f"    {T:>4d} {len(sub):>4d} {sp:>+16.3f} {pp:>+14.3f} {smr:>10.1%}")
    print("=" * 70)
    print("  Чтение: если на высоком T (соблазнительные пары) partial|freq и sign-match")
    print("  заметно растут — провал 3b был ШУМОМ измерения, не смертью теории.")


if __name__ == "__main__":
    main()
