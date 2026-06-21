#!/usr/bin/env python3
"""
Step 3b — Фронтир-подтверждение (ResearchPuancare.md §8.3, escalation)

Независимый сигнал #2: gpt-5.4-mini. Спрашиваем НЕ истинность (насыщается в ~1.0),
а градуированную ПРАВДОПОДОБНОСТЬ категориальной ошибки в обе стороны:
    frontier_asym(A,B) = plaus("A is a kind of B") - plaus("B is a kind of A")

Наш предиктор (заморожен в Шаге 2):  our(A,B) = h_ic(A) - h_ic(B) = cost-асим.
  >0  => A конкретнее => ошибка "A это B" соблазнительнее (кит→рыба, не рыба→кит).

Пары — ТЕ ЖЕ cousin-пары из Шага 3 (тот же SEED=0 => пред-регистрация).
Критерий тот же: partial Spearman(our, frontier | freq) должен выжить.
"""

import sys
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy import stats

from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic
from step3_asymmetry import build_pairs, lemma_freq, spearman, partial_spearman
from utils_azure import AzureJSON

SYSTEM = (
    "Ты модель когнитивной правдоподобности КАТЕГОРИАЛЬНЫХ ОШИБОК (json). "
    "Для пары существительных оцени не логическую истинность, а насколько "
    "ПРАВДОПОДОБНА, соблазнительна как интуитивная ошибка каждая фраза-вложение. "
    "Шкала 0..100: 0 = абсурд, никто так не подумает; "
    "100 = очень соблазнительно, люди/дети часто так ошибаются. "
    "Учитывай поверхностное сходство: среда обитания, форма тела, поведение, размер."
)

SCHEMA = {"a_is_b": "0..100", "b_is_a": "0..100", "shared": "общее свойство кратко"}


def query_one(az, wa, wb):
    user = (f"A = {wa}, B = {wb}. "
            f"Оцени правдоподобность интуитивной ошибки: "
            f"(1) 'a {wa} is a kind of {wb}'; (2) 'a {wb} is a kind of {wa}'.")
    return az.ask(system=SYSTEM, user=user, schema=SCHEMA)


def collect_frontier(cousins, words, az=None, workers=8, progress=True):
    """LIVE-опрос фронтира (без кэша). Возвращает dict 'wa|wb' -> {a_is_b,b_is_a,shared}."""
    if az is None:
        az = AzureJSON()
    results = {}

    def work(pair):
        a, b = pair
        wa, wb = words[a], words[b]
        try:
            r = query_one(az, wa, wb)
            return wa, wb, dict(a_is_b=float(r.get("a_is_b", 0)),
                                b_is_a=float(r.get("b_is_a", 0)),
                                shared=r.get("shared", ""))
        except Exception as e:
            return wa, wb, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p) for p in cousins]
        for fu in as_completed(futs):
            wa, wb, val = fu.result()
            if val is not None:
                results[f"{wa}|{wb}"] = val
            done += 1
            if progress and done % 20 == 0:
                print(f"  ... {done}/{len(cousins)}", file=sys.stderr)
    return results


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "animal.n.01"
    ensure_wordnet()
    root, nodes, edges_und, parents = collect_subtree(root_name)
    h_ic, ndesc = build_h_ic(root, nodes, parents)

    isa, cousins, words = build_pairs(nodes, parents)
    print(f"[data] cousin пар (как в Шаге 3): {len(cousins)}")

    az = AzureJSON()
    print(f"[azure] model={az.model}; LIVE без кэша. Запрашиваю...", file=sys.stderr)
    fr = collect_frontier(cousins, words, az)

    rows = []
    for a, b in cousins:
        v = fr.get(f"{words[a]}|{words[b]}")
        if v is None:
            continue
        rows.append((a, b, words[a], words[b], v["a_is_b"], v["b_is_a"], v["shared"]))
    good = rows
    print(f"[azure] успешно: {len(good)}/{len(cousins)}")

    O = [h_ic[a] - h_ic[b] for a, b, wa, wb, ab, ba, sh in good]
    Fr = [math.log(lemma_freq(a)) - math.log(lemma_freq(b)) for a, b, *_ in good]
    Front = [ab - ba for *_, ab, ba, sh in good]

    def signmatch(pred):
        n = sum(1 for f in Front if f != 0)
        m = sum(1 for p, f in zip(pred, Front) if (p > 0) == (f > 0) and f != 0)
        return m / n if n else float("nan")

    print("=" * 72)
    print(f"  ФРОНТИР-ТЕСТ (gpt-5.4-mini)  —  cousin-пары, N={len(good)}")
    print("=" * 72)
    print("  Spearman с фронтир-асимметрией правдоподобности ошибки:")
    print(f"    НАШ  h_ic-diff : {spearman(O, Front):+.3f}")
    print(f"    freq_diff      : {spearman(Fr, Front):+.3f}   <- конкурент")
    print(f"    sign-match НАШ : {signmatch(O):.1%}   (chance=50%)")
    print("-" * 72)
    ps = partial_spearman(O, Front, Fr)
    print("  *** КРИТЕРИЙ УБИЙСТВА ***")
    print(f"    partial Spearman(НАШ, frontier | freq) = {ps:+.3f}")
    print("    -> ВЫЖИЛО ✅" if abs(ps) >= 0.1 else "    -> СХЛОПНУЛОСЬ ⚰️")
    print("=" * 72)

    # классические спорные пары — явный sanity (не входят в стат-тест)
    classics = [("whale", "fish"), ("dolphin", "fish"), ("penguin", "bird"),
                ("bat", "bird"), ("seal", "fish"), ("shark", "whale")]
    print("  Классические пары (sanity, прямой запрос):")
    print(f"    {'A':9s} {'B':9s} {'A→B':>5s} {'B→A':>5s} {'наш':>6s}  shared")
    from step2_cost import resolve
    for wa, wb in classics:
        try:
            r = query_one(az, wa, wb)
        except Exception as e:
            print(f"    {wa:9s} {wb:9s}  ERR {e}"); continue
        a, b = resolve(wa, nodes), resolve(wb, nodes)
        our = (h_ic[a] - h_ic[b]) if (a and b) else float("nan")
        print(f"    {wa:9s} {wb:9s} {float(r.get('a_is_b',0)):5.0f} "
              f"{float(r.get('b_is_a',0)):5.0f} {our:+6.2f}  {r.get('shared','')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
