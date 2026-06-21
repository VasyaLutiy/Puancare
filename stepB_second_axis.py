#!/usr/bin/env python3
"""
Step B — вторая ось: поперечное СХОДСТВО (то, что требуют Шаг 0 и 3b).

Диагноз 3b: скаляр h (абстракция) не знает про поверхностное сходство
(вода, крылья) — а соблазн ошибки сидит именно там. Добавляем вторую ось
БЕЗ непрозрачных эмбеддингов: overlap глоссов (определений) WordNet.

  gloss_sim(A,B) = Jaccard(content-words определений) — симметричная, интерпретируемая.

Модель соблазна ошибки A->B (направленная):
  заманчиво, когда (1) A и B ПОХОЖИ поверхностно (gloss_sim высок)
  И (2) B абстрактнее A (обобщение вверх дёшево):  знак = h_ic(A)-h_ic(B).
  Комбинированный предиктор:  combined = (h_ic(A)-h_ic(B)) * gloss_sim(A,B)

Тест на кэше фронтира (без новых вызовов): бьёт ли combined и h_ic-одиночку,
и freq? partial Spearman(combined, frontier_asym | freq) должен ОЖИТЬ.
"""

import sys
import re
import json
import math
import os

from nltk.corpus import wordnet as wn
from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic, resolve
from step3_asymmetry import build_pairs, lemma_freq, spearman, partial_spearman

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step3b_cache.json")

STOP = set("a an the of to in on and or with is are was were be been being for "
           "as by at from that this these those it its their his her any some "
           "such other than then having have has which who whom small large "
           "used kind type form group one two having".split())


def content_words(synset):
    toks = re.findall(r"[a-z]+", synset.definition().lower())
    cw = {t for t in toks if t not in STOP and len(t) > 2}
    cw |= {l.lower() for l in synset.lemma_names() if l.isalpha()}
    return cw


def gloss_sim(a, b):
    ca, cb = content_words(a), content_words(b)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def main():
    ensure_wordnet()
    root, nodes, edges, parents = collect_subtree("animal.n.01")
    h_ic, _ = build_h_ic(root, nodes, parents)
    isa, cousins, words = build_pairs(nodes, parents)

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if not cache:
        print("НЕТ кэша step3b_cache.json — прогони step3b на VPS."); return

    rows = []
    for a, b in cousins:
        wa, wb = words[a], words[b]
        r = cache.get(f"{wa}|{wb}")
        if not r:
            continue
        ab, ba = float(r.get("a_is_b", 0)), float(r.get("b_is_a", 0))
        gs = gloss_sim(a, b)
        hd = h_ic[a] - h_ic[b]
        rows.append(dict(a=a, b=b, wa=wa, wb=wb, asym=ab - ba, sal=max(ab, ba),
                         our=hd, gloss=gs, combined=hd * gs,
                         freq=math.log(lemma_freq(a)) - math.log(lemma_freq(b))))

    A = [r["asym"] for r in rows]
    O = [r["our"] for r in rows]
    G = [r["gloss"] for r in rows]
    Cc = [r["combined"] for r in rows]
    F = [r["freq"] for r in rows]

    print("=" * 72)
    print(f"  ВТОРАЯ ОСЬ (gloss-сходство)  —  cousin-пары, N={len(rows)}")
    print("=" * 72)
    print("  Spearman с фронтир-асимметрией соблазна:")
    print(f"    h_ic-diff (1 ось, абстракция)      : {spearman(O, A):+.3f}")
    print(f"    gloss_sim (2 ось, сходство, симм.) : {spearman(G, A):+.3f}")
    print(f"    COMBINED = h_ic-diff * gloss_sim   : {spearman(Cc, A):+.3f}   <- две оси")
    print("-" * 72)
    print("  *** КРИТЕРИЙ УБИЙСТВА (partial | freq) ***")
    print(f"    h_ic-diff одиночка : {partial_spearman(O, A, F):+.3f}")
    print(f"    COMBINED две оси   : {partial_spearman(Cc, A, F):+.3f}")
    pc = partial_spearman(Cc, A, F)
    print("    -> ВТОРАЯ ОСЬ ОЖИВИЛА ✅" if abs(pc) >= 0.15 else "    -> всё равно слабо ⚰️")
    print("-" * 72)
    # на соблазнительном подмножестве
    sub = [r for r in rows if r["sal"] >= 30]
    if len(sub) >= 8:
        As = [r["asym"] for r in sub]; Cs = [r["combined"] for r in sub]; Fs = [r["freq"] for r in sub]
        print(f"  На соблазнительных парах (salience>=30, n={len(sub)}):")
        print(f"    COMBINED partial|freq = {partial_spearman(Cs, As, Fs):+.3f}")
    print("=" * 72)

    # классика
    classics = [("whale", "fish"), ("dolphin", "fish"), ("penguin", "bird"),
                ("bat", "bird"), ("seal", "fish"), ("shark", "whale")]
    print("  Классика: gloss_sim и combined (sanity):")
    print(f"    {'A':9s} {'B':9s} {'asym':>6s} {'h_ic-d':>7s} {'gloss':>6s} {'comb':>7s}")
    for wa, wb in classics:
        a, b = resolve(wa, nodes), resolve(wb, nodes)
        if not (a and b):
            continue
        r = cache.get(f"{wa}|{wb}")
        asym = (float(r["a_is_b"]) - float(r["b_is_a"])) if r else float("nan")
        hd = h_ic[a] - h_ic[b]; gs = gloss_sim(a, b)
        print(f"    {wa:9s} {wb:9s} {asym:6.0f} {hd:+7.2f} {gs:6.2f} {hd*gs:+7.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
