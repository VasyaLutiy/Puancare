#!/usr/bin/env python3
"""
Step A2 — ЧИСТЫЙ тест founding-феномена «ЭТО» (внутри пары, на истинных is-a).

Прошлая ошибка: тестировали cousin-пары кросс-парной корреляцией, которая слепа
к равномерному по знаку эффекту. Теперь правильно:

Для истинной is-a пары (S конкретное, G его предок-общее):
    вперёд  "S is a G"  — истинно, обобщение  (наша cost = 0)
    назад   "G is a S"  — ложно, конкретизация (наша cost = h_ic(S)-h_ic(G) = log|поддерева|)

Проверяем:
  (1) НАПРАВЛЕНИЕ: доля пар, где судья(вперёд) > судья(назад). Предсказание ~100%.
  (2) ВЕЛИЧИНА:    растёт ли asym судьи с нашей cost? и бьёт ли cost простую длину пути?
Судьи: GPT-2 logprob (быстрый) + gpt-5.4-mini (подтверждение). Контроль по частоте.
"""

import sys
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic
from step3_asymmetry import (build_pairs, lemma_freq, spearman, partial_spearman,
                             GPT2Scorer, sentence)


def depth_map(root, nodes, parents):
    children = {n: [] for n in nodes}
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    for n in nodes:
        depth.setdefault(n, 0)
    return depth


def main():
    ensure_wordnet()
    root, nodes, edges, parents = collect_subtree("animal.n.01")
    h_ic, _ = build_h_ic(root, nodes, parents)
    depth = depth_map(root, nodes, parents)
    isa, cousins, words = build_pairs(nodes, parents)
    print(f"[data] истинных is-a пар: {len(isa)}")

    # наша cost конкретизации (назад) = h_ic(S)-h_ic(G); путь = depth(S)-depth(G)
    items = []
    for s, g in isa:
        ws, wg = words[s], words[g]
        items.append(dict(s=s, g=g, ws=ws, wg=wg,
                          our=h_ic[s] - h_ic[g],
                          path=depth[s] - depth[g],
                          freq=math.log(lemma_freq(s)) - math.log(lemma_freq(g))))

    # ---------- судья 1: GPT-2 ----------
    print("[gpt2] loading ...", file=sys.stderr)
    sc = GPT2Scorer("gpt2")
    for it in items:
        fwd = sc.logprob(sentence(it["ws"], it["wg"]))   # "S is a G" (истина)
        bwd = sc.logprob(sentence(it["wg"], it["ws"]))   # "G is a S" (ложь)
        it["g2_fwd"], it["g2_bwd"] = fwd, bwd
        it["g2_asym"] = fwd - bwd

    g2_dir = sum(1 for it in items if it["g2_asym"] > 0) / len(items)
    G2 = [it["g2_asym"] for it in items]
    O = [it["our"] for it in items]
    P = [it["path"] for it in items]
    F = [it["freq"] for it in items]

    print("=" * 70)
    print("  СУДЬЯ 1 — GPT-2 logprob")
    print("=" * 70)
    print(f"  (1) НАПРАВЛЕНИЕ: вперёд>назад у {g2_dir:.1%} пар  (предсказание ~100%)")
    print(f"      средняя asym (биты лог-вероятности): {sum(G2)/len(G2):+.2f}")
    print(f"  (2) ВЕЛИЧИНА (масштабируется ли с трудностью конкретизации):")
    print(f"      Spearman(наша cost=log|поддерева|, asym) : {spearman(O, G2):+.3f}")
    print(f"      Spearman(длина пути, asym)               : {spearman(P, G2):+.3f}  <- наивный конкурент")
    print(f"      partial(наша cost, asym | freq)          : {partial_spearman(O, G2, F):+.3f}")
    print("=" * 70)

    # ---------- судья 2: gpt-5.4-mini ----------
    try:
        from utils_azure import AzureJSON
        az = AzureJSON()
        SYS = ("Ты оцениваешь ИСТИННОСТЬ категориальных утверждений (json). "
               "Шкала 0..100: 100 = очевидно истинно, 0 = очевидно ложно.")
        SCH = {"forward": "0..100", "backward": "0..100"}

        def ask(it):
            ws, wg = it["ws"], it["wg"]
            user = (f"Оцени истинность двух утверждений: "
                    f"forward = 'a {ws} is a kind of {wg}'; "
                    f"backward = 'a {wg} is a kind of {ws}'.")
            try:
                r = az.ask(SYS, user, SCH)
                return it, float(r.get("forward", 0)), float(r.get("backward", 0))
            except Exception:
                return it, None, None

        print("[azure] судья 2 (gpt-5.4-mini) live...", file=sys.stderr)
        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for it, fw, bw in [f.result() for f in
                               [ex.submit(ask, it) for it in items]]:
                if fw is not None:
                    it["fr_asym"] = fw - bw
                done += 1
        fr = [it for it in items if "fr_asym" in it]
        fr_dir = sum(1 for it in fr if it["fr_asym"] > 0) / len(fr)
        FR = [it["fr_asym"] for it in fr]
        Ofr = [it["our"] for it in fr]
        Pfr = [it["path"] for it in fr]
        Ffr = [it["freq"] for it in fr]
        print("=" * 70)
        print("  СУДЬЯ 2 — gpt-5.4-mini (истинность)")
        print("=" * 70)
        print(f"  (1) НАПРАВЛЕНИЕ: вперёд>назад у {fr_dir:.1%} из {len(fr)} пар")
        print(f"  (2) Spearman(наша cost, asym)   : {spearman(Ofr, FR):+.3f}")
        print(f"      Spearman(длина пути, asym)   : {spearman(Pfr, FR):+.3f}")
        print(f"      partial(наша cost | freq)    : {partial_spearman(Ofr, FR, Ffr):+.3f}")
        print("=" * 70)
    except Exception as e:
        print(f"[azure] пропущен: {e}")

    # примеры
    print("  Примеры (S→G истина / G→S ложь):")
    print(f"    {'S':10s} {'G':10s} {'our cost':>8s} {'g2_asym':>8s}")
    for it in sorted(items, key=lambda x: -x["our"])[:8]:
        print(f"    {it['ws']:10s} {it['wg']:10s} {it['our']:8.2f} {it['g2_asym']:+8.2f}")


if __name__ == "__main__":
    main()
