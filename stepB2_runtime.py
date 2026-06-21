#!/usr/bin/env python3
"""
Step B2 — РАНТАЙМ на атомах (БЕЗ LLM). Алгебра множеств над atoms.json.

  sim(X,Y)  = Jaccard(A(X), A(Y))            — ось СХОДСТВА (симметрична)
  incl(X,Y) = |A(X) ∩ A(Y)| / |A(Y)|         — «X это Y»: доля атомов Y, покрытых X

Проверяем:
  1. ВЛОЖЕНИЕ (несущая ставка): на истинных is-a — atoms(gen) ⊆ atoms(spec)?
     Считаем нарушения (атомы общего, которых нет у конкретного) = прототип-проблема.
  2. НАПРАВЛЕНИЕ founding: incl(spec,gen) > incl(gen,spec) на истинных is-a?
  3. ДИСКРИМИНАЦИЯ: отличает ли incl истинные is-a от cousins? (ROC-AUC)
  4. ОБЕ ОСИ из атомов: кит/рыба — sim высок, incl<1 (похожи, но не is-a)?
  5. Пингвин/птица — где ломается (нет can_fly)?
"""

import json
import os
import itertools

A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json")))
VOCAB = A["atom_vocab"]
C = {k: set(v) for k, v in A["concepts"].items()}


def sim(x, y):
    a, b = C[x], C[y]
    return len(a & b) / len(a | b) if (a | b) else 0.0


def incl(x, y):
    """доля атомов Y, покрытых X = насколько 'X это Y'."""
    if not C[y]:
        return 0.0
    return len(C[x] & C[y]) / len(C[y])


# истинные is-a (spec, gen)
ISA = [
    ("carp", "fish"), ("goldfish", "fish"), ("pike", "fish"), ("salmon", "fish"),
    ("shark", "fish"), ("tuna", "fish"), ("eel", "fish"),
    ("whale", "mammal"), ("dolphin", "mammal"), ("seal", "mammal"), ("bat", "mammal"),
    ("dog", "mammal"), ("cat", "mammal"), ("horse", "mammal"), ("cow", "mammal"),
    ("mouse", "mammal"), ("elephant", "mammal"),
    ("eagle", "bird"), ("penguin", "bird"), ("ostrich", "bird"), ("sparrow", "bird"),
    ("owl", "bird"), ("duck", "bird"),
    ("snake", "reptile"), ("lizard", "reptile"), ("turtle", "reptile"), ("crocodile", "reptile"),
    ("bee", "insect"), ("ant", "insect"), ("butterfly", "insect"),
    ("fish", "vertebrate"), ("mammal", "vertebrate"), ("bird", "vertebrate"),
    ("vertebrate", "animal"), ("mammal", "animal"), ("fish", "animal"), ("bird", "animal"),
]
# cousins (НЕ is-a)
COUSIN = [
    ("whale", "fish"), ("dolphin", "fish"), ("seal", "fish"), ("penguin", "fish"),
    ("shark", "whale"), ("bat", "bird"), ("crocodile", "fish"), ("eel", "snake"),
    ("dolphin", "shark"), ("seal", "whale"), ("frog", "fish"), ("turtle", "fish"),
    ("octopus", "fish"), ("crab", "spider"), ("bat", "mouse"),
]


def auc(pos, neg):
    """ROC-AUC: P(score(pos) > score(neg))."""
    win = ties = 0
    for p in pos:
        for n in neg:
            if p > n: win += 1
            elif p == n: ties += 1
    tot = len(pos) * len(neg)
    return (win + 0.5 * ties) / tot if tot else float("nan")


def main():
    print("=" * 72)
    print("  B2 РАНТАЙМ на атомах (без LLM)")
    print("=" * 72)

    # 1. вложение
    viol = []
    strict_ok = 0
    for s, g in ISA:
        missing = C[g] - C[s]   # атомы общего, которых нет у конкретного
        if not missing:
            strict_ok += 1
        else:
            viol.append((s, g, sorted(missing)))
    print(f"  1. СТРОГОЕ ВЛОЖЕНИЕ atoms(gen)⊆atoms(spec): {strict_ok}/{len(ISA)} пар чисты")
    print(f"     => прототип-проблема в {len(viol)} парах. Примеры (чего общему не хватает у конкретного):")
    for s, g, m in viol[:6]:
        print(f"       {s}⊉{g}: {g} имеет, {s} нет → {m}")
    print("-" * 72)

    # 2. направление founding
    dir_ok = sum(1 for s, g in ISA if incl(s, g) > incl(g, s))
    print(f"  2. НАПРАВЛЕНИЕ incl(spec,gen)>incl(gen,spec): {dir_ok}/{len(ISA)} = {dir_ok/len(ISA):.0%}")
    mean_sg = sum(incl(s, g) for s, g in ISA) / len(ISA)
    mean_gs = sum(incl(g, s) for s, g in ISA) / len(ISA)
    print(f"     ср. incl(spec→gen)={mean_sg:.2f}  vs  incl(gen→spec)={mean_gs:.2f}")
    print("-" * 72)

    # 3. дискриминация is-a vs cousin (по incl(spec,gen) макс из направлений)
    isa_scores = [max(incl(a, b), incl(b, a)) for a, b in ISA]
    cou_scores = [max(incl(a, b), incl(b, a)) for a, b in COUSIN]
    print(f"  3. ДИСКРИМИНАЦИЯ is-a vs cousin по incl: ROC-AUC = {auc(isa_scores, cou_scores):.3f}")
    print(f"     (incl одинаково высок и там и там? тогда incl НЕ различает — нужна 2-я ось)")
    print(f"     ср incl is-a={sum(isa_scores)/len(isa_scores):.2f}  cousin={sum(cou_scores)/len(cou_scores):.2f}")
    print("-" * 72)

    # 4. обе оси: кит/рыба и компания
    print("  4. ОБЕ ОСИ из атомов (sim=сходство, incl=вложение):")
    print(f"     {'X':9s} {'Y':9s} {'sim':>5s} {'incl X→Y':>9s} {'incl Y→X':>9s}  тип")
    show = [("carp", "fish", "is-a"), ("whale", "fish", "cousin"),
            ("dolphin", "fish", "cousin"), ("shark", "whale", "cousin"),
            ("whale", "mammal", "is-a"), ("bat", "bird", "cousin"),
            ("dog", "cat", "siblings"), ("octopus", "fish", "cousin")]
    for x, y, t in show:
        print(f"     {x:9s} {y:9s} {sim(x,y):5.2f} {incl(x,y):9.2f} {incl(y,x):9.2f}  {t}")
    print("-" * 72)

    # 5. пингвин
    print("  5. ПИНГВИН/ПТИЦА (прототип-исключение):")
    miss = C["bird"] - C["penguin"]
    print(f"     incl(penguin,bird)={incl('penguin','bird'):.2f}; птица имеет, пингвин нет → {sorted(miss)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
