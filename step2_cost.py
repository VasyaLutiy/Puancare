#!/usr/bin/env python3
"""
Step 2 — Формула стоимости (ResearchPuancare.md §8.2)

ОДНА формула, ноль per-pair тюнинга. Стоимость направленного перехода:

    cost(A -> B) = h_ic(B) - h_ic( LCA(A,B) )

  LCA = наиболее информативный общий предок (MICA, max h_ic среди общих предков).
  - подъём A -> LCA  = обобщение, бесплатно (≈0 бит);
  - спуск  LCA -> B  = конкретизация, стоит ΔIC = log2(|desc(LCA)|/|desc(B)|).

Асимметрия выпадает аналитически:
    cost(A->B) - cost(B->A) = h_ic(B) - h_ic(A)
  => дороже та сторона, что КОНКРЕТНЕЕ. Это founding-феномен «это» в формуле.

Связь с §8.1: на ребре is-a (child->parent) обобщение стоит 0, конкретизация = ΔIC.
"""

import sys
import math
from collections import deque

from nltk.corpus import wordnet as wn
from step0_tda_gate import ensure_wordnet, collect_subtree


def build_h_ic(root, nodes, parents):
    """h_ic и множества потомков (как в Шаге 1)."""
    children = {n: [] for n in nodes}
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    desc = {}
    sys.setrecursionlimit(1 << 20)

    def dfs(u):
        if u in desc:
            return desc[u]
        s = {u}
        for v in children[u]:
            s |= dfs(v)
        desc[u] = s
        return s

    for n in nodes:
        dfs(n)
    V = len(nodes)
    h_ic = {n: -math.log2(len(desc[n]) / V) for n in nodes}
    ndesc = {n: len(desc[n]) for n in nodes}
    return h_ic, ndesc


def ancestors(node, parents):
    """Множество предков ВКЛЮЧАЯ себя (транзитивно вверх по parents)."""
    seen = {node}
    q = deque([node])
    while q:
        u = q.popleft()
        for p in parents.get(u, ()):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return seen


def lca(a, b, parents, h_ic):
    """MICA — общий предок с максимальным h_ic (самый информативный)."""
    common = ancestors(a, parents) & ancestors(b, parents)
    if not common:
        return None
    return max(common, key=lambda c: h_ic[c])


def cost(a, b, parents, h_ic):
    l = lca(a, b, parents, h_ic)
    if l is None:
        return None, None
    return h_ic[b] - h_ic[l], l


def is_a(a, b, parents):
    """Истинно, если B — предок A (A is-a B), B != A."""
    return b != a and b in ancestors(a, parents)


def resolve(word, nodes):
    """Первый noun-синсет слова, лежащий в subtree (отсекает не-животные смыслы)."""
    for s in wn.synsets(word, pos="n"):
        if s in nodes:
            return s
    return None


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "animal.n.01"
    ensure_wordnet()
    root, nodes, edges_und, parents = collect_subtree(root_name)
    h_ic, ndesc = build_h_ic(root, nodes, parents)

    # --- 1. структурная валидация на ВСЕХ рёбрах is-a ---
    spec_costs = []
    gen_nonzero = 0
    for child, ps in parents.items():
        for p in ps:
            cg, _ = cost(child, p, parents, h_ic)   # обобщение child->parent
            cs, _ = cost(p, child, parents, h_ic)    # конкретизация parent->child
            if cg is not None and abs(cg) > 1e-9:
                gen_nonzero += 1
            if cs is not None:
                spec_costs.append(cs)
    n = len(spec_costs)
    print("=" * 70)
    print(f"  ФОРМУЛА СТОИМОСТИ  —  subtree {root_name}  ({n} рёбер is-a)")
    print("=" * 70)
    print(f"  обобщение (child->parent): рёбер с ненулевой ценой = {gen_nonzero}  "
          f"(ожидание 0 — обобщение всегда бесплатно)")
    spec_costs.sort()
    print(f"  конкретизация (parent->child) ΔIC, бит:")
    print(f"     min={spec_costs[0]:.2f}  med={spec_costs[n//2]:.2f}  "
          f"max={spec_costs[-1]:.2f}  mean={sum(spec_costs)/n:.2f}")
    print("=" * 70)

    # --- 2. боевые пары (founding-феномен) ---
    pairs = [
        ("pike", "fish"),     # щука это рыба — is-a, обобщение дёшево
        ("whale", "fish"),    # кит это рыба? — НЕ is-a, спуск нужен
        ("whale", "mammal"),  # кит это млекопитающее — is-a
        ("shark", "fish"),
        ("salmon", "fish"),
        ("dog", "animal"),
    ]
    print("  Направленная стоимость на парах (founding-феномен «это»):")
    print(f"    {'A':9s} {'B':9s} {'A is-a B':>9s} {'cost A→B':>9s} "
          f"{'cost B→A':>9s} {'асим':>7s}  LCA")
    for wa, wb in pairs:
        a, b = resolve(wa, nodes), resolve(wb, nodes)
        if a is None or b is None:
            print(f"    {wa:9s} {wb:9s}  (не найдено в subtree)")
            continue
        cab, lab = cost(a, b, parents, h_ic)
        cba, _ = cost(b, a, parents, h_ic)
        asym = cab - cba  # = h_ic(b) - h_ic(a)
        flag = "ДА" if is_a(a, b, parents) else "нет"
        print(f"    {wa:9s} {wb:9s} {flag:>9s} {cab:9.2f} {cba:9.2f} "
              f"{asym:+7.2f}  {lab.name()}")
    print("=" * 70)
    print("  Чтение: is-a ДА  =>  cost A→B ≈ 0 (обобщение дёшево, спонтанный 'да').")
    print("          асим > 0 =>  B конкретнее A (спуск к B дороже подъёма).")
    print("          'кит/рыба': is-a нет, но общий предок есть => 'нет' + цена спуска.")


if __name__ == "__main__":
    main()
