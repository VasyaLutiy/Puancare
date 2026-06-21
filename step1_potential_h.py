#!/usr/bin/env python3
"""
Step 1 — Потенциал h(x) из noun-hypernymy (ResearchPuancare.md §8.1)

h — скалярное поле над синсетами. Общее = низко/стабильно, конкретное = высоко.
is-a = ∇h (обобщение = спуск, конкретизация = подъём).

Строим ДВА кандидата и сравниваем по согласованности потенциала на рёбрах is-a:

  h_depth(x) = кратчайшая глубина от корня subtree   (Poincaré-радиус)
      -> ожидаемо ЛОМАЕТСЯ на loop-узлах (β₁ из Шага 0):
         у узла с 2 родителями min-путь даёт h(child) < h(parent) по длинной ветке.

  h_ic(x)    = -log2( |descendants(x)| / V )           (информационное содержание)
      -> определён через МНОЖЕСТВО потомков => h(child) >= h(parent) на КАЖДОМ ребре
         по включению множеств, даже на петлях. Это и есть log(N) из §8.2.

Критерий Шага 1: какой потенциал согласован? Ожидание — h_ic чист,
h_depth нарушается ровно у loop-узлов. Тогда Шаг 0 (β₁) и Шаг 1 сшиваются.
"""

import sys
import math
from collections import deque

from nltk.corpus import wordnet as wn
from step0_tda_gate import ensure_wordnet, collect_subtree


def depth_from_root(root, nodes, parents):
    """h_depth: BFS вниз по hyponym из корня => кратчайшая глубина (root=0)."""
    children = {n: [] for n in nodes}
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    # узлы, не достигнутые из корня по hyponym (изолированные ветви) — глубина по родителю
    for n in nodes:
        if n not in depth:
            depth[n] = 0
    return depth


def descendant_counts(root, nodes, parents):
    """|descendants(x) including self| через мемоизированный DFS по DAG hyponym."""
    children = {n: [] for n in nodes}
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)

    desc = {}

    def dfs(u):
        if u in desc:
            return desc[u]
        s = {u}
        for v in children[u]:
            s |= dfs(v)
        desc[u] = s
        return s

    sys.setrecursionlimit(1 << 20)
    for n in nodes:
        dfs(n)
    return {n: len(desc[n]) for n in nodes}


def consistency(edges_dir, h, name):
    """Доля рёбер (child->parent), где h(child) > h(parent). Нарушение = h(child) <= h(parent)."""
    ok = strict = viol = eq = 0
    violations = []
    for child, parent in edges_dir:
        hc, hp = h[child], h[parent]
        if hc > hp:
            strict += 1; ok += 1
        elif hc == hp:
            eq += 1; ok += 1
        else:
            viol += 1
            violations.append((child, parent, hc, hp))
    total = len(edges_dir)
    print(f"  [{name}] рёбер={total}  strict↑={strict}  равных={eq}  "
          f"НАРУШЕНИЙ={viol}  согласованность={ok/total:.4%}")
    return violations


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "animal.n.01"
    ensure_wordnet()
    root, nodes, edges_und, parents = collect_subtree(root_name)
    V = len(nodes)

    # направленные рёбра child -> parent (is-a)
    edges_dir = [(child, p) for child, ps in parents.items() for p in ps]

    depth = depth_from_root(root, nodes, parents)
    ndesc = descendant_counts(root, nodes, parents)
    h_depth = {n: float(depth[n]) for n in nodes}
    h_ic = {n: -math.log2(ndesc[n] / V) for n in nodes}

    print("=" * 70)
    print(f"  ПОТЕНЦИАЛ h  —  subtree {root_name}  (V={V}, is-a рёбер={len(edges_dir)})")
    print("=" * 70)
    dvals = [depth[n] for n in nodes]
    print(f"  h_depth: min={min(dvals)} max={max(dvals)} "
          f"mean={sum(dvals)/V:.2f}")
    icvals = [h_ic[n] for n in nodes]
    print(f"  h_ic   : min={min(icvals):.2f} max={max(icvals):.2f} "
          f"mean={sum(icvals)/V:.2f}  (биты)")
    print("-" * 70)
    print("  Согласованность потенциала (is-a должно давать h(child) > h(parent)):")
    v_depth = consistency(edges_dir, h_depth, "h_depth")
    v_ic = consistency(edges_dir, h_ic, "h_ic   ")
    print("-" * 70)

    # сшивка с Шагом 0: нарушения h_depth должны сидеть на loop-узлах
    loop_nodes = {s for s, ps in parents.items() if len(ps) > 1}
    if v_depth:
        on_loop = sum(1 for c, p, hc, hp in v_depth if c in loop_nodes)
        print(f"  h_depth нарушений: {len(v_depth)}, из них на loop-узлах (β₁): {on_loop}")
        print("  Примеры нарушений h_depth (длинная ветка тянет вверх):")
        for c, p, hc, hp in sorted(v_depth, key=lambda x: x[2]-x[3])[:6]:
            print(f"    {c.name():26s}(h={hc:.0f}) --is-a--> {p.name():22s}(h={hp:.0f})  Δ={hc-hp:+.0f}")
    else:
        print("  h_depth: нарушений нет")
    print("=" * 70)

    # sanity на якорях
    anchors = ["animal.n.01", "vertebrate.n.01", "fish.n.01", "whale.n.02",
               "mammal.n.01", "salmon.n.01", "dog.n.01"]
    print("  Якоря (общее=низко, конкретное=высоко):")
    print(f"    {'synset':22s} {'depth':>6s} {'h_ic(bit)':>10s} {'|desc|':>8s}")
    for a in anchors:
        try:
            s = wn.synset(a)
        except Exception:
            continue
        if s in nodes:
            print(f"    {a:22s} {depth[s]:6d} {h_ic[s]:10.2f} {ndesc[s]:8d}")
        else:
            print(f"    {a:22s}  (вне subtree {root_name})")
    print("=" * 70)


if __name__ == "__main__":
    main()
