#!/usr/bin/env python3
"""
Step 0 — TDA-ГЕЙТ (ResearchPuancare.md §5)

Цель: ДО постройки потенциала h измерить топологию noun-subtree WordNet.
Если посылка «дерево» чиста (β₁ ≈ 0) — строим h спокойно.
Если петель много и они устойчивы — посылка ложна, закладываем структуру под петли.

Метод (честный и точный, без хрупких PH-библиотек):
Иерархия гипернимии — это граф (1-комплекс). Для графа числа Бетти
вычисляются ТОЧНО, а не оценочно:
    β₀ = число компонент связности (кластеры)
    β₁ = E - V + β₀                      (circuit rank — число независимых петель)
Чистое дерево: E = V-1, β₀ = 1  =>  β₁ = 0.   ← ровно посылка плана.

Петли в WordNet берутся из МНОЖЕСТВЕННОГО НАСЛЕДОВАНИЯ (synset с >1 гипернимом).
Это и есть «полисемия как мостик» из гипотезы — мы её здесь измеряем, а не предполагаем.

Это эквивалентно β₁ персистентной гомологии клик-комплекса графа на пороге
включения всех рёбер: для 1-комплекса H_1 совпадает с circuit rank.
"""

import sys
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn


def ensure_wordnet():
    try:
        wn.synsets("dog")
    except LookupError:
        print("[setup] downloading wordnet ...", file=sys.stderr)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def collect_subtree(root_name="animal.n.01"):
    """Транзитивное замыкание гипонимов root + все рёбра гипернимии ВНУТРИ множества."""
    root = wn.synset(root_name)

    # 1. множество вершин = root + все его потомки (по hyponym)
    nodes = set()
    stack = [root]
    while stack:
        s = stack.pop()
        if s in nodes:
            continue
        nodes.add(s)
        stack.extend(s.hyponyms())

    # 2. рёбра = пары (child, parent) по hypernym, где ОБА конца внутри nodes.
    #    направленность нам сейчас не важна (β₁ считаем на неориентированном графе).
    edges = set()
    parents = defaultdict(set)   # child -> set(parents)  внутри subtree
    for s in nodes:
        for h in s.hypernyms():
            if h in nodes:
                a, b = sorted((s.name(), h.name()))
                edges.add((a, b))
                parents[s].add(h)
    return root, nodes, edges, parents


def connected_components(nodes, edges):
    """β₀ — число компонент связности (union-find)."""
    parent = {n.name(): n.name() for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    roots = {find(n.name()) for n in nodes}
    return len(roots)


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "animal.n.01"
    ensure_wordnet()

    root, nodes, edges, parents = collect_subtree(root_name)

    V = len(nodes)
    E = len(edges)
    b0 = connected_components(nodes, edges)
    b1 = E - V + b0  # circuit rank = первое число Бетти графа

    multi_parent = {s: ps for s, ps in parents.items() if len(ps) > 1}

    print("=" * 64)
    print(f"  TDA-ГЕЙТ  —  noun-subtree:  {root_name}")
    print("=" * 64)
    print(f"  V (синсеты-вершины)        : {V}")
    print(f"  E (рёбра гипернимии)       : {E}")
    print(f"  β₀ (компоненты / кластеры) : {b0}")
    print(f"  β₁ (независимые петли)     : {b1}")
    print("-" * 64)
    print(f"  узлов с >1 родителем       : {len(multi_parent)}  (источник петель)")
    if E == V - 1 and b0 == 1:
        print("  форма                      : ЧИСТОЕ ДЕРЕВО (E = V-1)")
    print("=" * 64)

    # --- решение гейта ---
    ratio = b1 / V if V else 0.0
    print(f"\n  β₁ / V = {ratio:.4f}")
    if b1 == 0:
        verdict = "β₁ = 0  ->  дерево идеально. Строим h ✅"
    elif ratio < 0.01:
        verdict = (f"β₁ мал ({b1}, {ratio:.2%} от V)  ->  посылка дерева чиста. "
                   f"Строим h ✅ (петли как локальные поправки)")
    else:
        verdict = (f"β₁ ВЕЛИК ({b1}, {ratio:.2%} от V)  ->  иерархия не дерево. "
                   f"Закладываем структуру под петли ❌ (одного h мало)")
    print(f"  ВЕРДИКТ: {verdict}\n")

    # топ источников петель — что именно рвёт дерево
    if multi_parent:
        print("  Топ-12 узлов с множественным наследованием (рвут дерево):")
        top = sorted(multi_parent.items(), key=lambda kv: -len(kv[1]))[:12]
        for s, ps in top:
            pnames = ", ".join(sorted(p.name() for p in ps))
            print(f"    {s.name():28s} <- {{{pnames}}}")


if __name__ == "__main__":
    main()
