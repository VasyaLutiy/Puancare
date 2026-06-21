#!/usr/bin/env python3
"""Быстрая проверка текущей формулы стоимости на разнотипных парах (в домене animal)."""
from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic, cost as cost_fn, is_a, resolve

ensure_wordnet()
root, nodes, edges, parents = collect_subtree("animal.n.01")
h_ic, _ = build_h_ic(root, nodes, parents)

PAIRS = [
    ("carp", "fish", "is-a близкая"),
    ("goldfish", "fish", "is-a близкая"),
    ("pike", "fish", "is-a близкая"),
    ("whale", "fish", "cousin БЛИЗКАЯ (трудная)"),
    ("fish", "dog", "cousin ДАЛЁКАЯ (лёгкая)"),
    ("shark", "dog", "cousin ДАЛЁКАЯ (лёгкая)"),
    ("salmon", "eagle", "cousin ДАЛЁКАЯ (лёгкая)"),
]
print(f"{'A':9s} {'B':9s} {'A is-a B':>9s} {'cost A→B':>9s} {'cost B→A':>9s}  тип")
for wa, wb, tag in PAIRS:
    a, b = resolve(wa, nodes), resolve(wb, nodes)
    if not (a and b):
        print(f"{wa:9s} {wb:9s}  (нет в домене animal)"); continue
    cab, _ = cost_fn(a, b, parents, h_ic)
    cba, _ = cost_fn(b, a, parents, h_ic)
    flag = "ДА" if is_a(a, b, parents) else "нет"
    print(f"{wa:9s} {wb:9s} {flag:>9s} {cab:9.2f} {cba:9.2f}  {tag}")
print("\nкамень/дерево — ВНЕ домена animal.n.01: h_ic для них не определён (нужен корень entity, ~80k узлов).")
