"""
E3 — РОБАСТНОСТЬ РЕФЬЮТА. «docker переспоривает LLM» — свойство или мой подобранный кейс?

В v4 рефьют показан N=1 на окне pool [80,100], которое Я выбрал так, чтобы удвоение его
перелетело. Атака: может, рефьют срабатывает только на «удобных» окнах. Гоняем N СЛУЧАЙНЫХ
окон (живой LLM). Если петля СХОДИТСЯ на любом окне — рефьют+восстановление робастны, не подгонка.
Сущность — только pool (без mem/config), чтобы изолировать non-monotone.
"""

import random
import sys

from devops_agent.v4.graph import KnowledgeGraph
from devops_agent.v4.loop import run_task
from devops_agent.v4.proposer import Proposer
from devops_agent.v4.sandbox import Sandbox

N = 10
_POW2 = [64, 128, 256, 512, 1024]      # шаги удвоения от base=64


def _doubling_lands(lo, hi):
    return any(lo <= v <= hi for v in _POW2)


def main() -> None:
    print(f"=== E3: робастность рефьюта (N={N} СЛУЧАЙНЫХ окон pool, живой LLM) ===\n")
    rng = random.Random(7)
    levers = {"pool": {"type": "numeric"}}
    defaults = {"pool": 64}
    proposer = Proposer()
    rows = []
    for i in range(N):
        lo = rng.randint(120, 400)
        hi = lo + rng.randint(40, 140)
        truth = {"app": {"pool_lo": lo, "pool_hi": hi}}    # без footprint/good_config → только pool
        sb = Sandbox(truth, network="v5e3")
        sb.prefetch() if i == 0 else None
        sb.reset()
        try:
            r = run_task(KnowledgeGraph(), sb, proposer, "app", levers, defaults,
                         learned={}, verbose=False)
        finally:
            sb.teardown()
        landed = _doubling_lands(lo, hi)
        v = r.values.get("pool")
        rows.append((lo, hi, r.success, r.refutes, v, landed))
        print(f"  run {i:2d}: окно[{lo},{hi}] landed_pow2={landed!s:5} → "
              f"success={r.success} рефьютов={r.refutes} pool={v}")

    success_rate = sum(1 for _, _, s, _, _, _ in rows if s) / N
    refuted = sum(1 for _, _, _, rf, _, _ in rows if rf >= 1)
    # рефьют ДОЛЖЕН срабатывать там, где удвоение НЕ попадает в окно
    misaligned = [(lo, hi) for lo, hi, s, rf, _, landed in rows if (rf >= 1) == landed]
    print(f"\nсходимость: {success_rate:.0%}  | рефьютов было: {refuted}/{N}  | "
          f"рассогласований рефьют↔перелёт: {len(misaligned)}")
    print("=" * 62)
    if success_rate >= 0.9 and len(misaligned) <= 1:
        print("ВЕРДИКТ E3: ВЫЖИЛ ✓ — петля сходится на СЛУЧАЙНЫХ окнах; рефьют срабатывает там,")
        print("  где удвоение перелетает окно (LLM реально неправ), а не произвольно. Не подгонка.")
        print("  'docker переспоривает LLM' — принципиальное свойство, не мой подобранный кейс.")
    else:
        print(f"ВЕРДИКТ E3: ПАДЁТ/ХРУПКО ✗ — сходимость {success_rate:.0%} и/или рефьют рассогласован.")
        print(f"  проблемные окна: {misaligned}")


if __name__ == "__main__":
    main()
