"""
E2 — ШУМ → ЛОЖНОЕ ЗНАНИЕ. Падает ли verify→trust под недетерминизмом?

Закон агента: один успех → доверяй. Под шумом транзиентный ложный успех на НЕБЕЗОПАСНОМ
значении заставляет петлю остановиться рано и зафиксировать ложный порог.

Чтобы изолировать ШУМ (а не добычу симптомов LLM): измерение mem засеяно, LLM запрещён.
Гоняем настоящую v4-петлю N раз на шумной песочнице; считаем долю прогонов, сошедшихся на
значении НИЖЕ истинно-безопасного. Любая ложь = закон под шумом неверен.
"""

import sys

from devops_agent.v4.graph import KnowledgeGraph
from devops_agent.v4.loop import run_task
from devops_agent.v5_research.noisy_sandbox import NoisySandbox

TRUE_SAFE = 512        # footprint 350 → безопасно ≥512 (256 → OOM). Истина среды.
N = 12
FLAKE_P = 0.3


class _NoProposer:
    """LLM-запрет: если измерение засеяно, propose не должен вызываться. Изолируем шум."""
    n_calls = 0

    def propose(self, *a, **k):
        raise AssertionError("LLM вызван — а должен быть засеян (эксперимент о шуме, не о добыче)")


def main() -> None:
    print(f"=== E2: ШУМ → ЛОЖНОЕ ЗНАНИЕ (flake_p={FLAKE_P}, N={N}, истинно-безопасно≥{TRUE_SAFE}) ===\n")
    truth = {"app": {"footprint": 350}}
    levers = {"mem": {"type": "numeric"}}
    defaults = {"mem": 64}
    seed_learned = {"oom_killed": {"lever": "mem", "kind": "ordered_monotone"}}

    NoisySandbox(truth, network="v5e2").prefetch()
    converged = []
    for i in range(N):
        sb = NoisySandbox(truth, flake_p=FLAKE_P, seed=1000 + i, network="v5e2")
        sb.reset()
        try:
            r = run_task(KnowledgeGraph(), sb, _NoProposer(), "app", levers, defaults,
                         learned=dict(seed_learned), verbose=False)
        finally:
            sb.teardown()
        v = r.values.get("mem") if r.success else None
        converged.append(v)
        print(f"  run {i:2d}: сошёлся на mem={v}  [{'ЛОЖНО' if (v is None or v < TRUE_SAFE) else 'ок'}]")

    false = [v for v in converged if v is None or v < TRUE_SAFE]
    rate = len(false) / N
    print(f"\nложных порогов: {len(false)}/{N} ({rate:.0%}) — ниже истинно-безопасного {TRUE_SAFE}")
    print("=" * 62)
    if false:
        print(f"ВЕРДИКТ E2: ПАДЁТ ✗ — verify→trust фиксирует ЛОЖНОЕ знание под шумом.")
        print(f"  {rate:.0%} прогонов сошлись на небезопасном значении из-за транзиентного успеха.")
        print("  Следствие: персистентный граф под недетерминизмом = уверенно-неверен.")
        print("  Нужен СТАТИСТИЧЕСКИЙ verify (N проб / доверие p), не 'один успех → доверяй'.")
    else:
        print("ВЕРДИКТ E2: ВЫЖИЛ ✓ — под шумом ложного знания не зафиксировано (неожиданно — проверить flake_p).")


if __name__ == "__main__":
    main()
