"""
GD2 — ЗАЗЕМЛЕНИЕ ПОД ШУМОМ. Наследует ли grounding провал E2?

Claim M2/M3: «заземление пробой срезает галлюцинации → реальность переспоривает LLM».
Но grounding делает ОДНУ пробу на элемент = ровно `verify→trust` после одного наблюдения (E2).
Атака: шумная песочница (провал транзиентно «проходит»). Тогда проба РЕАЛЬНОГО рычага/связи
(стресс→провал) может мигнуть в running → grounding решит «не важно» → ЛОЖНО СРЕЖЕТ реальное.
Гоняем N раз; считаем, как часто заземлённая структура НЕВЕРНА (потеряли mem/cache).
"""

import sys

from devops_agent.v5_research.noisy_sandbox import NoisySandbox
from devops_agent.v6.grounding import Grounder

N = 8
FLAKE_P = 0.3
TRUTH = {"db": {"footprint": 350, "deps": ["cache"]}, "cache": {"footprint": 64}}
CORRECT = ({"mem"}, {"cache"})    # истинная заземлённая структура db


def main() -> None:
    print(f"=== GD2: заземление под шумом (flake_p={FLAKE_P}, N={N}) ===\n")
    print(f"  истина: db = levers{{mem}} + dep{{cache}}; claimed levers=[mem,config] deps=[cache]\n")
    NoisySandbox(TRUTH, network="gd2").prefetch()
    wrong = 0
    for i in range(N):
        sb = NoisySandbox(TRUTH, flake_p=FLAKE_P, seed=200 + i, network="gd2")
        sb.reset()
        try:
            grounded, _ = Grounder(sb).ground("db", ["mem", "config"], ["cache"])
        finally:
            sb.teardown()
        if grounded is None:
            wrong += 1
            print(f"  run {i}: baseline мигнул → заземление НЕ удалось [НЕВЕРНО]")
            continue
        got = (set(grounded["levers"]), set(grounded["deps"]))
        ok = got == CORRECT
        wrong += 0 if ok else 1
        print(f"  run {i}: заземлено levers={sorted(got[0])} deps={sorted(got[1])}  "
              f"[{'ок' if ok else 'НЕВЕРНО — потеряли реальное'}]")

    rate = wrong / N
    print(f"\n  неверных заземлений: {wrong}/{N} ({rate:.0%})")
    print("=" * 64)
    if wrong > 0:
        print("ВЕРДИКТ GD2: ПАДЁТ ✗ — заземление = ОДНА проба = verify→trust (E2) → под шумом")
        print(f"  ЛОЖНО срезает РЕАЛЬНУЮ структуру в {rate:.0%} прогонов (мигнувший провал → «не важно»).")
        print("  «Реальность переспоривает LLM» держится лишь в ДЕТЕРМИНИРОВАННОМ мире; под шумом")
        print("  grounding сам уверенно-неверен. Нужна СТАТИСТИЧЕСКАЯ проба (N раз/доверие) — как и в E2.")
    else:
        print("ВЕРДИКТ GD2: устоял (неожиданно — проверить flake_p/проб).")


if __name__ == "__main__":
    main()
