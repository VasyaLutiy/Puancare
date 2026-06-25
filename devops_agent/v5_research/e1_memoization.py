"""
E1 — MEMOIZATION vs ОБУЧЕНИЕ. «LLM→0» — это обобщение по СТРУКТУРЕ или кэш по строке?

3 рычага ОДНОГО kind (ordered_monotone), но РАЗНЫЕ симптомы (low_buf0/1/2). Живой LLM.
  обобщающий агент → 1 вызов (выучил kind, применил ко всем того же типа);
  кэширующий агент → 3 вызова (платит за каждую новую строку симптома).
Если вызовов == числу различных симптомов при ОДНОМ kind → амортизация мнимая (мемоизация).
"""

import sys

from devops_agent.v4.graph import KnowledgeGraph
from devops_agent.v4.loop import run_task
from devops_agent.v4.proposer import Proposer
from devops_agent.v5_research.multibuf_sandbox import MultiBufSandbox


def main() -> None:
    print("=== E1: memoization vs обучение (3 рычага ОДНОГО kind, РАЗНЫЕ симптомы) ===\n")
    bufs = {"buf0": 200, "buf1": 200, "buf2": 200}     # все ordered_monotone (порог 200 → 256 ок)
    levers = {b: {"type": "numeric"} for b in bufs}
    defaults = {b: 64 for b in bufs}

    sb = MultiBufSandbox(bufs)
    sb.prefetch(); sb.reset()
    proposer = Proposer()
    learned = {}
    try:
        r = run_task(KnowledgeGraph(), sb, proposer, "app", levers, defaults,
                     learned=learned, verbose=True)
    finally:
        sb.teardown()

    kinds = {p["kind"] for p in learned.values()}
    distinct = len(learned)
    print(f"\nуспех={r.success} trials={r.trials} LLM-вызовов={r.llm_calls} (токенов={proposer.total_tokens})")
    print(f"различных симптомов выучено: {distinct}; различных kind среди них: {kinds}")
    print("=" * 62)
    if r.llm_calls >= distinct and distinct >= 2 and len(kinds) == 1:
        print(f"ВЕРДИКТ E1: ПАДЁТ ✗ — {r.llm_calls} LLM-вызовов на {distinct} симптома ОДНОГО kind {kinds}.")
        print("  Агент платит LLM за КАЖДУЮ новую строку симптома, хотя структура идентична.")
        print("  Это мемоизация по строке, НЕ обобщение по структуре. «LLM→0» — лишь для ПОВТОРОВ.")
        print("  Нужен retrieval по СТРУКТУРНОЙ сигнатуре (v5 apply): 1 вызов → применить ко всем того же kind.")
    else:
        print("ВЕРДИКТ E1: ВЫЖИЛ ✓ — агент обобщил по структуре (неожиданно — проверить).")


if __name__ == "__main__":
    main()
