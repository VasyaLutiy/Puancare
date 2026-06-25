"""
run.py — РЕШАЮЩИЙ ЗАМЕР: задача «подними db2 здоровой» при ХОЛОДНОЙ базе vs ТЁПЛОЙ.

COLD: пустая база → агент заземляет db2 с нуля (дорого).
WARM: сперва решает db1 (учит postgres:15), потом ту же задачу для db2 → узнаёт, заземление пропускает.
Если WARM дешевле и так же верно → база ОПРАВДАНА (потребитель есть). Нет разницы → бесполезна.
"""

import sys

from devops_agent.v6.grounding import Grounder
from devops_agent.v6.nlu import LiveNLU
from devops_agent.v7.task_agent import CountingSandbox, solve

TRUTH = {
    "db1": {"footprint": 350, "deps": ["cache1"]},
    "db2": {"footprint": 380, "deps": ["cache1"]},   # структурно как db1 (postgres:15)
    "cache1": {"footprint": 64},
}
RAW = {
    "db1": {"image": "postgres:15", "connects_to": ["cache1"]},
    "db2": {"image": "postgres:15", "connects_to": ["cache1"]},
    "cache1": {"image": "redis:7", "connects_to": []},
}


def main() -> None:
    print("=== v7 ЗАМЕР: задача 'подними db2' — ХОЛОДНАЯ база vs ТЁПЛАЯ ===\n")
    CountingSandbox(TRUTH, network="v7task").prefetch()

    # COLD: пустой KB, сразу db2
    sb = CountingSandbox(TRUTH, network="v7task"); sb.reset()
    nlu, gr, kb = LiveNLU(), Grounder(sb), {}
    try:
        ok_c, cost_c = solve(("db2", RAW["db2"]), kb, sb, nlu, gr)
    finally:
        sb.teardown()
    print(f"COLD  db2: ok={ok_c}  LLM={cost_c['llm']}  docker-проб={cost_c['provisions']}  [{cost_c['path']}]")

    # WARM: пустой KB, сперва db1 (учится postgres:15), затем db2
    sb = CountingSandbox(TRUTH, network="v7task"); sb.reset()
    nlu, gr, kb = LiveNLU(), Grounder(sb), {}
    try:
        ok_w1, _ = solve(("db1", RAW["db1"]), kb, sb, nlu, gr)        # разогрев: учит postgres:15
        ok_w, cost_w = solve(("db2", RAW["db2"]), kb, sb, nlu, gr)    # та же задача, теперь тёплая
    finally:
        sb.teardown()
    print(f"WARM  db2: ok={ok_w}  LLM={cost_w['llm']}  docker-проб={cost_w['provisions']}  [{cost_w['path']}]  "
          f"(после разогрева db1, ok={ok_w1})")

    print("=" * 64)
    saved_p = cost_c["provisions"] - cost_w["provisions"]
    saved_llm = cost_c["llm"] - cost_w["llm"]
    if ok_c and ok_w and saved_p > 0:
        print(f"ВЕРДИКТ v7: БАЗА ОПРАВДАНА ✓ — обе задачи решены (running), но ТЁПЛАЯ дешевле:")
        print(f"  docker-проб {cost_c['provisions']}→{cost_w['provisions']} (−{saved_p}), LLM {cost_c['llm']}→{cost_w['llm']} (−{saved_llm}).")
        print("  Узнавание по базе ПРОПУСКАЕТ заземление → знание переиспользовано, а не переоткрыто.")
        print("  Это первый раз, когда KB что-то ДАЁТ: потребитель (задача) её прочитал и сэкономил.")
    else:
        print(f"ВЕРДИКТ v7: база НЕ оправдана — ok_c={ok_c} ok_w={ok_w} экономия проб={saved_p}")


if __name__ == "__main__":
    main()
