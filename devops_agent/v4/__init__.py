"""
devops_agent.v4 — Knowledge Graph: знание агента живёт в типизированном графе (6 мета-типов).

Полный замысел: ../../DevopsPlanV4.md. Сдвиг: знание = граф (не PDDL); 6 мета-типов = ISA;
PDDL meta-domain = упаковщик/ground-truth (что не проецируется в валидный PDDL — не входит в граф).

Слои (снизу вверх):
  graph.py       — KnowledgeGraph (узлы/рёбра + JSON).
  contract.py    — структурный ground-truth (инварианты 6 типов + легальные рёбра).
  projection.py  — PDDL ground-truth (KG→PDDL + FD), + keystone selftest (контент-нейтральность).
  packer.py      — (след.) фрагмент → validate → commit|reject; поглощает v3 mint_dimension.

Прогон keystone: на srv:~/Puancare — `venv/bin/python -m devops_agent.v4.projection --selftest`.
"""
