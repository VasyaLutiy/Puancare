"""
devops_agent.v3 — MVP `v3_meta_bios_pddl`: самовыращиваемый PDDL-домен.

Полный дизайн: ../../DevopsPlanV3.md. Гипотеза: механизм v2 `propose→verify→amortize`,
поднятый на уровень выше, выращивает САМУ схему (предикаты+интервенции), а не только
значения. Домен ГЕНЕРИТСЯ из реестра, не пишется. Критерий ЖИВ: llm_calls/задача → 0.

═══════════════════════════════════════════════════════════════════════════════
СТАТУС: СКЕЛЕТ (signatures + docstrings; тела = TODO(v3)).  Создан в сессии-дизайне.
═══════════════════════════════════════════════════════════════════════════════

ПОРЯДОК ДОПИСЫВАНИЯ (снизу вверх, каждый слой тестируется на srv:~/Puancare):
  1. meta_schema.py   — мета-типы + PROPOSAL_SCHEMA + canonical_predicate_name/validate (чистое, unit-тест).
  2. domain_gen.py    — MetaDomain → domain.pddl/problem.pddl + solve() (копия bios.plan). КЕЙСТОУН:
                        прогнать сгенерённый домен через FD на srv, сверить с эталоном v2 (3 экшена).
  3. strategies.py    — DoublingStrategy / EnumerateStrategy (порт v2 agent.py / _choose_config).
  4. meta_bios.py     — MetaBios.seed/mint_dimension/value_for/record_running (зависит от 1,3).
  5. proposer.py      — ReplayProposer (offline) + LiveProposer (AzureJSON). Fixtures уже записаны.
  6. meta_agent.py    — петля run_episode (зависит 2-5 + world). --selftest в replay (без токенов).
  7. harness_v3.py    — §8-v3: кривая llm_calls↓, детерминизм, F'. Живой LLM. Вердикт.

РАБОЧИЙ ЦИКЛ: правки локально → rsync devops_agent/v3 на srv → прогон на srv
(локальный .venv без unified_planning/docker; всё реальное — на srv).
Возобновление: `git checkout v3_meta_bios` + читать DevopsPlanV3.md.
"""
