# Запуск локально (macOS)

Всё гоняется на маке. v6 НЕ требует `unified_planning` (PDDL выкинут) — только `openai` + `pyyaml` + Docker.
(`unified_planning` нужен лишь для старых v3/v4 PDDL-демо.)

## Предпосылки
- **Python 3.14** (brew): `/opt/homebrew/opt/python@3.14` — наш код использует синтаксис `str | None` (нужен ≥3.10).
- **Docker Desktop** — запущен (`open -a Docker`; на mac sudo НЕ нужен, контекст `desktop-linux`).
- **`.env`** с боевыми ключами Azure — в корне репо (`/private/tmp/ccc/.env`). Забор с srv:
  `rsync -az srv:Puancare/.env /private/tmp/ccc/.env`  (в `.gitignore`, не коммитится).

## Установка venv
```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/pip install openai pyyaml
```

## Запуск демо (из корня репо)
```bash
P=/private/tmp/ccc; V=$P/.venv/bin/python

# 1) чистый python (без docker/LLM): распознавание категорий из потока
PYTHONPATH=$P $V -m devops_agent.v6.run

# 2) живой NLU (openai + .env): сырьё → JSON-фрагмент
PYTHONPATH=$P $V -m devops_agent.v6.run --live

# 3) docker (Docker Desktop, без LLM): заземление — реальность срезает галлюцинации
PYTHONPATH=$P $V -m devops_agent.v6.grounding

# 4) полный живой конвейер (LLM + docker): NLU→заземление→распознавание→DAG
PYTHONPATH=$P $V -m devops_agent.v6.pipeline
```

## Заметки
- `_run` в `devops_agent/v4/sandbox.py` добавляет `sudo` только на Linux (srv-ubuntu); на mac — без sudo.
- Первый docker-запуск тянет `python:3.11-slim` (один раз).
- Старые v3/v4 PDDL-демо локально потребуют `pip install unified-planning up-fast-downward` (для v6 не нужно).
