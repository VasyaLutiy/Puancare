# DevOps-агент v2 — операбельный PDDL + numeric-память + декларативная онтология

## Откуда стартуем (итог v1)

v1 дошёл до **M7** и вердикта §8 — **ЖИВ**: `trials` падает (4→1), `oracle_calls→0`
(пик 1 на ep5, хвост 0), агент детерминирован (distinct=1 против 2 у LLM-baseline).
Гипотеза «саморемонт модели мира из провалов» доказана на узком кейсе (память/OOM + config).

v2 **не меняет гипотезу и не ослабляет критерий убийства** — он чинит *репрезентацию*,
чтобы модель стала операбельной и масштабируемой.

## Что именно лечим (диагноз)

В v1 три разных «хардкода», и важно их не путать:

1. **`compile_to_up` строит домен императивно** (UP API: `Fluent`, `InstantaneousAction`,
   `add_precondition`). Домена-как-объекта не существует — его нельзя `cat`, отредактировать,
   отдать будущему конструктору-редактору. Хуже: ветка `has_config` **переписывает сам домен**
   под инстанс.
2. **`world.py` if/elif (`exit_code==3 → unhealthy`)** — это **сенсор**, перевод реальности
   (docker State) в словарь симптомов. Его **нельзя** «парсить из PDDL»: PDDL знает предикаты
   планирования, но не знает `exit_code==3`. Это отдельный слой.
3. **Словарь симптомов/причин размазан по трём файлам**: `world.py` (obs→phase),
   `agent.py` (`_classify_symptom`, `_OBVIOUS_CAUSES`, `_FALLBACK_CAUSES`, применение фиксов),
   `oracle.py` (`CAUSES`). Это и есть «ужас» — лечится не PDDL'ем, а онтологией.

## Зафиксированные решения (с пользователем)

- **Единый always-on домен.** Один статичный `domain.pddl`. config-измерение всегда включено;
  сервисы без config получают `config_ok=true` для всех opt. Ноль ветвлений в коде.
- **Онтология во внешних YAML.** `ontology.yaml` — единый источник правды для sensor-правил,
  словаря симптомов/причин и `cause→repair`. Добавляем зависимость `pyyaml`.
- **Numeric-память (ENHSP).** Бакеты убираются совсем. Память — вещественный fluent;
  выученная величина — порог `min_safe_mem`. Это меняет домен, мир, BIOS и метрики.

## Архитектура v2 — три слоя, три источника правды

```
1. ПЛАНИРОВОЧНЫЙ ДОМЕН (PDDL)          devops_agent/pddl/domain.pddl     ← статичный, рукописный
   типы / предикаты / numeric functions    (цель будущего конструктора-редактора)
   / actions / :metric

2. ГЕНЕРАТОР ПРОБЛЕМЫ                   devops_agent/model/problem.py
   BiosState + goal → problem.pddl(текст) → PDDLReader.parse_problem() → up.Problem
   (проекция выученных знаний BIOS в :objects/:init/:goal; умеет дамп на диск)

3. ОНТОЛОГИЯ МИРА/ДИАГНОСТИКИ           devops_agent/model/ontology.yaml + ontology.py
   sensor-правила (obs→phase) · словарь симптомов · причины (CAUSES) · cause→repair
   (схлопывает ВСЕ if/elif из world.py / agent.py / oracle.py)
```

**Связь слоёв:** правило `cause→repair` из онтологии правит ровно те fluents
(`min_safe_mem`, `config_ok`), что объявлены в `domain.pddl`. «Conditions» из твоего примера
живут как **данные** в онтологии и ссылаются на предикаты домена — не как `if/elif`.

## Numeric-модель (главное изменение)

Бакеты-как-объекты исчезают. Память — вещественный fluent на сервис.

```pddl
(define (domain devops)
  (:requirements :typing :numeric-fluents :action-costs)
  (:types service configopt)
  (:predicates
    (running       ?s - service)
    (mem_set       ?s - service)
    (config_set    ?s - service ?o - configopt)
    (config_ok     ?s - service ?o - configopt))
  (:functions
    (mem           ?s - service)   ; выделенная память [MiB] — решение
    (min_safe_mem  ?s - service)   ; ВЫУЧЕННЫЙ порог (проецируется из BIOS)
    (total-cost))

  ; set_mem: выделить минимально-безопасную память. Стоимость = выделенный объём.
  (:action set_mem
    :parameters (?s - service)
    :effect (and (mem_set ?s)
                 (assign (mem ?s) (min_safe_mem ?s))
                 (increase (total-cost) (min_safe_mem ?s))))

  (:action set_config
    :parameters (?s - service ?o - configopt)
    :effect (config_set ?s ?o))

  (:action deploy
    :parameters (?s - service ?o - configopt)
    :precondition (and (mem_set ?s)
                       (>= (mem ?s) (min_safe_mem ?s))
                       (config_set ?s ?o)
                       (config_ok ?s ?o))
    :effect (running ?s))

  (:metric minimize (total-cost)))
```

**Кто что делает (честная граница):**
- **Планировщик** секвенирует `set_config → set_mem → deploy`, доказывает достижимость цели и
  минимизирует выделенную память. Числовое значение он *читает* из модели.
- **Обучение** (петля агента) добывает само число `min_safe_mem` интервенцией в мире.
  Это полностью соответствует тезису проекта: **учим модель мира, не политику**; планировщик
  потребляет модель.

## Схема знаний BIOS (меняется под numeric)

Уходят `unsafe_mem`/`unsafe_svc` (множества бакетов). Приходят **пороги**:

```python
mem_threshold:     dict[str, int]   # workload_class → выученный min safe [MiB]  (класс-правило)
mem_threshold_svc: dict[str, int]   # svc_name      → per-service override       (карвинг M5b)
bad_config:        set[tuple[str,str]]   # как в v1
known_safe:        set[tuple[str,int]]   # подтверждённые (svc, mem) running
reverse_index:     dict[str, list[str]]  # как в v1 (НЕ засеваем — зарабатываем опытом)
```

Проекция в `:init`: `min_safe_mem(s) = mem_threshold_svc.get(s) or mem_threshold[wc] or MAX_BOUND`.
До первого знания о классе — `MAX_BOUND` (консервативно безопасно, план дорогой → есть что
минимизировать обучением).

## Петля обучения = numeric-бисекция

v1 «помечал бакет unsafe». v2 сужает интервал на вещественной оси:
- держим на класс `lo_safe` (наименьший known-running) и `hi_unsafe` (наибольший known-OOM);
- OOM при `X` → `hi_unsafe = max(hi_unsafe, X)`; running при `X` → `lo_safe = min(lo_safe, X)`;
- следующая проба = середина `(hi_unsafe, lo_safe)`, стоп при гранулярности `STEP` (напр. 32 MiB);
- `min_safe_mem` ← `lo_safe`. Перенос на новый сервис того же класса → 1 проба (как M4).

Метрики §8 сохраняются: trials-to-converge = шаги бисекции (первый раз `O(log(range/STEP))`,
повтор → 1); oracle-кривая, детерминизм, человеко-правки — без изменений.

## Онтология (YAML-схема)

```yaml
sensors:                       # реальность → словарь симптомов (заменяет world.py if/elif)
  - when: {oom_killed: true}            then: oom_killed
  - when: {exit_code: 0}                then: running
  - when: {exit_code: 3}                then: unhealthy
  - default:                            error
symptoms: [running, oom_killed, unhealthy, error]
causes:   [memory, config]              # источник CAUSES для oracle.py
obvious:  {oom_killed: memory}          # _OBVIOUS_CAUSES — оракул не нужен
fallback: {unhealthy: config}           # _FALLBACK_CAUSES при oracle=None
repair:                                 # cause → как править BIOS (ссылается на предикаты домена)
  memory: {kind: numeric_threshold, fluent: min_safe_mem, carve: true}
  config: {kind: set_member,        store: bad_config}
```

Один файл — единственный источник правды. `world.py`, `agent.py`, `oracle.py` читают его,
никаких локальных таблиц-словарей.

## Раскладка модулей v2

```
devops_agent/
  pddl/domain.pddl          # статичный домен (numeric, always-on config)
  model/
    domain.py               # PddlDomain: загрузка domain.pddl, метаданные, write()
    problem.py              # ProblemBuilder: BiosState+goal → problem.pddl → up.Problem (+ dump)
    ontology.py             # WorldModel: загрузка ontology.yaml, sensor/symptom/cause/repair API
    ontology.yaml
  bios.py                   # ТОЛЬКО хранилище знаний (пороги). compile_to_up УДАЛЁН.
  planner.py                # plan(bios, goal): ProblemBuilder + OneshotPlanner(ENHSP/FD)
  world.py                  # apply(): docker --memory=<число>; obs→phase через WorldModel
  agent.py                  # фиксы через ontology.repair (table-driven, без if/elif)
  oracle.py                 # CAUSES из ontology
  run_experiment.py         # харнесс M7 — перепроверка §8 на numeric-мире
```

## Милстоуны v2 (каждый верифицируется отдельно)

- **V2-M0 — smoke-тест стека (риск-гейт, делаем ПЕРВЫМ).** На VPS: `pip install up-enhsp pyyaml`.
  Проверить, что `PDDLReader` парсит `domain.pddl` (`:numeric-fluents` + `:action-costs` +
  `assign`/`increase` с fluent-RHS) и что `OneshotPlanner` (ENHSP-opt) решает тривиальную
  проблему оптимально. **Если не парсит/не решает — пересматриваем numeric до кода.**
- **V2-M1 — PDDL-домен + генератор проблемы.** `domain.pddl` + `PddlDomain` + `ProblemBuilder`
  (BIOS→problem.pddl→Problem, дамп на диск). `planner.py`. `compile_to_up` удалён.
  Регресс: план для известного порога идентичен ожиданиям (теперь 3 шага, numeric).
- **V2-M2 — онтология.** `ontology.yaml` + `WorldModel`. `world.py`/`agent.py`/`oracle.py`
  переключены на неё; локальные словари/if-elif удалены. Поведение байт-в-байт как v1 на тех же
  входах (онтология воспроизводит текущие правила).
- **V2-M3 — numeric-мир + бисекция.** `world.py` снимает ограничение `MEM_BUCKETS`
  (`--memory=<число>m`). BIOS: пороги вместо множеств. Агент: numeric-бисекция + перенос + карвинг.
- **V2-M4 — перепрогон харнесса.** `run_experiment.py` на numeric-мире → вердикт §8.
  **Должен остаться ЖИВ** (trials↓, oracle→0, детерминизм). Иначе — чиним или хороним честно.

## Честные риски v2

- **ENHSP-доступность/совместимость** — главный гейт (V2-M0). Версия UP/движка локально не стоит
  (прогон на VPS). `assign`-эффект с fluent-RHS и `:action-costs` одновременно — проверить, что
  движок берёт это в *оптимальном* режиме.
- **«Тонкая» роль планировщика в numeric** — при одном сервисе минимум = сам порог, планировщик
  почти ничего не «решает». Интересное (поиск числа) — в обучении. Это честно по тезису проекта,
  но надо явно проговаривать, чтобы не выдавать за «numeric planning ради planning».
- **Детерминизм numeric-планировщика** — метрика B (distinct=1) требует стабильного вывода;
  ENHSP-эвристики могут давать варьирующиеся, но равно-оптимальные планы → нормализовать сравнение
  по стоимости, не по строке.
- **Round-trip PDDL-текста** — детерминированный порядок объектов/инициализации сохранить
  (как в v1), иначе плывёт сравнение планов.
- **Сходимость бисекции** — нужна верхняя безопасная граница `MAX_BOUND` и гранулярность `STEP`;
  без них поиск не завершается / даёт ложный порог.

## Критерий убийства — БЕЗ изменений

Тот же §8: если trials не падает, ИЛИ перенос не работает, ИЛИ LLM-кривая не снижается —
хороним честно. Судья — песочница (Docker), не мнение LLM. v2 обязан воспроизвести вердикт ЖИВ
на numeric-мире; иначе репрезентационный рефактор сломал суть — и это надо признать.
```
```

## Открытый вопрос на потом (НЕ в этом заходе)

Мета-домен (эпистемическое планирование экспериментов: планировщик сам решает, *какую* пробу
ставить) — по-прежнему отложен. v2 оставляет бисекцию захардкоженной в петле, как v1 оставлял
бинарный поиск по бакетам.
