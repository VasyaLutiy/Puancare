"""
pipeline.py — ЕДИНЫЙ ЖИВОЙ ПОТОК (M1+M2+NLU в один конвейер).

На каждый сырой сигнал:
  NLU(LLM)  : сырьё → кандидат {levers,deps,status}        — восприятие (гипотеза)
  grounder  : проба в песочнице → срезать галлюцинации     — реальность переспоривает LLM
  recognizer: сигнатура ЗАЗЕМЛЁННОЙ структуры → категория   — обобщение по структуре (E1)
  place     : положить заземлённую сущность в DAG
Всё под капотом. «увидел незнакомое → понял структурно → проверил реальностью → запомнил».

Линия: LLM = восприятие без состояния; распознавание/заземление — в агенте. Без PDDL.
Категоризируем по РЕАЛЬНОСТИ (заземлённой структуре), не по claim'у LLM.
"""

import sys

from devops_agent.v4.sandbox import Sandbox
from devops_agent.v6.grounding import Grounder
from devops_agent.v6.nlu import KNOB_KIND, LiveNLU

# (id, сырьё для NLU, СКРЫТЫЙ truth для песочницы). db2 структурно = db1 → должен УЗНАТЬСЯ.
ENTITIES = [
    ("cache1", {"image": "redis:7", "listens": 6379, "role": "in-memory cache"},
     {"footprint": 64}),
    ("db1", {"image": "postgres:15", "listens": 5432, "connects_to": ["cache1"], "storage": "disk"},
     {"footprint": 350, "deps": ["cache1"]}),
    ("db2", {"image": "postgres:15", "listens": 5432, "connects_to": ["cache1"]},
     {"footprint": 380, "deps": ["cache1"]}),
    ("gw1", {"image": "nginx", "listens": 80, "config": "tls required", "upstream": "db1"},
     {"good_config": "good"}),
]


def _signature(grounded, kinds, status):
    return (tuple(sorted(kinds.get(l, "?") for l in grounded["levers"])),
            len(grounded["deps"]), bool(status))


def main() -> None:
    print("=== v6 ЕДИНЫЙ ЖИВОЙ ПОТОК: NLU → заземление → распознавание → DAG ===\n")
    truth = {eid: t for eid, _, t in ENTITIES}
    sb = Sandbox(truth, network="v6pipe")
    sb.prefetch(); sb.reset()
    nlu = LiveNLU()
    grounder = Grounder(sb)

    categories, instances, edges = {}, {}, []
    pruned_total, recognized = 0, 0

    try:
        for eid, raw, _ in ENTITIES:
            print(f"[{eid}] raw={raw}")
            cand = nlu.perceive_entity(raw, eid)
            print(f"  NLU claim: levers={cand['levers']} deps={cand['deps']} status={cand['status']!r}")

            # неизвестные миру deps = галлюцинация (авто-прун, песочнице не из чего пробовать)
            known_deps = [d for d in cand["deps"] if d in truth]
            unknown = [d for d in cand["deps"] if d not in truth]

            # Агент НЕ доверяет структурным claim'ам LLM (они НЕСОГЛАСОВАНЫ между похожими!):
            # грунтуем ВСЮ поверхность knob'ов (ОТКРЫВАЕМ структуру), deps — кандидаты от LLM (проверяем),
            # status — НАБЛЮДАЕМ (дошёл ли baseline до running). LLM даёт только ярлык + кандидатов-связи.
            grounded, verdicts = grounder.ground(eid, list(KNOB_KIND), known_deps)
            if grounded is None:
                print(f"  ⚠ baseline не получен ({verdicts}) — пропуск\n")
                continue
            for d in unknown:
                verdicts[f"dep:{d}"] = ("ПРУНЕД", "нет такой сущности в мире")
            pruned = [k for k, (v, _) in verdicts.items() if v == "ПРУНЕД"]
            pruned_total += len(pruned)
            status_grounded = True   # baseline дошёл до running → у сущности есть здоровое состояние
            print(f"  LLM labeled '{cand['label']}', claimed levers={cand['levers']} (НЕ доверяем структуре);")
            print(f"  агент ОТКРЫЛ пробой: levers={grounded['levers']} deps={grounded['deps']} | срезано: {pruned}")

            sig = _signature(grounded, KNOB_KIND, status_grounded)
            is_new = sig not in categories
            if is_new:
                categories[sig] = f"cat{len(categories)}"
            cat = categories[sig]
            recognized += 0 if is_new else 1
            instances[eid] = {"category": cat, "label": cand["label"],
                              "levers": grounded["levers"], "deps": grounded["deps"]}
            for d in grounded["deps"]:
                edges.append((eid, d))
            print(f"  распознавание: {'НОВАЯ КАТЕГОРИЯ ' + cat if is_new else 'УЗНАЛ ' + cat} "
                  f"(sig={sig})\n")
    finally:
        sb.teardown()

    print("=" * 62)
    print(f"инстансов: {len(instances)} | категорий: {len(categories)} | "
          f"узнано: {recognized} | срезано галлюцинаций: {pruned_total} | "
          f"NLU вызовов={nlu.n_calls} токенов={nlu.total_tokens}")
    print("граф (instances):")
    for iid, meta in instances.items():
        print(f"  {iid}: {meta}")
    print(f"edges: {edges}")

    db2 = instances.get("db2", {})
    db1 = instances.get("db1", {})
    db2_recognized = db2.get("category") and db2.get("category") == db1.get("category")
    print("=" * 62)
    if db2_recognized and len(categories) < len(instances):
        print("ВЕРДИКТ: ЖИВОЙ ПОТОК РАБОТАЕТ ✓")
        print("  Сквозь: NLU(LLM) воспринял → реальность срезала галлюцинации → агент УЗНАЛ db2")
        print("  как категорию db1 по ЗАЗЕМЛЁННОЙ структуре → положил в DAG. Всё под капотом.")
        print(f"  Реальность переспорила LLM {pruned_total} раз; категорий {len(categories)} ≪ инстансов {len(instances)}.")
    else:
        print(f"ВЕРДИКТ: ЧАСТИЧНО — db2_recognized={db2_recognized}, cat={len(categories)}, inst={len(instances)}")
        print("  (возможно LLM недо/переутвердил — это видно в трейсе выше; честно репортим что вышло)")


if __name__ == "__main__":
    main()
