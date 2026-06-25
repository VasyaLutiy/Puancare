"""
run.py — демо v6.
  (по умолчанию) ОБЪЁМ: поток фрагментов → распознаватель → под капотом видно, как
     агент формирует КОНЕЧНОЕ число категорий из растущего потока инстансов (обобщение).
     + валидатор без PDDL (контракт; ацикличность DAG — по построению: deps только на ранних).
  --live: ЖИВОЙ NLU (Azure): сырьё → JSON-фрагмент (мембрана восприятия реальна).
"""

import sys

from devops_agent.v6 import world
from devops_agent.v6.hood import Hood
from devops_agent.v6.recognizer import KB, Recognizer


def volume() -> None:
    print("=== v6 ОБЪЁМ: распознавание+размещение, всё под капотом ===\n")
    kb = KB()
    rec = Recognizer(kb)
    hood = Hood()
    for frag in world.stream(n=30, new_at=18):
        hood.record(rec.observe(frag), len(kb.categories))
    hood.summary(kb)

    print("\n  -- валидатор без PDDL (контракт + DAG-консистентность) --")
    bads = [
        {"id": "bad1", "label": "x", "resources": ["weird_kind"], "settings": [], "status": "ready", "deps": []},
        {"id": "bad2", "label": "x", "resources": [], "settings": [], "status": "ready", "deps": ["ghost"]},
    ]
    rejected_all = True
    for b in bads:
        d = rec.observe(b)
        ok = d.get("rejected")
        rejected_all = rejected_all and bool(ok)
        print(f"    {b['id']}: {'ОТКЛОНЁН ' + str(d['violations']) if ok else 'ПРОПУЩЕН(!)'}")
    print("    (циклы в DAG невозможны по построению: deps ссылаются только на УЖЕ размещённые)")

    n_cat, n_inst = len(kb.categories), len(kb.instances)
    done = hood.recognized + hood.placed
    rate = hood.recognized / done if done else 0
    print("\n  граф знаний (срез):")
    print("   ", kb.to_json()[:500].replace("\n", "\n    "))

    ok = n_cat < n_inst and n_cat <= 6 and rate >= 0.6 and rejected_all
    print("\n" + "=" * 60)
    if ok:
        print(f"ВЕРДИКТ v6: РАБОТАЕТ ✓ — из {n_inst} инстансов агент сформировал {n_cat} КАТЕГОРИЙ")
        print(f"  (плато ≪ поток), распознал {rate:.0%} структурно (не по строке) — это обобщение (E1-фикс).")
        print("  Всё видно под капотом: трейс решений + кривая категорий + граф. Валидатор без PDDL держит форму.")
    else:
        print(f"ВЕРДИКТ v6: НЕ ОК ✗ — cat={n_cat} inst={n_inst} rate={rate:.0%} rejected_all={rejected_all}")


def live() -> None:
    from devops_agent.v6.nlu import LiveNLU
    print("=== v6 --live: NLU (Azure) сырьё → JSON-фрагмент ===\n")
    nlu = LiveNLU()
    raws = [
        {"image": "postgres:15", "listens": 5432, "env": ["PGDATA", "auth=required"], "disk": "persistent"},
        {"image": "redis:7", "listens": 6379, "maxmemory_policy": "lru", "maxmemory": "tunable"},
        {"image": "nginx", "listens": 80, "upstream": "api", "tls": "on|off"},
    ]
    for raw in raws:
        print(f"  raw: {raw}\n   → фрагмент: {nlu.perceive(raw)}\n")
    print(f"  n_calls={nlu.n_calls} tokens={nlu.total_tokens}")
    print("  (LLM воспринял КАЖДЫЙ сигнал в изоляции → JSON; распознавание — НЕ здесь, оно в агенте.)")


if __name__ == "__main__":
    live() if "--live" in sys.argv else volume()
