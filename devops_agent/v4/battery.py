"""
battery.py — v4 acceptance: батарея situation-тестов (item 3, консолидация).

Доказывает контент-нейтральность РОСТА из фрагментов через packer на СТРУКТУРНО РАЗНЫХ
вымышленных доменах (не переименованные копии): 1-ресурс, settings-only, транзитивная
цепь, ромб-DAG, вырожденный (без ресурсов). Плюс:
  • персистентность СКВОЗЬ packer (save → load в «новом процессе» = лечение амнезии);
  • идемпотентность и отказ малформа (граф цел);
  • анти-хардкод гвард: в чистых модулях (graph/contract) нет devops-лексики.

Анти-хардкод по сути: ВСЕ домены прогоняются ТЕМ ЖЕ кодом, ноль правок логики. Если домен
потребовал бы правки — печатаем «не обобщается». Судья проекции — FD.
"""

import os
import sys

from devops_agent.v4.graph import KnowledgeGraph
from devops_agent.v4.packer import Packer
from devops_agent.v4.projection import kg_to_pddl, solve

_OUT = os.path.join(os.path.dirname(__file__), "_generated", "battery")


# --- генератор фрагментов одной сущности (контент-нейтрально) ---------------

def _entity(ent, cls, resources=(), settings=(), depends=()):
    """Фрагменты для сущности: ресурсы/настройки + per-resource set_* + provision_<ent>."""
    frags = [{"op": "entity", "id": ent, "cls": cls}]
    reqs = []
    for r in resources:
        frags += [{"op": "resource", "id": r, "of": ent},
                  {"op": "intervention", "id": f"set_{r}", "establishes": [r]}]
        reqs.append(r)
    for s in settings:
        frags += [{"op": "setting", "id": s, "of": ent},
                  {"op": "intervention", "id": f"set_{s}", "establishes": [s]}]
        reqs.append(s)
    goal_id = f"{ent}_ready"
    frags += [{"op": "status", "id": goal_id, "of": ent, "goal": True},
              {"op": "intervention", "id": f"provision_{ent}", "establishes": [goal_id], "requires": reqs}]
    for d in depends:
        frags.append({"op": "dependency", "of": ent, "on": d})
    return frags


# --- структурно разнообразные вымышленные домены ----------------------------
# (name, fragments, goal_entity, expected_plan_len, [(before, after) ...])

def _domains():
    cache = _entity("cache", "cache", resources=["eviction_budget"])

    cdn = _entity("edge", "cdn", settings=["region", "tls_mode"])          # settings-only, 0 ресурсов

    # транзитивная цепь: ingest → transform → load (load первым)
    pipeline = (_entity("load", "loader", resources=["load_disk"])
                + _entity("transform", "worker", resources=["xf_cpu"], depends=["load"])
                + _entity("ingest", "api", resources=["in_buffer"], depends=["transform"]))

    # ромб-DAG: frontend → {api, worker} → db (db общий, провижится один раз и первым)
    web = (_entity("db", "database", resources=["db_disk"])
           + _entity("api", "service", resources=["api_threads"], depends=["db"])
           + _entity("worker", "service", resources=["worker_mem"], depends=["db"])
           + _entity("frontend", "spa", settings=["fe_theme"], depends=["api", "worker"]))

    bare = _entity("beacon", "daemon")                                     # ни ресурсов, ни настроек

    return [
        ("cache (1 resource)", cache, "cache", 2, []),
        ("cdn (settings-only)", cdn, "edge", 3, []),
        ("pipeline (transitive chain)", pipeline, "ingest", 6,
         [("provision_load", "provision_transform"), ("provision_transform", "provision_ingest")]),
        ("web_stack (diamond DAG)", web, "frontend", 8,
         [("provision_db", "provision_api"), ("provision_db", "provision_worker"),
          ("provision_api", "provision_frontend"), ("provision_worker", "provision_frontend")]),
        ("bare (no resources)", bare, "beacon", 1, []),
    ]


def _order_violations(plan, constraints):
    j = [str(x) for x in (plan or [])]
    def idx(name):
        return next((i for i, s in enumerate(j) if name in s), -1)
    bad = []
    for before, after in constraints:
        ib, ia = idx(before), idx(after)
        if ib < 0 or ia < 0 or ib >= ia:
            bad.append(f"{before}({ib}) !< {after}({ia})")
    return bad


def main() -> None:
    print("=== v4 БАТАРЕЯ (item 3) — контент-нейтральный рост из фрагментов ===\n")
    errors = []

    # 1. структурно разные домены: фрагмент → packer → проекция → FD
    for name, frags, goal, exp_len, order in _domains():
        p = Packer()
        r = p.pack(frags)
        if not r.ok:
            errors.append(f"{name}: packer отклонил валидный домен: {r.violations}")
            continue
        dpath, ppath = kg_to_pddl(p.g, goal, out_dir=os.path.join(_OUT, goal))
        plan = solve(dpath, ppath)
        ord_bad = _order_violations(plan, order)
        ok = plan is not None and len(plan) == exp_len and not ord_bad
        print(f"  [{'OK' if ok else 'FAIL'}] {name:32s} len={len(plan or [])}/{exp_len} plan={[str(x) for x in (plan or [])]}")
        if plan is None:
            errors.append(f"{name}: FD не решил")
        elif len(plan) != exp_len:
            errors.append(f"{name}: длина {len(plan)} != {exp_len}")
        if ord_bad:
            errors.append(f"{name}: порядок зависимостей нарушен: {ord_bad}")

    # 2. персистентность СКВОЗЬ packer (save → load «в новом процессе» = лечение амнезии)
    print("\n-- персистентность сквозь packer --")
    p = Packer()
    p.pack(_domains()[3][1])                       # вырастили web_stack (ромб)
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "persisted.json")
    p.g.save(path)
    reloaded = KnowledgeGraph.load(path)           # свежий граф, как новый процесс
    if reloaded.to_json() != p.g.to_json():
        errors.append("persistence: load != save")
    d2, p2 = kg_to_pddl(reloaded, "frontend", out_dir=os.path.join(_OUT, "reloaded"))
    plan2 = solve(d2, p2)
    ord_bad2 = _order_violations(plan2, [("provision_db", "provision_frontend")])
    print(f"  reload → plan(frontend): len={len(plan2 or [])}; dep-order ok={not ord_bad2}")
    if plan2 is None or len(plan2) != 8 or ord_bad2:
        errors.append(f"persistence: выживший граф не решается/не секвенирует: {plan2}")
    else:
        print("  ✓ знание пережило save/load и по-прежнему планируется (амнезии нет)")

    # 3. идемпотентность + отказ малформа (граф цел)
    print("\n-- идемпотентность / отказ малформа --")
    p = Packer()
    p.pack(_domains()[0][1])
    before = p.g.to_json()
    p.pack(_domains()[0][1])                       # повтор
    if p.g.to_json() != before:
        errors.append("идемпотентность: повторный pack изменил граф")
    rbad = p.pack([{"op": "intervention", "id": "ghost"}])   # без establishes
    if rbad.ok or p.g.to_json() != before:
        errors.append("малформ: закоммичен или граф изменён")
    print(f"  идемпотентность ✓; малформ отклонён ✓ ({rbad.violations})")

    # 4. анти-хардкод гвард: чистые модули без devops-лексики
    print("\n-- анти-хардкод гвард --")
    _here = os.path.dirname(__file__)
    devops_terms = ["memory", "mem_", "config", "docker", "oom", "bucket", "workload"]
    for mod in ("graph.py", "contract.py"):
        src = open(os.path.join(_here, mod)).read().lower()
        hit = [t for t in devops_terms if t in src]
        if hit:
            errors.append(f"анти-хардкод: {mod} содержит devops-лексику {hit}")
    print(f"  graph.py / contract.py — devops-лексики нет ✓")

    # --- вердикт ---
    print("\n" + "=" * 60)
    if errors:
        print("ВЕРДИКТ: НЕ ОБОБЩАЕТСЯ ✗")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("ВЕРДИКТ: ОБОБЩАЕТСЯ ✓ — 5 структурно разных доменов выросли из фрагментов тем же кодом,")
    print("  FD решает и секвенирует DAG-зависимости; знание переживает save/load; ноль devops-хардкода.")


if __name__ == "__main__":
    main()
