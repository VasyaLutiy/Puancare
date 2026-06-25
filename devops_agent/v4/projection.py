"""
projection.py — KG → PDDL (контент-нейтрально, мульти-сущность) + solve (ground-truth через FD).

PDDL-слой ground-truth: упаковка корректна ⟺ граф проецируется в PDDL, который FD парсит и решает.
Форма: сущности — :constants домена, по entity — ground-actions (гетерогенные ресурсы единым
lifted-action невыразимы). Dependency (Entity DEPENDS_ON Entity) → предусловие: интервенция,
устанавливающая goal-статус дочерней, требует goal-статус родителя ⇒ FD ВЫНУЖДЕН секвенировать.
Имя предиката выводится из узла (детерминизм). Движок solve переиспользуем из v3.
"""

import os

from devops_agent.v3.domain_gen import solve  # переиспользуем FD-движок v3
from devops_agent.v4.graph import MetaType, Rel

_OUT = os.path.join(os.path.dirname(__file__), "_generated")


def _pred_of(g, node_id: str) -> str:
    """resource/setting → '<id>_ok'; status → '<id>'."""
    mt = g.nodes[node_id].mtype
    if mt in (MetaType.RESOURCE, MetaType.SETTING):
        return f"{node_id}_ok"
    if mt == MetaType.STATUS:
        return node_id
    raise ValueError(f"узел {node_id!r} типа {mt.value} не даёт предиката")


def _owner_map(g) -> dict:
    """node_id → entity_id (кто HAS/HAS_GOAL этот узел)."""
    owner = {}
    for e in g.edges:
        if e.rel in (Rel.HAS, Rel.HAS_GOAL):
            owner[e.dst] = e.src
    return owner


def _involved(g, goal_entity: str) -> list:
    """goal_entity + транзитивное замыкание DEPENDS_ON."""
    seen, stack = [], [goal_entity]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.append(x)
        for e in g.out(x, Rel.DEPENDS_ON):
            stack.append(e.dst)
    return seen


def _goal_pred_of_entity(g, ent: str) -> str:
    gs = g.out(ent, Rel.HAS_GOAL)
    if len(gs) != 1:
        raise ValueError(f"{ent!r}: ожидался ровно один HAS_GOAL, найдено {len(gs)}")
    return _pred_of(g, gs[0].dst)


def kg_to_pddl(g, goal_entity: str, out_dir: str = _OUT):
    """Срез графа (goal_entity + его зависимости) → (domain.pddl, problem.pddl). Возвращает пути."""
    if goal_entity not in g.nodes or g.nodes[goal_entity].mtype != MetaType.ENTITY:
        raise ValueError(f"{goal_entity!r} не Entity")
    os.makedirs(out_dir, exist_ok=True)

    owner = _owner_map(g)
    involved = _involved(g, goal_entity)
    involved_set = set(involved)

    # предикаты: статусы + ресурсы/настройки всех вовлечённых сущностей
    preds = set()
    for ent in involved:
        for e in g.out(ent, Rel.HAS):
            preds.add(_pred_of(g, e.dst))
        for e in g.out(ent, Rel.HAS_GOAL):
            preds.add(_pred_of(g, e.dst))
    all_preds = sorted(preds)

    # интервенции, чей owner (по establishes) — вовлечённая сущность
    ivs = []
    for n in g.nodes_of(MetaType.INTERVENTION):
        est_nodes = [e.dst for e in g.out(n.id, Rel.ESTABLISHES)]
        if est_nodes and owner.get(est_nodes[0]) in involved_set:
            ivs.append(n)
    ivs.sort(key=lambda n: n.id)

    lines = [
        "(define (domain kg)",
        "  (:requirements :strips :typing :negative-preconditions)",
        "  (:types entity - object)",
        f"  (:constants {' '.join(sorted(involved))} - entity)",
        "",
        "  (:predicates",
    ]
    for i, p in enumerate(all_preds):
        lines.append(f"    ({p} ?e - entity){')' if i == len(all_preds) - 1 else ''}")

    for iv in ivs:
        est_nodes = [e.dst for e in g.out(iv.id, Rel.ESTABLISHES)]
        ent = owner[est_nodes[0]]                       # сущность-владелец интервенции
        est = [_pred_of(g, nid) for nid in est_nodes]
        req = sorted(_pred_of(g, e.dst) for e in g.out(iv.id, Rel.REQUIRES))
        terms = [f"(not ({p} {ent}))" for p in est] + [f"({p} {ent})" for p in req]
        # Dependency: если интервенция устанавливает goal-статус ent — требуем goal-статусы родителей
        if _goal_pred_of_entity(g, ent) in est:
            for de in g.out(ent, Rel.DEPENDS_ON):
                terms.append(f"({_goal_pred_of_entity(g, de.dst)} {de.dst})")
        pre = "(and " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        eff = "(and " + " ".join(f"({p} {ent})" for p in est) + ")" if len(est) > 1 else f"({est[0]} {ent})"
        lines += ["", f"  (:action {iv.id}", "    :parameters ()",
                  f"    :precondition {pre}", f"    :effect {eff})"]
    lines += [")", ""]

    dpath = os.path.join(out_dir, "domain.pddl")
    with open(dpath, "w") as f:
        f.write("\n".join(lines))

    goal_pred = _goal_pred_of_entity(g, goal_entity)
    ppath = os.path.join(out_dir, "problem.pddl")
    with open(ppath, "w") as f:
        f.write("\n".join([
            "(define (problem kg-p)", "  (:domain kg)", "",
            "  (:init)", "",
            f"  (:goal ({goal_pred} {goal_entity}))", ")", "",
        ]))
    return dpath, ppath


# ---------------------------------------------------------------------------
# Keystone selftest: вымышленные не-devops графы → PDDL → FD.
#   -1: одиночные сущности (database, queue) — контент-нейтральность.
#   -2: мульти-сущность + Dependency (api_gw → orders_db) — планировщик секвенирует.
# ---------------------------------------------------------------------------

def _build_database():
    from devops_agent.v4.graph import KnowledgeGraph
    g = KnowledgeGraph()
    g.add_node("orders_db", MetaType.ENTITY, cls="database")
    g.add_node("conn_pool", MetaType.RESOURCE, kind="ordered_monotone")
    g.add_node("cache_size", MetaType.RESOURCE, kind="ordered_monotone")
    g.add_node("ready", MetaType.STATUS)
    g.add_edge("orders_db", Rel.HAS, "conn_pool")
    g.add_edge("orders_db", Rel.HAS, "cache_size")
    g.add_edge("orders_db", Rel.HAS_GOAL, "ready")
    g.add_node("set_conn_pool", MetaType.INTERVENTION); g.add_edge("set_conn_pool", Rel.ESTABLISHES, "conn_pool")
    g.add_node("set_cache_size", MetaType.INTERVENTION); g.add_edge("set_cache_size", Rel.ESTABLISHES, "cache_size")
    g.add_node("provision", MetaType.INTERVENTION)
    g.add_edge("provision", Rel.ESTABLISHES, "ready")
    g.add_edge("provision", Rel.REQUIRES, "conn_pool")
    g.add_edge("provision", Rel.REQUIRES, "cache_size")
    return g


def _build_queue():
    from devops_agent.v4.graph import KnowledgeGraph
    g = KnowledgeGraph()
    g.add_node("jobs_q", MetaType.ENTITY, cls="queue")
    g.add_node("max_size", MetaType.SETTING, options=["small", "large"])
    g.add_node("serving", MetaType.STATUS)
    g.add_edge("jobs_q", Rel.HAS, "max_size")
    g.add_edge("jobs_q", Rel.HAS_GOAL, "serving")
    g.add_node("set_max_size", MetaType.INTERVENTION); g.add_edge("set_max_size", Rel.ESTABLISHES, "max_size")
    g.add_node("start", MetaType.INTERVENTION)
    g.add_edge("start", Rel.ESTABLISHES, "serving")
    g.add_edge("start", Rel.REQUIRES, "max_size")
    return g


def _build_microservice():
    """api_gw ЗАВИСИТ от orders_db → provision_api требует db_ready ⇒ FD секвенирует db перед api."""
    from devops_agent.v4.graph import KnowledgeGraph
    g = KnowledgeGraph()
    # database
    g.add_node("orders_db", MetaType.ENTITY, cls="database")
    g.add_node("conn_pool", MetaType.RESOURCE)
    g.add_node("db_ready", MetaType.STATUS)
    g.add_edge("orders_db", Rel.HAS, "conn_pool")
    g.add_edge("orders_db", Rel.HAS_GOAL, "db_ready")
    g.add_node("set_conn_pool", MetaType.INTERVENTION); g.add_edge("set_conn_pool", Rel.ESTABLISHES, "conn_pool")
    g.add_node("provision_db", MetaType.INTERVENTION)
    g.add_edge("provision_db", Rel.ESTABLISHES, "db_ready")
    g.add_edge("provision_db", Rel.REQUIRES, "conn_pool")
    # gateway, зависит от db
    g.add_node("api_gw", MetaType.ENTITY, cls="gateway")
    g.add_node("routes", MetaType.RESOURCE)
    g.add_node("api_ready", MetaType.STATUS)
    g.add_edge("api_gw", Rel.HAS, "routes")
    g.add_edge("api_gw", Rel.HAS_GOAL, "api_ready")
    g.add_node("set_routes", MetaType.INTERVENTION); g.add_edge("set_routes", Rel.ESTABLISHES, "routes")
    g.add_node("provision_api", MetaType.INTERVENTION)
    g.add_edge("provision_api", Rel.ESTABLISHES, "api_ready")
    g.add_edge("provision_api", Rel.REQUIRES, "routes")
    g.add_edge("api_gw", Rel.DEPENDS_ON, "orders_db")
    return g


def _selftest() -> None:
    import sys

    from devops_agent.v4.contract import validate
    from devops_agent.v4.graph import KnowledgeGraph

    print("=== v4 keystone-1/2: контент-нейтральный граф → PDDL → FD ===\n")
    errors = []

    # keystone-1: одиночные сущности
    for name, g, goal_e, expect_len in [("database", _build_database(), "orders_db", 3),
                                        ("queue", _build_queue(), "jobs_q", 2)]:
        viol = validate(g)
        if viol:
            errors.append(f"{name}: контракт нашёл нарушения в ВАЛИДНОМ графе: {viol}")
            continue
        dpath, ppath = kg_to_pddl(g, goal_e, out_dir=os.path.join(_OUT, name))
        plan = solve(dpath, ppath)
        print(f"--- {name} ({goal_e}) → {plan}")
        if plan is None:
            errors.append(f"{name}: FD не решил")
        elif len(plan) != expect_len:
            errors.append(f"{name}: ожидали {expect_len} шага, got {plan}")
        if ":metric" in open(dpath).read() or ":functions" in open(dpath).read():
            errors.append(f"{name}: ГВАРД #8 — :metric/:functions")

    # keystone-2: мульти-сущность + Dependency → секвенирование
    g = _build_microservice()
    viol = validate(g)
    if viol:
        errors.append(f"microservice: контракт нашёл нарушения: {viol}")
    else:
        dpath, ppath = kg_to_pddl(g, "api_gw", out_dir=os.path.join(_OUT, "microservice"))
        plan = solve(dpath, ppath)
        print("\n--- microservice (api_gw зависит от orders_db) ---")
        print(open(dpath).read())
        print(f"plan: {plan}\n")
        if plan is None:
            errors.append("microservice: FD не решил")
        else:
            j = [str(x) for x in plan]
            if len(plan) != 4:
                errors.append(f"microservice: ожидали 4 шага, got {plan}")
            try:
                i_db = next(i for i, s in enumerate(j) if "provision_db" in s)
                i_api = next(i for i, s in enumerate(j) if "provision_api" in s)
                if not (i_db < i_api):
                    errors.append(f"microservice: порядок нарушен — provision_db не раньше provision_api: {plan}")
                else:
                    print(f"  ✓ секвенс: provision_db[{i_db}] раньше provision_api[{i_api}] (Dependency сработала)")
            except StopIteration:
                errors.append(f"microservice: в плане нет provision_db/provision_api: {plan}")

    # malformed → контракт ловит
    bad = KnowledgeGraph()
    bad.add_node("svc", MetaType.ENTITY)
    bad.add_node("floating", MetaType.RESOURCE)
    bad.add_node("noop", MetaType.INTERVENTION)
    bad.add_edge("svc", Rel.HAS_GOAL, "floating")
    mviol = validate(bad)
    print(f"malformed → violations ({len(mviol)}): {mviol}")
    if len(mviol) < 3:
        errors.append(f"контракт пропустил малформ (ждали ≥3): {mviol}")

    # persistence round-trip
    g = _build_microservice()
    j1 = g.to_json()
    if KnowledgeGraph.from_json(j1).to_json() != j1:
        errors.append("persistence round-trip: JSON не совпал")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("v4 keystone-1/2 OK — графы (вкл. мульти-сущность с Dependency) пакуются в валидный PDDL, "
          "FD решает и СЕКВЕНИРУЕТ по зависимостям; контракт ловит малформ; persistence ок.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v4.projection --selftest")
