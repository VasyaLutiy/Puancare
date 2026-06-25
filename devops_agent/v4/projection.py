"""
projection.py — KG → PDDL (контент-нейтрально) + solve (ground-truth через FD).

PDDL-слой ground-truth: упаковка корректна ⟺ граф проецируется в PDDL, который FD парсит
и решает. Обобщение v3 domain_gen на произвольный типизированный граф ОДНОЙ сущности
(мульти-сущность + Dependency — кейстоун-2, нужны ground-actions/:constants). Имя предиката
выводится из узла (детерминизм). Движок solve переиспользуем из v3.
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


def kg_to_pddl(g, goal_entity: str, out_dir: str = _OUT):
    """Срез графа вокруг goal_entity → (domain.pddl, problem.pddl). Возвращает пути."""
    if goal_entity not in g.nodes or g.nodes[goal_entity].mtype != MetaType.ENTITY:
        raise ValueError(f"{goal_entity!r} не Entity")
    os.makedirs(out_dir, exist_ok=True)

    goals = [e.dst for e in g.out(goal_entity, Rel.HAS_GOAL)]
    if len(goals) != 1:
        raise ValueError(f"{goal_entity!r}: ожидался ровно один HAS_GOAL, найдено {len(goals)}")
    goal_pred = _pred_of(g, goals[0])
    res_preds = [_pred_of(g, e.dst) for e in g.out(goal_entity, Rel.HAS)]

    ivs = sorted(g.nodes_of(MetaType.INTERVENTION), key=lambda n: n.id)

    # все предикаты, что встретятся (цель + ресурсы сущности + всё, что трогают интервенции)
    used = set([goal_pred]) | set(res_preds)
    for iv in ivs:
        for e in g.out(iv.id, Rel.ESTABLISHES):
            used.add(_pred_of(g, e.dst))
        for e in g.out(iv.id, Rel.REQUIRES):
            used.add(_pred_of(g, e.dst))
    all_preds = sorted(used)

    lines = [
        "(define (domain kg)",
        "  (:requirements :strips :typing :negative-preconditions)",
        "  (:types entity - object)",
        "",
        "  (:predicates",
    ]
    for i, p in enumerate(all_preds):
        lines.append(f"    ({p} ?e - entity){')' if i == len(all_preds) - 1 else ''}")

    for iv in ivs:
        est = [_pred_of(g, e.dst) for e in g.out(iv.id, Rel.ESTABLISHES)]
        req = sorted(_pred_of(g, e.dst) for e in g.out(iv.id, Rel.REQUIRES))
        terms = [f"(not ({p} ?e))" for p in est] + [f"({p} ?e)" for p in req]
        pre = "(and " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        eff = "(and " + " ".join(f"({p} ?e)" for p in est) + ")" if len(est) > 1 else f"({est[0]} ?e)"
        lines += ["", f"  (:action {iv.id}", "    :parameters (?e - entity)",
                  f"    :precondition {pre}", f"    :effect {eff})"]
    lines += [")", ""]

    dpath = os.path.join(out_dir, "domain.pddl")
    with open(dpath, "w") as f:
        f.write("\n".join(lines))
    ppath = os.path.join(out_dir, "problem.pddl")
    with open(ppath, "w") as f:
        f.write("\n".join([
            "(define (problem kg-p)", "  (:domain kg)", "",
            f"  (:objects {goal_entity} - entity)", "", "  (:init)", "",
            f"  (:goal ({goal_pred} {goal_entity}))", ")", "",
        ]))
    return dpath, ppath


# ---------------------------------------------------------------------------
# Keystone-1 selftest: ВЫМЫШЛЕННЫЕ не-devops графы → PDDL → FD. Контент-нейтральность.
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
    g.add_node("set_conn_pool", MetaType.INTERVENTION)
    g.add_edge("set_conn_pool", Rel.ESTABLISHES, "conn_pool")
    g.add_node("set_cache_size", MetaType.INTERVENTION)
    g.add_edge("set_cache_size", Rel.ESTABLISHES, "cache_size")
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
    g.add_node("set_max_size", MetaType.INTERVENTION)
    g.add_edge("set_max_size", Rel.ESTABLISHES, "max_size")
    g.add_node("start", MetaType.INTERVENTION)
    g.add_edge("start", Rel.ESTABLISHES, "serving")
    g.add_edge("start", Rel.REQUIRES, "max_size")
    return g


def _selftest() -> None:
    import sys

    from devops_agent.v4.contract import validate
    from devops_agent.v4.graph import KnowledgeGraph

    print("=== v4 keystone-1: контент-нейтральный граф → PDDL → FD ===\n")
    errors = []

    cases = [
        ("database", _build_database(), "orders_db", 3),
        ("queue", _build_queue(), "jobs_q", 2),
    ]
    for name, g, goal_e, expect_len in cases:
        viol = validate(g)
        if viol:
            errors.append(f"{name}: контракт нашёл нарушения в ВАЛИДНОМ графе: {viol}")
            continue
        dpath, ppath = kg_to_pddl(g, goal_e, out_dir=os.path.join(_OUT, name))
        plan = solve(dpath, ppath)
        print(f"--- {name} ({goal_e}) ---")
        print(open(dpath).read())
        print(f"plan: {plan}\n")
        if plan is None:
            errors.append(f"{name}: FD не решил сгенерённый домен")
        elif len(plan) != expect_len:
            errors.append(f"{name}: ожидали {expect_len} шага, got {plan}")
        if ":metric" in open(dpath).read() or ":functions" in open(dpath).read():
            errors.append(f"{name}: ГВАРД #8 — появился :metric/:functions")

    # malformed → контракт ловит (висячий resource, intervention без effect, нелегальное ребро)
    bad = KnowledgeGraph()
    bad.add_node("svc", MetaType.ENTITY)
    bad.add_node("floating", MetaType.RESOURCE)        # ничей
    bad.add_node("noop", MetaType.INTERVENTION)        # без establishes
    bad.add_edge("svc", Rel.HAS_GOAL, "floating")      # нелегально: HAS_GOAL → resource
    mviol = validate(bad)
    print(f"malformed → violations ({len(mviol)}): {mviol}")
    if len(mviol) < 3:
        errors.append(f"контракт пропустил малформ (ждали ≥3): {mviol}")

    # персистентность round-trip
    g = _build_database()
    j1 = g.to_json()
    if KnowledgeGraph.from_json(j1).to_json() != j1:
        errors.append("persistence round-trip: JSON не совпал после load")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("v4 keystone-1 OK — контент-нейтральный граф пакуется в валидный PDDL и решается FD; "
          "контракт ловит малформ; persistence ок.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v4.projection --selftest")
