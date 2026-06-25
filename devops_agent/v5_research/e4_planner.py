"""
E4 — ПЛАНИРОВЩИК = ТОПОСОРТ? Делает ли FD реальную работу, или PDDL — оверхед?

Атака: «планировщик секвенирует DAG» = топологическая сортировка, O(V+E), не поиск. FD оправдан
лишь когда есть ВЫБОР: взаимоисключение, разделяемый бюджет, оптимизация. Показываем:
  (1) на наших доменах план — вынужденно-линейный (топосорт, нет ветвления);
  (2) контеншн НЕВЫРАЗИМ в 6 типах: проекция даёт только ПОЛОЖИТЕЛЬНЫЕ эффекты (нет delete →
      нет mutex) и без чисел (#8 убрал) → нет бюджета/ёмкости. ⇒ FD никогда не ищет.
"""

import sys

from devops_agent.v4.graph import KnowledgeGraph, MetaType, Rel
from devops_agent.v4.projection import kg_to_pddl, solve


def _build_dep_chain():
    """A зависит от B; у каждого свой ресурс. Классический случай 'планировщик упорядочивает'."""
    g = KnowledgeGraph()
    for e, r, st in [("A", "rA", "A_ready"), ("B", "rB", "B_ready")]:
        g.add_node(e, MetaType.ENTITY)
        g.add_node(r, MetaType.RESOURCE)
        g.add_node(st, MetaType.STATUS)
        g.add_edge(e, Rel.HAS, r)
        g.add_edge(e, Rel.HAS_GOAL, st)
        g.add_node(f"set_{r}", MetaType.INTERVENTION); g.add_edge(f"set_{r}", Rel.ESTABLISHES, r)
        g.add_node(f"provision_{e}", MetaType.INTERVENTION)
        g.add_edge(f"provision_{e}", Rel.ESTABLISHES, st)
        g.add_edge(f"provision_{e}", Rel.REQUIRES, r)
    g.add_edge("A", Rel.DEPENDS_ON, "B")
    return g


def main() -> None:
    print("=== E4: планировщик = топосорт? (нужен ли PDDL вообще) ===\n")
    g = _build_dep_chain()
    dpath, ppath = kg_to_pddl(g, "A")
    plan = solve(dpath, ppath)
    dom = open(dpath).read()
    print(f"план (A зависит от B): {plan}\n")

    # (1) план вынужденно-линейный = топосорт: все действия по разу, порядок задан зависимостями
    n_actions = dom.count("(:action ")
    linear = plan is not None and len(plan) == n_actions
    j = [str(x) for x in (plan or [])]
    dep_ok = (next((i for i, s in enumerate(j) if "provision_B" in s), -1)
              < next((i for i, s in enumerate(j) if "provision_A" in s), 99))

    # (2) контеншн невыразим: в эффектах нет delete (нет mutex); нет числовой ёмкости
    del_eff = sum(1 for ln in dom.splitlines()
                  if ln.strip().startswith(":effect") and "(not (" in ln)
    has_numbers = ":functions" in dom or ":metric" in dom    # #8 убрал числа → ожидаем False

    print(f"(1) действий={n_actions}, длина плана={len(plan or [])} → линейный/вынужденный: {linear}; "
          f"provision_B раньше provision_A: {dep_ok}  ⇒ это ТОПОСОРТ (ветвления нет)")
    print(f"(2) delete-эффектов (=возможность mutex): {del_eff};  числа/ёмкость в домене: {has_numbers}")
    print("    ⇒ взаимоисключение и разделяемый бюджет НЕВЫРАЗИМЫ (только +эффекты, чистый STRIPS-булев)")
    print("=" * 62)

    toposort_only = linear and dep_ok and del_eff == 0 and not has_numbers
    if toposort_only:
        print("ВЕРДИКТ E4: ПАДЁТ ✗ — FD делает только вынужденное упорядочивание (топосорт).")
        print("  Контеншн/взаимоисключение/оптимизация НЕвыразимы в 6 типах (нет delete-эффектов = нет")
        print("  mutex; #8 убрал числа = нет ёмкости). ⇒ настоящий поиск недостижим, PDDL/FD — оверхед")
        print("  над 5-строчным топосортом. Чтобы оправдать PDDL, ISA нужны ёмкость/mutex/числа.")
    else:
        print("ВЕРДИКТ E4: ВЫЖИЛ ✓ — FD делает работу за пределами топосорта (неожиданно — проверить).")


if __name__ == "__main__":
    main()
