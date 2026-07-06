"""
contract.py — ISA-контракт: что есть КОРРЕКТНО упакованное знание (структурный слой ground-truth).

Второй слой — проекция в PDDL + FD (см. projection.py). validate(g) → список нарушений
(пусто = структурно корректно). Контент-нейтрально: никакой devops-специфики, только правила
6 мета-типов и легальных рёбер.
"""

from devops_agent.v4.graph import MetaType, Rel

# Легальные рёбра: (тип источника, отношение, тип цели)
_LEGAL = {
    (MetaType.ENTITY, Rel.HAS, MetaType.RESOURCE),
    (MetaType.ENTITY, Rel.HAS, MetaType.SETTING),
    (MetaType.ENTITY, Rel.HAS_GOAL, MetaType.STATUS),
    (MetaType.ENTITY, Rel.DEPENDS_ON, MetaType.ENTITY),
    (MetaType.INTERVENTION, Rel.ESTABLISHES, MetaType.RESOURCE),
    (MetaType.INTERVENTION, Rel.ESTABLISHES, MetaType.SETTING),
    (MetaType.INTERVENTION, Rel.ESTABLISHES, MetaType.STATUS),
    (MetaType.INTERVENTION, Rel.REQUIRES, MetaType.RESOURCE),
    (MetaType.INTERVENTION, Rel.REQUIRES, MetaType.SETTING),
    (MetaType.INTERVENTION, Rel.REQUIRES, MetaType.STATUS),
}


def validate(g) -> list:
    """Структурные нарушения упаковки. Пусто = корректно. Контент-нейтрально."""
    v = []
    # 1. рёбра: оба конца существуют + сигнатура ребра легальна
    for e in g.edges:
        if e.src not in g.nodes:
            v.append(f"ребро на несуществующий src: {e}")
            continue
        if e.dst not in g.nodes:
            v.append(f"ребро на несуществующий dst: {e}")
            continue
        sig = (g.nodes[e.src].mtype, e.rel, g.nodes[e.dst].mtype)
        if sig not in _LEGAL:
            v.append(f"нелегальное ребро {sig[0].value}-{sig[1].value}->{sig[2].value}: {e}")
    # 2. узловые инварианты
    for n in g.nodes.values():
        if n.mtype in (MetaType.RESOURCE, MetaType.SETTING):
            if not g.has_incoming(n.id, Rel.HAS):
                v.append(f"висячий {n.mtype.value} {n.id!r}: ни один Entity не HAS его")
        if n.mtype == MetaType.INTERVENTION:
            if not g.out(n.id, Rel.ESTABLISHES):
                v.append(f"intervention {n.id!r} ничего не establishes (нет эффекта)")
    return v
