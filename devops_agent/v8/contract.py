"""
contract.py — v8 ЗАКОН: что есть корректно упакованное знание. ЕДИНЫЙ ИСТОЧНИК.

Урок 6 июля: закон жил в валидаторе (v4/contract), а промпт компилятора — в другом
файле со своей ослабленной схемой; они разъехались, и 74% книжных мет вошли в KB
беззаконными. Поэтому здесь закон записан ДАННЫМИ (таблицы ниже), а из них
генерятся ОБА потребителя: validate() для машины и LAW_PROMPT для LLM-компилятора.
Разъехаться им больше не из чего.

Слои закона:
  1. лексика: легальные рёбра (LEGAL_EDGES) + виды (KINDS);
  2. узловые инварианты: владение, эффект, цель;
  3. логический слой: mutex/atmost — через constraints.validate_constraints.

Контракт валидирует УПАКОВКУ, не истинность: истину даёт только проба (слои A–E).
"""

from __future__ import annotations

from devops_agent.v8.constraints import validate_constraints
from devops_agent.v8.graph import MetaType as MT
from devops_agent.v8.graph import Rel

# ---------------------------- ЗАКОН КАК ДАННЫЕ ----------------------------------

LEGAL_EDGES = {
    (MT.ENTITY, Rel.HAS, MT.RESOURCE),
    (MT.ENTITY, Rel.HAS, MT.SETTING),
    (MT.ENTITY, Rel.HAS_STATUS, MT.STATUS),
    (MT.ENTITY, Rel.DEPENDS_ON, MT.ENTITY),
    (MT.INTERVENTION, Rel.ESTABLISHES, MT.RESOURCE),
    (MT.INTERVENTION, Rel.ESTABLISHES, MT.SETTING),
    (MT.INTERVENTION, Rel.ESTABLISHES, MT.STATUS),
    (MT.INTERVENTION, Rel.REQUIRES, MT.RESOURCE),
    (MT.INTERVENTION, Rel.REQUIRES, MT.SETTING),
    (MT.INTERVENTION, Rel.REQUIRES, MT.STATUS),
    (MT.INTERVENTION, Rel.USES, MT.ENTITY),
    (MT.INTERVENTION, Rel.ACTS_ON, MT.ENTITY),
}

KINDS = {
    MT.RESOURCE: {"ordered", "bounded"},
    MT.SETTING: {"categorical", "boolean"},
}

# узловые инварианты — (тип, человекочитаемая формулировка, проверка ниже в validate)
INVARIANTS = [
    "resource/setting принадлежит ровно объявленной entity (HAS)",
    "status принадлежит объявленной entity (HAS_STATUS)",
    "intervention обязана establishes ≥1 (действие без эффекта нелегально)",
    "goal ⊆ объявленных status-id",
    "id непустые; в одной мете id уникален across типов",
]

QUARANTINED = ["if/условное (guard = requires)", "part-of/композиция (кейс Chef)"]


def validate(g) -> list:
    """Структурные нарушения упаковки одной меты. Пусто = корректно. Контент-нейтрально."""
    v = []
    # 1. рёбра: существование концов + легальность сигнатуры
    for e in g.edges:
        if e.src not in g.nodes:
            v.append(f"ребро-на-несуществующий-src:{e.src}->{e.dst}")
            continue
        if e.dst not in g.nodes:
            v.append(f"ребро-на-несуществующий-dst:{e.src}->{e.dst}")
            continue
        sig = (g.nodes[e.src].mtype, e.rel, g.nodes[e.dst].mtype)
        if sig not in LEGAL_EDGES:
            v.append(f"нелегальное-ребро:{sig[0].value}-{sig[1].value}->{sig[2].value}:{e.src}->{e.dst}")
    # 2. узловые инварианты
    for n in g.nodes.values():
        if not n.id:
            v.append("пустой-id")
        if n.mtype in (MT.RESOURCE, MT.SETTING):
            if not g.has_incoming(n.id, Rel.HAS):
                v.append(f"безхозный-{n.mtype.value}:{n.id}")
            kind = n.attrs.get("kind")
            if kind is not None and kind not in KINDS[n.mtype]:
                v.append(f"нелегальный-kind:{n.mtype.value}:{n.id}:{kind}")
        if n.mtype == MT.STATUS and not g.has_incoming(n.id, Rel.HAS_STATUS):
            v.append(f"status-без-entity:{n.id}")
        if n.mtype == MT.INTERVENTION and not g.out(n.id, Rel.ESTABLISHES):
            v.append(f"intervention-без-эффекта:{n.id}")
    # 3. цель
    st = g.ids_of(MT.STATUS)
    for gid in g.goal:
        if gid not in st:
            v.append(f"goal-не-status:{gid}")
    # 4. логический слой
    v += validate_constraints(g.layer, g.ids_of(MT.RESOURCE), st, g.ids_of(MT.SETTING))
    return v


# ------------------- ЗАКОН КАК ПРОМПТ (из тех же таблиц) -------------------------

def _law_lines():
    by_src = {}
    for s, r, d in sorted(LEGAL_EDGES, key=lambda x: (x[0].value, x[1].value, x[2].value)):
        by_src.setdefault((s, r), []).append(d.value)
    for (s, r), ds in by_src.items():
        yield f"{s.value}.{r.value} -> {'|'.join(ds)}"


LAW_PROMPT = (
    "STRUCTURAL LAW (violations are rejected by a mechanical gate):\n"
    + "".join(f"  - {l}\n" for l in _law_lines())
    + "  - intervention role slots — distinguish THREE roles, never conflate: "
      "acts_on = the PATIENT entity the action operates ON / changes (configure server -> server); "
      "uses = the INSTRUMENT entity the action is done WITH (configure server with ansible -> ansible); "
      "requires = STATE preconditions only (declared resource/setting/status ids). "
      "The agent/doer and the audience are NOT slots — omit them\n"
      "  - every intervention MUST establish >=1 declared id (its effect; e.g. develop_cookbooks "
      "establishes cookbook_developed status)\n"
      "  - every resource/setting/status MUST carry \"entity\": <declared entity id> (its owner)\n"
      "  - goal is a list of DECLARED status ids (empty for a pure fact)\n"
      "  - resource.kind in [ordered,bounded]; setting.kind in [categorical,boolean]\n"
      "  - ids: snake_case English, unique across ALL lists of this one structure\n"
      "  - constraints: mutex = exactly 2 declared same-kind ids; atmost = >=2 declared same-kind "
      "ids sharing ONE budget with integer bound; no genuine either-or/shared-limit => []"
)

# Схема меты (JSON-вход компилятора). uses — новое поле по закону USES.
META_SCHEMA = {
    "entities": "list of {id} — the things",
    "resources": "list of {id, entity, kind} kind='ordered'|'bounded' — quantitative adjustable properties",
    "settings": "list of {id, entity, kind} kind='categorical'|'boolean' — non-quantitative settings",
    "statuses": "list of {id, entity} — state axes (health/goal states) of a declared entity",
    "interventions": ("list of {id, establishes, requires, uses, acts_on} — actions; "
                      "establishes/requires are lists of declared resource/setting/status ids; "
                      "uses = declared entity ids the action is done WITH (instruments); "
                      "acts_on = declared entity ids the action operates ON (patients)"),
    "dependencies": "list of {from, to} — entity 'from' depends on entity 'to'",
    "goal": "list of declared status ids that are the desired target (empty for a pure fact)",
    "constraints": ("list of {kind, members, bound} — logical layer: kind='mutex' (2 same-kind ids, "
                    "hard either-or) or 'atmost' (>=2 same-kind ids sharing one budget, integer bound)"),
}
