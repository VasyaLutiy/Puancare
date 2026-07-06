"""
gate.py — v8 ГЕЙТ: единственная дверь между NLU-компилятором и KB.

Переписан заново 6 июл (решение Кирилла: не тянуть v4) после того, как книжный
пайплайн дважды прошёл МИМО гейта и 74% мет легли в KB беззаконными.

  meta-JSON → to_graph() → contract.validate()
     ├─ mode='quarantine' (массовое чтение книг, НОЛЬ LLM):
     │    детерминированное отмывание: requires→entity переезжает в USES (закон),
     │    незаконное отрезается и СЧИТАЕТСЯ (residue — сырьё для роста ISA);
     │    на выходе граф, чистый по построению.
     └─ mode='repair' (задачи, дорого): нарушения прокидываются обратно
          компилятору (LAW_PROMPT), он чинит; остаток — в карантин.

В KB ложится ТОЛЬКО то, что прошло гейт. Residue не мусор: mutex/atmost выросли
ровно из residue старого гейта; part-of (кейс Chef) — следующий кандидат.
"""

from __future__ import annotations

import json
from collections import Counter

from devops_agent.v8.constraints import AtMost, ConstraintLayer, Mutex
from devops_agent.v8.contract import LAW_PROMPT, META_SCHEMA, validate
from devops_agent.v8.graph import KnowledgeGraph, MetaType as MT, Rel


# ---------------------------- meta-JSON → граф ----------------------------------

def _items(meta, key):
    for x in meta.get(key) or []:
        if isinstance(x, dict):
            yield x
        elif isinstance(x, str):
            yield {"id": x}


def to_graph(meta: dict) -> tuple:
    """meta-JSON → (KnowledgeGraph, build-нарушения, которых граф не выражает)."""
    g, bv = KnowledgeGraph(), []

    def node(nid, mt, **attrs):
        if not isinstance(nid, str) or not nid:
            bv.append(f"пустой-id:{mt.value}")
            return None
        try:
            g.add_node(nid, mt, **attrs)
            return nid
        except ValueError:
            bv.append(f"id-коллизия:{nid}")
            return None

    for e in _items(meta, "entities"):
        node(str(e.get("id", "")), MT.ENTITY)
    for key, mt, rel in (("resources", MT.RESOURCE, Rel.HAS),
                         ("settings", MT.SETTING, Rel.HAS),
                         ("statuses", MT.STATUS, Rel.HAS_STATUS)):
        for r in _items(meta, key):
            nid = node(str(r.get("id", "")), mt, **({"kind": r["kind"]} if r.get("kind") else {}))
            if nid and r.get("entity"):
                g.add_edge(str(r["entity"]), rel, nid)
    for iv in _items(meta, "interventions"):
        iid = node(str(iv.get("id", "")), MT.INTERVENTION)
        if not iid:
            continue
        for ref in iv.get("establishes") or []:
            if str(ref):
                g.add_edge(iid, Rel.ESTABLISHES, str(ref))
        for ref in iv.get("requires") or []:
            if str(ref):
                g.add_edge(iid, Rel.REQUIRES, str(ref))
        for ref in iv.get("uses") or []:
            if str(ref):
                g.add_edge(iid, Rel.USES, str(ref))
        for ref in iv.get("acts_on") or []:
            if str(ref):
                g.add_edge(iid, Rel.ACTS_ON, str(ref))
    for d in _items(meta, "dependencies"):
        f, t = str(d.get("from", "")), str(d.get("to", ""))
        if f and t:
            g.add_edge(f, Rel.DEPENDS_ON, t)
    g.goal = [str(x) for x in meta.get("goal") or [] if str(x)]
    for c in meta.get("constraints") or []:
        mem = tuple(str(m) for m in (c.get("members") or []))
        if c.get("kind") == "mutex":
            g.layer.mutexes.append(Mutex(members=mem))
        elif c.get("kind") == "atmost":
            try:
                g.layer.atmosts.append(AtMost(members=mem, bound=int(c.get("bound", 0))))
            except (TypeError, ValueError):
                bv.append(f"atmost-bad-bound:{c.get('bound')}")
    return g, bv


# ---------------------------- карантин (механика, 0 LLM) ------------------------

def quarantine(g: KnowledgeGraph) -> Counter:
    """Детерминированное отмывание до законного графа. Возвращает residue-счётчик.

    Порядок важен: сначала законные ПЕРЕКЛАССИФИКАЦИИ, потом отрезание, потом
    каскад (сирота → отрезать её рёбра) до фикспойнта.
    """
    res = Counter()

    # 1) requires→entity — это USES по закону (инструмент). Переклассификация, не потеря.
    moved = []
    for e in list(g.edges):
        if (e.rel == Rel.REQUIRES and e.src in g.nodes and e.dst in g.nodes
                and g.nodes[e.dst].mtype == MT.ENTITY):
            g.edges.remove(e)
            g.add_edge(e.src, Rel.USES, e.dst)
            moved.append(e)
    res["requires→uses (переклассифицировано)"] = len(moved)

    # 2) establishes→entity: «действие создаёт вещь» — невыразимо (кандидат CREATES). Резать.
    for e in list(g.edges):
        if (e.rel == Rel.ESTABLISHES and e.dst in g.nodes
                and g.nodes[e.dst].mtype == MT.ENTITY):
            g.edges.remove(e)
            res["residue: establishes→entity (кандидат CREATES)"] += 1

    # 3) фикспойнт отрезаний
    changed = True
    while changed:
        changed = False
        # рёбра: висячие или нелегальные по сигнатуре
        for e in list(g.edges):
            if e.src not in g.nodes or e.dst not in g.nodes:
                g.edges.remove(e); res["residue: висячее ребро"] += 1; changed = True
                continue
            sig = (g.nodes[e.src].mtype, e.rel, g.nodes[e.dst].mtype)
            from devops_agent.v8.contract import LEGAL_EDGES
            if sig not in LEGAL_EDGES:
                g.edges.remove(e)
                res[f"residue: {sig[0].value}-{sig[1].value}->{sig[2].value}"] += 1
                changed = True
        # узлы: безхозные resource/setting/status; интервенции без эффекта
        for n in list(g.nodes.values()):
            if n.mtype in (MT.RESOURCE, MT.SETTING) and not g.has_incoming(n.id, Rel.HAS):
                g.drop_node(n.id); res[f"residue: безхозный {n.mtype.value}"] += 1; changed = True
            elif n.mtype == MT.STATUS and not g.has_incoming(n.id, Rel.HAS_STATUS):
                g.drop_node(n.id); res["residue: status без entity"] += 1; changed = True
            elif n.mtype == MT.INTERVENTION and not g.out(n.id, Rel.ESTABLISHES):
                g.drop_node(n.id); res["residue: intervention без эффекта"] += 1; changed = True
        # kind-мусор → снять атрибут (узел законен без kind? нет: kind обязателен смыслом,
        # но контракт ругает только нелегальный kind; неизвестный снимаем)
        for n in g.nodes.values():
            k = n.attrs.get("kind")
            if k is not None and n.mtype in (MT.RESOURCE, MT.SETTING):
                from devops_agent.v8.contract import KINDS
                if k not in KINDS[n.mtype]:
                    n.attrs.pop("kind"); res["residue: нелегальный kind (снят)"] += 1
    # 4) цель: только объявленные статусы
    st = g.ids_of(MT.STATUS)
    bad_goal = [x for x in g.goal if x not in st]
    if bad_goal:
        g.goal = [x for x in g.goal if x in st]
        res["residue: goal-не-status"] += len(bad_goal)
    # 5) связки: незаконные — в residue
    from devops_agent.v8.constraints import validate_constraints
    viol = validate_constraints(g.layer, g.ids_of(MT.RESOURCE), st, g.ids_of(MT.SETTING))
    if viol:
        keep_m, keep_a = [], []
        for m in g.layer.mutexes:
            l = ConstraintLayer(mutexes=[m])
            if not validate_constraints(l, g.ids_of(MT.RESOURCE), st, g.ids_of(MT.SETTING)):
                keep_m.append(m)
            else:
                res["residue: незаконный mutex"] += 1
        for a in g.layer.atmosts:
            l = ConstraintLayer(atmosts=[a])
            if not validate_constraints(l, g.ids_of(MT.RESOURCE), st, g.ids_of(MT.SETTING)):
                keep_a.append(a)
            else:
                res["residue: незаконный atmost"] += 1
        g.layer.mutexes, g.layer.atmosts = keep_m, keep_a
    return res


# ---------------------------- repair (LLM-ретрай) --------------------------------

FIX_SYS_TAIL = (
    "\nYour previous structure FAILED the structural law above. Re-emit CORRECTED JSON "
    "fixing exactly the listed violations. Keep everything that was already legal."
)


def repair(human: str, meta: dict, az, compiler_sys: str, max_fix: int = 2):
    """Нарушения → обратно компилятору. Возвращает (meta, попыток, остаток нарушений)."""
    g, bv = to_graph(meta)
    viol = bv + validate(g)
    attempts = 0
    while viol and attempts < max_fix:
        meta = az.ask(system=compiler_sys + "\n" + LAW_PROMPT + FIX_SYS_TAIL,
                      user=f"human: {human}\nprevious: {json.dumps(meta, ensure_ascii=False)}\n"
                           f"violations: {sorted(set(viol))[:20]}",
                      schema=META_SCHEMA)
        attempts += 1
        g, bv = to_graph(meta)
        viol = bv + validate(g)
    return meta, attempts, sorted(set(viol))


# ---------------------------- единая дверь ---------------------------------------

def gate_meta(meta: dict, mode: str = "quarantine", az=None, human: str = "",
              compiler_sys: str = "") -> tuple:
    """meta → (законный KnowledgeGraph, report). Единственный вход в KB."""
    if mode == "repair" and az is not None:
        meta, attempts, left = repair(human, meta, az, compiler_sys)
    g, bv = to_graph(meta)
    res = quarantine(g)
    for b in bv:
        res[f"build: {b.split(':')[0]}"] += 1
    leftover = validate(g)
    assert not leftover, f"гейт пропустил незаконное: {leftover[:5]}"   # чистота по построению
    return g, res
