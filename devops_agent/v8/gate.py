"""
gate.py — КОНТРАКТ-ГЕЙТ между NLU-компилятором и meta-KB.

Переиспускает meta-JSON компилятора в v4.KnowledgeGraph и валидирует ровно через
v4/contract.py (легальные рёбра 6 мета-типов + узловые инварианты). Малформ → ретрай:
нарушения прокидываются обратно компилятору, он чинит. В KB ложится только структурно-чистое.

Так первый слой ground-truth (структура), который мы пропустили, встаёт на место —
семантику (судью) меряем уже на валидных графах.
"""

import json
import sys
from pathlib import Path

from devops_agent.v4 import contract as v4c
from devops_agent.v4.graph import KnowledgeGraph, MetaType, Rel
from devops_agent.v8.metagen import META_SCHEMA, COMPILER_SYS
from utils_azure import AzureJSON

DS = Path(__file__).resolve().parent / "datasets" / "tasks.jsonl"

FIX_SYS = COMPILER_SYS + (
    " Your previous structure FAILED the structural contract. Re-emit CORRECTED JSON obeying: "
    "(1) every id in intervention.establishes/requires, dependencies and goal must be DECLARED "
    "as an entity/resource/setting/status; (2) entity field of resource/setting/status must be a "
    "DECLARED entity id; (3) dependency.from and .to must both be ENTITY ids; (4) entities are "
    "objects {id}; (5) every intervention must establish at least one thing; (6) resource.kind in "
    "[ordered,bounded], setting.kind in [categorical,boolean]. Fix exactly the listed violations."
)


def to_graph(meta: dict):
    """meta-JSON → v4.KnowledgeGraph (+ нарушения сборки, которых граф не выражает)."""
    g, build_v = KnowledgeGraph(), []

    def node(nid, mt):
        if not isinstance(nid, str) or not nid:
            build_v.append(f"пустой id у {mt.value}")
            return
        try:
            g.add_node(nid, mt)
        except ValueError as e:
            build_v.append(f"id-коллизия: {e}")

    for e in meta.get("entities", []):
        if isinstance(e, dict):
            node(e.get("id"), MetaType.ENTITY)
        else:
            node(e, MetaType.ENTITY)
            build_v.append("entity-строкой, не {id}")
    for r in meta.get("resources", []):
        node(r.get("id"), MetaType.RESOURCE)
        if r.get("kind") not in ("ordered", "bounded"):
            build_v.append(f"resource-bad-kind:{r.get('kind')}")
        if r.get("entity"):
            g.add_edge(r["entity"], Rel.HAS, r.get("id"))
    for s in meta.get("settings", []):
        node(s.get("id"), MetaType.SETTING)
        if s.get("kind") not in ("categorical", "boolean"):
            build_v.append(f"setting-bad-kind:{s.get('kind')}")
        if s.get("entity"):
            g.add_edge(s["entity"], Rel.HAS, s.get("id"))
    for s in meta.get("statuses", []):
        node(s.get("id"), MetaType.STATUS)
        if s.get("entity"):
            g.add_edge(s["entity"], Rel.HAS_GOAL, s.get("id"))
        else:
            build_v.append(f"status без entity: {s.get('id')!r}")
    for iv in meta.get("interventions", []):
        node(iv.get("id"), MetaType.INTERVENTION)
        for ref in iv.get("establishes", []):
            g.add_edge(iv.get("id"), Rel.ESTABLISHES, ref)
        for ref in iv.get("requires", []):
            g.add_edge(iv.get("id"), Rel.REQUIRES, ref)
    for d in meta.get("dependencies", []):
        if d.get("from") and d.get("to"):
            g.add_edge(d["from"], Rel.DEPENDS_ON, d["to"])
    return g, build_v


def validate_meta(meta: dict) -> list:
    g, build_v = to_graph(meta)
    return build_v + v4c.validate(g)


def repair(human: str, meta: dict, az: AzureJSON, max_fix: int = 2):
    """Ретрай: прокидываем нарушения компилятору, пока не чисто или не исчерпан лимит."""
    viol = validate_meta(meta)
    attempts = 0
    while viol and attempts < max_fix:
        meta = az.ask(system=FIX_SYS,
                      user=f"human: {human}\nprevious: {json.dumps(meta, ensure_ascii=False)}\n"
                           f"violations: {sorted(set(viol))}",
                      schema=META_SCHEMA)
        attempts += 1
        viol = validate_meta(meta)
    return meta, attempts, sorted(set(viol))


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = [json.loads(l) for l in DS.open()]
    clean0 = [r for r in rows if not validate_meta(r["meta"])]
    dirty = [r for r in rows if validate_meta(r["meta"])]
    print(f"в meta-KB сейчас: чистых {len(clean0)}/{len(rows)}, малформ {len(dirty)}")
    print(f"прогоняю гейт (ретрай) на {min(cap, len(dirty))} малформ-графах из {len(dirty)}:\n")

    az = AzureJSON()
    fixed = irreparable = 0
    residue = {}
    for r in dirty[:cap]:
        before = sorted(set(validate_meta(r["meta"])))
        meta2, att, after = repair(r["human"], r["meta"], az)
        ok = not after
        fixed += ok
        irreparable += (not ok)
        tag = "ПОЧИНЕН" if ok else "НЕ ЧИНИТСЯ"
        print(f"  [{tag} за {att}] {r['human'][:70]}")
        print(f"      было: {before}")
        if not ok:
            print(f"      осталось: {after}")
            for x in after:
                residue[x] = residue.get(x, 0) + 1
    print(f"\nна выборке: починено {fixed}, не чинится {irreparable}")
    if residue:
        print("неустранимый остаток (кандидаты в дыры схемы E4/E5):")
        for x, c in sorted(residue.items(), key=lambda kv: -kv[1]):
            print(f"  {c}  {x}")
    proj = len(clean0) + round(len(dirty) * (fixed / max(1, fixed + irreparable)))
    print(f"\nпроекция на все 100 после гейта: ~{proj}/100 чистых "
          f"(было {len(clean0)}); остальное — на ретрай/в карантин как находки.")


if __name__ == "__main__":
    main()
