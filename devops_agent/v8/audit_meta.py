"""
audit_meta.py — аудит того, что РЕАЛЬНО легло в meta-KB граф (не верность судьи).

Читает datasets/tasks.jsonl и проверяет каждую meta-структуру против контракта 6 мета-типов
(ISA v4): ссылки не висят, виды рычагов легальны, dependency только Entity→Entity,
goal → существующий Status, status привязан к entity. Без LLM, мгновенно.
"""

import collections
import json
from pathlib import Path

DS = Path(__file__).resolve().parent / "datasets" / "tasks.jsonl"


def ent_ids(meta: dict):
    ids, weird = set(), False
    for e in meta.get("entities", []):
        if isinstance(e, dict):
            ids.add(e.get("id"))
        else:
            ids.add(e)
            weird = True                      # entities строками — не по контракту
    return ids, weird


def audit_one(meta: dict) -> list:
    v = []
    ents, weird = ent_ids(meta)
    if weird:
        v.append("entities-as-strings")
    res_ids, set_ids, st_ids = set(), set(), set()
    for r in meta.get("resources", []):
        res_ids.add(r.get("id"))
        if r.get("entity") not in ents:
            v.append("resource→unknown-entity")
        if r.get("kind") not in ("ordered", "bounded"):
            v.append("resource-bad-kind")
    for s in meta.get("settings", []):
        set_ids.add(s.get("id"))
        if s.get("entity") not in ents:
            v.append("setting→unknown-entity")
        if s.get("kind") not in ("categorical", "boolean"):
            v.append("setting-bad-kind")
    for s in meta.get("statuses", []):
        st_ids.add(s.get("id"))
        if "entity" not in s:
            v.append("status-no-entity")
        elif s.get("entity") not in ents:
            v.append("status→unknown-entity")
    known = res_ids | set_ids | st_ids
    for iv in meta.get("interventions", []):
        for k in ("establishes", "requires"):
            for ref in iv.get(k, []):
                if ref not in known:
                    v.append(f"intervention-{k}→unknown-id")
    for d in meta.get("dependencies", []):
        if d.get("from") not in ents or d.get("to") not in ents:
            v.append("dependency-not-Entity→Entity")
    for g in meta.get("goal", []):
        if g not in st_ids:
            v.append("goal→unknown-status")
    return v


def main() -> None:
    rows = [json.loads(l) for l in DS.open()]
    n = len(rows)
    type_tot = collections.Counter()
    viol = collections.Counter()
    clean = 0
    for r in rows:
        m = r["meta"]
        for k in ("entities", "resources", "settings", "statuses", "interventions", "dependencies", "goal"):
            type_tot[k] += len(m.get(k, []))
        vs = audit_one(m)
        if not vs:
            clean += 1
        for x in set(vs):
            viol[x] += 1

    print(f"строк: {n} | структурно ЧИСТЫХ графов: {clean}/{n}")
    print("среднее на граф по мета-типам:", {k: round(type_tot[k] / n, 1) for k in type_tot})
    print("\nнарушения (в скольких графах встречается):")
    for x, c in viol.most_common():
        print(f"  {c:3}  {x}")

    print("\n— примеры малформ —")
    shown = 0
    for r in rows:
        vs = audit_one(r["meta"])
        if vs and shown < 3:
            print(f"  human: {r['human'][:90]}")
            print(f"  meta : {json.dumps(r['meta'], ensure_ascii=False)[:320]}")
            print(f"  нарушения: {sorted(set(vs))}\n")
            shown += 1


if __name__ == "__main__":
    main()
