"""
book_graph.py — ТЕСТ на ЖИВОМ тексте: книга (PDF→markdown) вместо синтетики NeMo.

Вопрос эксперимента: что вырастает из человеческого текста, который НЕ писался
под наши оси? NeMo-датасет генерился ПОД схему (tradeoff-оси и т.п.) — книга
"DevOps for Dummies" ничего о 6 метатипах не знает. Это novel-instance тест
для NLU-компилятора: скармливаем абзацы, смотрим какой граф типов вырастает.

  [книга.md] → абзацы → фильтр (мусор/оглавления) → NLU-компилятор (SCHEMA v8)
             → пер-абзацные меты → АГРЕГАЦИЯ в один граф:
                 узлы: entity / resource / setting / status
                 рёбра: dependency, intervention(establishes/requires), constraint

Выход: graph.json + graph.mermaid + консольный scorecard.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_root = HERE.parents[1]                # корень репо: там utils_azure
for p in (str(_root), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils_azure import AzureJSON

import kb_store

# схема = extract_constraints.SCHEMA (6 метатипов + логический слой), без изменений
SCHEMA = {
    "entities": "list of {id} — the things",
    "resources": "list of {id, entity, kind} kind='ordered'|'bounded' — quantitative adjustable properties",
    "settings": "list of {id, entity, kind} kind='categorical'|'boolean' — non-quantitative settings",
    "statuses": "list of {id, entity} — health/goal states",
    "interventions": "list of {id, establishes, requires} — actions; establishes/requires are id lists",
    "dependencies": "list of {from, to} — entity 'from' depends on entity 'to'",
    "goal": "list of status ids that are the desired target (empty for a pure fact)",
    "constraints": (
        "list of {kind, members, bound} — the LOGICAL LAYER over the 6 types. "
        "kind='mutex' (exactly TWO things cannot both be ON / a hard either-or; members=[id,id]) OR "
        "kind='atmost' (TWO OR MORE things share ONE limited budget; members=[id,id,...], bound=integer). "
        "RULES: (1) every member MUST be a DECLARED id; (2) model an either-or as boolean SETTINGS and "
        "constrain THOSE, never an entity id; (3) all members same kind; (4) atmost needs >=2 members; "
        "(5) no genuine shared-limit/either-or => constraints: []."
    ),
}
COMPILER_SYS = (
    "You are the NLU COMPILER of an agent that thinks ONLY in six meta-types PLUS a thin logical layer. "
    "Translate the human text into that structure; named things are ids only (snake_case English), no "
    "domain words in structure. ENTITY (a thing); RESOURCE (quantitative knob, kind 'ordered'|'bounded'); "
    "SETTING (non-quantitative knob, kind 'categorical'|'boolean'); STATUS (health/goal state); "
    "INTERVENTION (action establishing states, may require others); DEPENDENCY (entity depends on entity). "
    "LOGICAL LAYER: MUTEX = hard either-or of two booleans; ATMOST = several things share ONE limited "
    "budget. Use ONLY what the text implies; do not invent. The text is from a BOOK — it may be "
    "narrative/advice with no concrete system at all; then emit empty lists. Respond JSON only."
)


# ---------------------------- нарезка книги ------------------------------------

def paragraphs(md_path: Path, min_words=25, max_words=180):
    """Абзацы книги, отфильтрованные от оглавлений/заголовков/копирайта."""
    text = md_path.read_text(errors="ignore")
    raw = re.split(r"\n\s*\n", text)
    out = []
    for p in raw:
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r"^#+\s*", "", p)
        w = p.split()
        if not (min_words <= len(w) <= max_words):
            continue
        letters = sum(c.isalpha() for c in p)
        if letters / max(len(p), 1) < 0.6:                 # таблицы/оглавления/номера
            continue
        if re.search(r"(copyright|all rights reserved|isbn|trademark)", p, re.I):
            continue
        out.append(p)
    return out


# ---------------------------- агрегация в граф ----------------------------------

def aggregate(metas):
    """Пер-абзацные меты → один граф типов. Узлы мержим по id (плоско, честно)."""
    nodes = {}                                            # id -> {id, meta_type, kind?, entity?, freq}
    edges = []                                            # {kind, from, to, para}
    con_stats = Counter()

    def items(meta, key):
        """LLM иногда выдаёт список строк вместо {id,...} — коэрцируем, не роняем прогон."""
        for x in meta.get(key) or []:
            if isinstance(x, dict):
                yield x
            elif isinstance(x, str):
                yield {"id": x}

    def touch(nid, meta_type, **attrs):
        if not nid:
            return
        n = nodes.setdefault(nid, {"id": nid, "meta_type": meta_type, "freq": 0, **attrs})
        n["freq"] += 1
        if n["meta_type"] != meta_type:                   # коллизия типов между абзацами — метка
            n.setdefault("type_conflicts", set()).add(meta_type)

    for pi, meta in metas:
        if not isinstance(meta, dict):
            continue
        for e in items(meta, "entities"):
            touch(str(e.get("id", "")), "entity")
        for r in items(meta, "resources"):
            touch(str(r.get("id", "")), "resource", kind=r.get("kind"), entity=r.get("entity"))
        for s in items(meta, "settings"):
            touch(str(s.get("id", "")), "setting", kind=s.get("kind"), entity=s.get("entity"))
        for s in items(meta, "statuses"):
            touch(str(s.get("id", "")), "status", entity=s.get("entity"))
        for d in items(meta, "dependencies"):
            f, t = str(d.get("from", "")), str(d.get("to", ""))
            if f and t:
                edges.append({"kind": "dependency", "from": f, "to": t, "para": pi})
        for iv in items(meta, "interventions"):
            iid = str(iv.get("id", ""))
            touch(iid, "intervention")
            for est in iv.get("establishes", []) or []:
                if iid and str(est):                       # пустой id -> не ребро (баг вскрыт kb_qa)
                    edges.append({"kind": "establishes", "from": iid, "to": str(est), "para": pi})
            for req in iv.get("requires", []) or []:
                if iid and str(req):
                    edges.append({"kind": "requires", "from": iid, "to": str(req), "para": pi})
        for c in meta.get("constraints", []) or []:
            k = c.get("kind")
            mem = [str(m) for m in (c.get("members") or [])]
            if k in ("mutex", "atmost") and len(mem) >= 2:
                con_stats[k] += 1
                for a in mem:
                    for b in mem:
                        if a < b:
                            edges.append({"kind": k, "from": a, "to": b, "para": pi,
                                          **({"bound": c.get("bound")} if k == "atmost" else {})})
    return nodes, edges, con_stats


def to_mermaid(nodes, edges, top=40):
    """Ядро графа (top-N узлов по freq + их рёбра) в mermaid."""
    keep = {n["id"] for n in sorted(nodes.values(), key=lambda n: -n["freq"])[:top]}
    shape = {"entity": ("[", "]"), "resource": ("([", "])"), "setting": ("{{", "}}"),
             "status": ("((", "))"), "intervention": (">", "]")}
    lines = ["graph LR"]
    for nid in sorted(keep):
        n = nodes[nid]
        l, r = shape.get(n["meta_type"], ("[", "]"))
        lines.append(f'  {nid}{l}"{nid} ({n["meta_type"]},{n["freq"]})"{r}')
    arrow = {"dependency": "-->", "establishes": "==>", "requires": "-.->",
             "mutex": "<-. mutex .->", "atmost": "<-. atmost .->"}
    seen = set()
    for e in edges:
        if e["from"] in keep and e["to"] in keep:
            key = (e["kind"], e["from"], e["to"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f'  {e["from"]} {arrow[e["kind"]]} {e["to"]}')
    return "\n".join(lines)


# ---------------------------- прогон -------------------------------------------

def main():
    md = Path(sys.argv[1])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    paras = paragraphs(md)
    print(f"книга: {md.name}; абзацев после фильтра: {len(paras)}; компилирую первые {cap}\n")

    az = AzureJSON()
    metas, empty, failed = [], 0, 0
    for i, p in enumerate(paras[:cap]):
        try:
            meta = az.ask(system=COMPILER_SYS, user=f"book paragraph: {p}", schema=SCHEMA)
        except Exception as ex:
            failed += 1
            print(f"[{i}] LLM-FAIL: {ex}")
            continue
        n_things = sum(len(meta.get(k, []) or []) for k in
                       ("entities", "resources", "settings", "statuses", "interventions"))
        if n_things == 0:
            empty += 1
        metas.append((i, meta))
        ncon = len(meta.get("constraints", []) or [])
        ndep = len(meta.get("dependencies", []) or [])
        print(f"[{i}] things={n_things} deps={ndep} constraints={ncon} :: {p[:70]}…")

    nodes, edges, con_stats = aggregate(metas)
    by_type = Counter(n["meta_type"] for n in nodes.values())
    edge_kinds = Counter(e["kind"] for e in edges)

    print("\n" + "=" * 64)
    print(f"абзацев скомпилировано: {len(metas)}  (пустых мет: {empty}, LLM-fail: {failed})")
    print(f"узлы графа:  {dict(by_type)}  (всего {len(nodes)})")
    print(f"рёбра:       {dict(edge_kinds)}  (всего {len(edges)})")
    print(f"связки:      {dict(con_stats)}")
    conf = [n for n in nodes.values() if "type_conflicts" in n]
    print(f"конфликты метатипа между абзацами: {len(conf)}"
          + (f"  напр. {[n['id'] for n in conf[:5]]}" if conf else ""))
    hubs = sorted(nodes.values(), key=lambda n: -n["freq"])[:10]
    print("hub-узлы (freq): " + ", ".join(f"{n['id']}({n['meta_type']},{n['freq']})" for n in hubs))

    out = HERE / "book_out"
    out.mkdir(exist_ok=True)
    for n in nodes.values():
        if "type_conflicts" in n:
            n["type_conflicts"] = sorted(n["type_conflicts"])
    (out / "graph.json").write_text(json.dumps(
        {"nodes": list(nodes.values()), "edges": edges}, ensure_ascii=False, indent=1))
    (out / "graph.mermaid").write_text(to_mermaid(nodes, edges))
    (out / "metas.jsonl").write_text("\n".join(json.dumps(
        {"para": i, "meta": m}, ensure_ascii=False) for i, m in metas))
    print(f"\nсохранено: {out}/graph.json, graph.mermaid, metas.jsonl")

    # ---- ЗНАНИЕ НЕ ИСПАРЯЕТСЯ: вливаем прогон в персистентную KB ----
    run_id = sys.argv[3] if len(sys.argv) > 3 else f"{md.stem[:24]}-{cap}"
    rep = kb_store.merge_run(run_id, nodes, edges, source=md.name)
    print(f"KB merge [{run_id}]: +{len(rep['new_nodes'])} узлов, ~{rep['bumped_nodes']} подкреплено, "
          f"+{len(rep['new_edges'])} рёбер, споров о типе: {len(set(rep['type_disputes']))}")
    print(f"анализ: python -m devops_agent.v8.kb_store show|history|run {run_id}")


if __name__ == "__main__":
    main()
