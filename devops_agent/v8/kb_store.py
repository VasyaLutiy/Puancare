"""
kb_store.py — ПЕРСИСТЕНТНАЯ KB: граф типов, накапливаемый МЕЖДУ экспериментами.

Дырка, которую закрываем: эксперимент прогнали → scorecard посмотрели → знание
испарилось. book_out/graph.json — дамп ОДНОГО прогона; анализировать «что у нас
накопилось в знаниях» и «что изменил конкретный эксперимент» было нечем.

Устройство (плоские файлы, всё инспектируемо руками):

  devops_agent/v8/kb/
    kb.json         — материализованное состояние KB (узлы+рёбра, с провенансом)
    journal.jsonl   — append-only журнал merge-событий: что каждый run ДОБАВИЛ
    runs/<run>.json — сырой граф каждого прогона (воспроизводимость/повторный merge)

Семантика merge (домен-иррелевантна, только 6 метатипов):
  • узлы сливаются по id; freq копится; provenance = [{run, freq}];
  • метатип узла = ГОЛОСОВАНИЕ между прогонами (votes) — конфликт не затирается,
    а хранится: расхождение типов между контекстами само по себе знание;
  • рёбра дедупятся по (kind, from, to); freq копится; provenance тот же.

CLI-анализ накопленного:
  python -m devops_agent.v8.kb_store show             — сводка KB
  python -m devops_agent.v8.kb_store history          — журнал прогонов (что добавил каждый)
  python -m devops_agent.v8.kb_store run <run_id>     — детальный дифф одного прогона
  python -m devops_agent.v8.kb_store conflicts        — узлы со спором о метатипе
  python -m devops_agent.v8.kb_store top [N]          — hub-узлы по freq
  python -m devops_agent.v8.kb_store mermaid [N]      — ядро KB в mermaid (stdout)
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "kb"


# ---------------------------- состояние ----------------------------------------

def load_kb():
    p = KB_DIR / "kb.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"nodes": {}, "edges": {}}          # edges: "kind|from|to" -> edge


def _save_kb(kb):
    KB_DIR.mkdir(exist_ok=True)
    (KB_DIR / "kb.json").write_text(json.dumps(kb, ensure_ascii=False, indent=1))


def _journal(event):
    KB_DIR.mkdir(exist_ok=True)
    with (KB_DIR / "journal.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------------------- merge --------------------------------------------

def merge_run(run_id: str, nodes: dict, edges: list, source: str = "",
              layer: str = "testimony") -> dict:
    """Влить граф одного прогона в KB. Возвращает отчёт: что именно добавилось.

    nodes: id -> {id, meta_type, freq, kind?, entity?}   (формат book_graph.aggregate)
    edges: [{kind, from, to, ...}]
    layer: эпистемический статус источника (решение 6 июл, из спарринга с форком):
      'testimony' — вычитано из текста (книга, NeMo-фразы) — БЕЗ исключений, жанр не угадываем;
      'grounded'  — заверено пробой мира-судьи.
    Промоушен testimony→grounded — ТОЛЬКО через promote() с уликой-пробой.
    """
    kb = load_kb()
    rep = {"run": run_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
           "layer": layer, "new_nodes": [], "bumped_nodes": 0, "new_edges": [], "dup_edges": 0,
           "type_disputes": []}

    for nid, n in nodes.items():
        cur = kb["nodes"].get(nid)
        # внутри-прогонные споры о типе (aggregate.type_conflicts) — тоже голоса, не теряем
        extra_votes = {t: 1 for t in n.get("type_conflicts", ())}
        if cur is None:
            kb["nodes"][nid] = {
                "id": nid,
                "layer": layer,
                "votes": {n["meta_type"]: n["freq"], **extra_votes},
                "freq": n["freq"],
                **({"kind": n["kind"]} if n.get("kind") else {}),
                **({"entity": n["entity"]} if n.get("entity") else {}),
                "sources": [{"run": run_id, "freq": n["freq"], "layer": layer}],
            }
            rep["new_nodes"].append(nid)
            if extra_votes:
                rep["type_disputes"].append(nid)
        else:
            cur["freq"] += n["freq"]
            cur["votes"][n["meta_type"]] = cur["votes"].get(n["meta_type"], 0) + n["freq"]
            for t, v in extra_votes.items():
                cur["votes"][t] = cur["votes"].get(t, 0) + v
            cur["sources"].append({"run": run_id, "freq": n["freq"], "layer": layer})
            if layer == "grounded":                       # grounded-источник поднимает узел
                cur["layer"] = "grounded"
            if len(cur["votes"]) > 1:
                rep["type_disputes"].append(nid)
            rep["bumped_nodes"] += 1

    for e in edges:
        key = f'{e["kind"]}|{e["from"]}|{e["to"]}'
        cur = kb["edges"].get(key)
        if cur is None:
            kb["edges"][key] = {"kind": e["kind"], "from": e["from"], "to": e["to"],
                                "layer": layer, "freq": 1, "sources": [run_id],
                                **({"bound": e["bound"]} if "bound" in e else {})}
            rep["new_edges"].append(key)
        else:
            cur["freq"] += 1
            if run_id not in cur["sources"]:
                cur["sources"].append(run_id)
            if layer == "grounded":
                cur["layer"] = "grounded"
            rep["dup_edges"] += 1

    (KB_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (KB_DIR / "runs" / f"{run_id}.json").write_text(json.dumps(
        {"nodes": nodes, "edges": edges, "source": source},
        ensure_ascii=False, default=sorted))      # set (type_conflicts из aggregate) -> list
    _save_kb(kb)
    _journal({**rep, "new_nodes": len(rep["new_nodes"]), "new_edges": len(rep["new_edges"]),
              "type_disputes": len(set(rep["type_disputes"]))})
    return rep


def meta_type(node: dict) -> str:
    """Доминирующий метатип узла по голосованию."""
    return max(node["votes"].items(), key=lambda kv: kv[1])[0]


def promote(ids: list, probe_evidence: str) -> int:
    """testimony → grounded. ЕДИНСТВЕННЫЙ легальный путь — предъявить пробу.

    probe_evidence: идентификатор пробы/прогона мира-судьи (не текст — улика).
    Возвращает число промоутнутых узлов/рёбер. Понижения нет: grounded
    опровергается только новой пробой (рефьют — отдельный механизм, впереди).
    """
    kb = load_kb()
    n = 0
    for i in ids:
        obj = kb["nodes"].get(i) or kb["edges"].get(i)
        if obj is not None and obj.get("layer") != "grounded":
            obj["layer"] = "grounded"
            obj.setdefault("probes", []).append(probe_evidence)
            n += 1
    _save_kb(kb)
    _journal({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": f"promote:{probe_evidence}",
              "source": probe_evidence, "layer": "grounded", "new_nodes": 0,
              "bumped_nodes": n, "new_edges": 0, "type_disputes": 0})
    return n


# ---------------------------- анализ (CLI) --------------------------------------

def _show(kb):
    by = Counter(meta_type(n) for n in kb["nodes"].values())
    ek = Counter(e["kind"] for e in kb["edges"].values())
    runs = {s["run"] for n in kb["nodes"].values() for s in n["sources"]}
    ln = Counter(n.get("layer", "?") for n in kb["nodes"].values())
    le = Counter(e.get("layer", "?") for e in kb["edges"].values())
    print(f"KB: {len(kb['nodes'])} узлов {dict(by)}")
    print(f"    {len(kb['edges'])} рёбер {dict(ek)}")
    print(f"    слои: узлы {dict(ln)}; рёбра {dict(le)}   (grounded = заверено пробой)")
    print(f"    прогонов влито: {len(runs)} -> {sorted(runs)}")
    disputes = [n for n in kb["nodes"].values() if len(n["votes"]) > 1]
    print(f"    узлов со спором о метатипе: {len(disputes)}")


def _history():
    p = KB_DIR / "journal.jsonl"
    if not p.exists():
        print("журнал пуст")
        return
    for l in p.open():
        e = json.loads(l)
        print(f'{e["ts"]}  {e["run"]:28s} +узлы {e["new_nodes"]:4d}  ~узлы {e["bumped_nodes"]:4d}  '
              f'+рёбра {e["new_edges"]:4d}  споры {e["type_disputes"]}  ({e["source"]})')


def _run_diff(run_id, kb):
    p = KB_DIR / "runs" / f"{run_id}.json"
    if not p.exists():
        print(f"нет такого прогона: {run_id}")
        return
    raw = json.loads(p.read_text())
    fresh = [nid for nid in raw["nodes"]
             if kb["nodes"].get(nid, {}).get("sources", [{}])[0].get("run") == run_id]
    print(f"прогон {run_id} (source={raw.get('source','')}):")
    print(f"  узлов в прогоне: {len(raw['nodes'])}, из них ВПЕРВЫЕ появились в KB: {len(fresh)}")
    for nid in fresh[:30]:
        n = kb["nodes"][nid]
        print(f"    + {nid} ({meta_type(n)}, freq={n['freq']})")
    ek = Counter(e["kind"] for e in raw["edges"])
    print(f"  рёбер в прогоне: {len(raw['edges'])} {dict(ek)}")


def _conflicts(kb):
    rows = [n for n in kb["nodes"].values() if len(n["votes"]) > 1]
    print(f"узлы со спором о метатипе: {len(rows)}")
    for n in sorted(rows, key=lambda n: -n["freq"]):
        print(f"  {n['id']:28s} votes={n['votes']}  (доминирует {meta_type(n)})")


def _top(kb, n=15):
    for node in sorted(kb["nodes"].values(), key=lambda x: -x["freq"])[:n]:
        print(f"  {node['id']:28s} {meta_type(node):12s} freq={node['freq']:3d} "
              f"runs={len({s['run'] for s in node['sources']})}")


def _mermaid(kb, top=40):
    keep = {n["id"] for n in sorted(kb["nodes"].values(), key=lambda x: -x["freq"])[:top]}
    shape = {"entity": ("[", "]"), "resource": ("([", "])"), "setting": ("{{", "}}"),
             "status": ("((", "))"), "intervention": (">", "]")}
    arrow = {"dependency": "-->", "depends_on": "-->", "establishes": "==>",
             "requires": "-.->", "uses": "-.->", "acts_on": "==>", "has": "---", "has_status": "---",
             "mutex": "<-. mutex .->", "atmost": "<-. atmost .->"}
    print("graph LR")
    for nid in sorted(keep):
        n = kb["nodes"][nid]
        l, r = shape.get(meta_type(n), ("[", "]"))
        print(f'  {nid}{l}"{nid} ({meta_type(n)},{n["freq"]})"{r}')
    for e in kb["edges"].values():
        if e["from"] in keep and e["to"] in keep:
            print(f'  {e["from"]} {arrow[e["kind"]]} {e["to"]}')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    kb = load_kb()
    if cmd == "show":
        _show(kb)
    elif cmd == "history":
        _history()
    elif cmd == "run":
        _run_diff(sys.argv[2], kb)
    elif cmd == "conflicts":
        _conflicts(kb)
    elif cmd == "top":
        _top(kb, int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    elif cmd == "mermaid":
        _mermaid(kb, int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
