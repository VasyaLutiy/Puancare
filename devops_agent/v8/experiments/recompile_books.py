"""Перекомпиляция обеих книг ЗАКОННЫМ компилятором (META_SCHEMA+LAW_PROMPT) через гейт → KB v2.

Каждая мета: compile → gate_meta(quarantine) → законный граф → агрегация чанка → merge_run.
Идемпотентно: сырые меты кешируются в book_out/metas2_<prefix>.jsonl.
"""
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for line in open('/private/tmp/ccc/.env'):
    line = line.strip()
    if '=' not in line or line.startswith('#'):
        continue
    k, v = line.split('=', 1)
    os.environ[{'AZURE_OPENAI_ENDPROINT': 'AZURE_OPENAI_ENDPOINT',
                'AZURE_OPENAI_KEY': 'AZURE_API_KEY'}.get(k.strip(), k.strip())] = v.strip().strip('"\'')

sys.path.insert(0, '/private/tmp/ccc')
sys.path.insert(0, '/private/tmp/ccc/devops_agent/v8')
from devops_agent.v8.contract import LAW_PROMPT, META_SCHEMA
from devops_agent.v8.gate import gate_meta
from devops_agent.v8.graph import MetaType as MT
from book_graph import paragraphs
import kb_store

COMPILER_SYS_V2 = (
    "You are the NLU COMPILER of an agent that thinks ONLY in six meta-types plus a thin logical "
    "layer. Translate the human text into that structure; named things are snake_case English ids "
    "only, no domain words in structure. ENTITY (a thing); RESOURCE (quantitative knob OF an entity, "
    "kind 'ordered'|'bounded'); SETTING (non-quantitative knob OF an entity, kind "
    "'categorical'|'boolean'); STATUS (state axis OF an entity: health/readiness/goal); INTERVENTION "
    "(an action: it USES tool-entities, REQUIRES state preconditions, ESTABLISHES its effects); "
    "DEPENDENCY (entity depends on entity).\n" + LAW_PROMPT +
    "\nUse ONLY what the text implies; do not invent. The text is from a BOOK — it may be "
    "narrative/advice with no concrete system at all; then emit empty lists. Respond JSON only."
)

BOOKS = [
    ("/private/tmp/ccc/docspdf/output/eBook_DevOps_for-Dummies.md", "dummies2"),
    ("/private/tmp/ccc/docspdf/output/IaC Thales.md", "iac2"),
]
OUT = Path('/private/tmp/ccc/devops_agent/v8/book_out')
CHUNK = 49

_tl = threading.local()
def _az():
    if not hasattr(_tl, 'az'):
        from utils_azure import AzureJSON
        _tl.az = AzureJSON()
    return _tl.az


def aggregate_graphs(items):
    """[(para, KnowledgeGraph)] → (nodes, edges) для kb_store.merge_run."""
    nodes, edges = {}, []
    for pi, g in items:
        for n in g.nodes.values():
            cur = nodes.setdefault(n.id, {"id": n.id, "meta_type": n.mtype.value, "freq": 0,
                                          **({"kind": n.attrs["kind"]} if n.attrs.get("kind") else {})})
            cur["freq"] += 1
            if cur["meta_type"] != n.mtype.value:
                cur.setdefault("type_conflicts", set()).add(n.mtype.value)
        for e in g.edges:
            edges.append({"kind": e.rel.value, "from": e.src, "to": e.dst, "para": pi})
        for m in g.layer.mutexes:
            a, b = sorted(m.members)[:2]
            edges.append({"kind": "mutex", "from": a, "to": b, "para": pi})
        for am in g.layer.atmosts:
            mem = sorted(am.members)
            for i, a in enumerate(mem):
                for b in mem[i+1:]:
                    edges.append({"kind": "atmost", "from": a, "to": b, "para": pi, "bound": am.bound})
    return nodes, edges


total_residue = Counter()
for md_path, prefix in BOOKS:
    md = Path(md_path)
    paras = paragraphs(md)
    metas_path = OUT / f"metas2_{prefix}.jsonl"
    done = {}
    if metas_path.exists():
        for l in metas_path.open():
            r = json.loads(l)
            done[r["para"]] = r["meta"]
    merged = set()
    jp = Path('/private/tmp/ccc/devops_agent/v8/kb/journal.jsonl')
    if jp.exists():
        merged = {json.loads(l)["run"] for l in jp.open()}

    def compile_one(i):
        try:
            return i, _az().ask(system=COMPILER_SYS_V2, user=f"book paragraph: {paras[i]}",
                                schema=META_SCHEMA)
        except Exception as ex:
            return i, {"__fail__": str(ex)}

    for lo in range(0, len(paras), CHUNK):
        hi = min(lo + CHUNK, len(paras))
        run = f"{prefix}-p{lo:03d}-{hi-1:03d}"
        if run in merged:
            print(f"[{run}] уже в KB, пропускаю", flush=True)
            continue
        todo = [i for i in range(lo, hi) if i not in done]
        with ThreadPoolExecutor(4) as ex:
            results = sorted(ex.map(compile_one, todo))
        fails = [(i, m["__fail__"]) for i, m in results if "__fail__" in m]
        with metas_path.open('a') as f:
            for i, m in results:
                if "__fail__" not in m:
                    f.write(json.dumps({"para": i, "meta": m}, ensure_ascii=False) + "\n")
                    done[i] = m
        gated, chunk_res, empty = [], Counter(), 0
        for i in range(lo, hi):
            meta = done.get(i)
            if meta is None:
                continue
            n_in = sum(len(meta.get(k) or []) for k in
                       ("entities", "resources", "settings", "statuses", "interventions"))
            if n_in == 0:
                empty += 1
                continue
            g, res = gate_meta(meta)
            chunk_res.update(res)
            gated.append((i, g))
        total_residue.update(chunk_res)
        nodes, edges = aggregate_graphs(gated)
        rep = kb_store.merge_run(run, nodes, edges, source=md.name, layer="testimony")
        moved = chunk_res.get("requires→uses (переклассифицировано)", 0)
        dropped = sum(c for k, c in chunk_res.items() if k.startswith("residue"))
        print(f"[{run}] пустых {empty}, fail {len(fails)} | гейт: uses+{moved}, residue−{dropped} | "
              f"KB: +{len(rep['new_nodes'])} узлов, ~{rep['bumped_nodes']}, +{len(rep['new_edges'])} рёбер",
              flush=True)

print("\n=== residue по всем книгам ===", flush=True)
for k, c in total_residue.most_common(12):
    print(f"  {c:5d}  {k}", flush=True)
print("DONE", flush=True)
