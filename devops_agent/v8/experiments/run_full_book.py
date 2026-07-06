"""Прогон абзацев 30..226 книги через компилятор, чанками -> KB (по run на чанк)."""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

for line in open('/private/tmp/ccc/.env'):
    line = line.strip()
    if '=' not in line or line.startswith('#'):
        continue
    k, v = line.split('=', 1)
    os.environ[{'AZURE_OPENAI_ENDPROINT': 'AZURE_OPENAI_ENDPOINT',
                'AZURE_OPENAI_KEY': 'AZURE_API_KEY'}.get(k.strip(), k.strip())] = v.strip().strip('"\'')

sys.path.insert(0, '/private/tmp/ccc/devops_agent/v8')
sys.path.insert(0, '/private/tmp/ccc')
from pathlib import Path
from book_graph import COMPILER_SYS, SCHEMA, aggregate, paragraphs
import kb_store

MD = Path('/private/tmp/ccc/docspdf/output/eBook_DevOps_for-Dummies.md')
OUT = Path('/private/tmp/ccc/devops_agent/v8/book_out')
OUT.mkdir(exist_ok=True)
paras = paragraphs(MD)
LO, HI, CHUNK = 30, len(paras), 49

_tl = threading.local()
def _az():
    if not hasattr(_tl, 'az'):
        from utils_azure import AzureJSON
        _tl.az = AzureJSON()
    return _tl.az

def compile_one(i):
    try:
        return i, _az().ask(system=COMPILER_SYS, user=f"book paragraph: {paras[i]}", schema=SCHEMA)
    except Exception as ex:
        return i, {"__fail__": str(ex)}

metas_path = OUT / 'metas_full.jsonl'
done = {}                                   # уже скомпилированное — не жжём Azure повторно
if metas_path.exists():
    for l in metas_path.open():
        r = json.loads(l)
        done[r["para"]] = r["meta"]
merged_runs = set()
jp = Path('/private/tmp/ccc/devops_agent/v8/kb/journal.jsonl')
if jp.exists():
    merged_runs = {json.loads(l)["run"] for l in jp.open()}

for lo in range(LO, HI, CHUNK):
    hi = min(lo + CHUNK, HI)
    run = f"dummies-p{lo:03d}-{hi-1:03d}"
    if run in merged_runs:
        print(f"[{run}] уже в KB, пропускаю", flush=True)
        continue
    todo = [i for i in range(lo, hi) if i not in done]
    with ThreadPoolExecutor(4) as ex:
        results = sorted(ex.map(compile_one, todo))
    good = [(i, m) for i, m in results if "__fail__" not in m]
    fails = [(i, m["__fail__"]) for i, m in results if "__fail__" in m]
    with metas_path.open('a') as f:
        for i, m in good:
            f.write(json.dumps({"para": i, "meta": m}, ensure_ascii=False) + "\n")
            done[i] = m
    chunk_metas = [(i, done[i]) for i in range(lo, hi) if i in done]
    empty = sum(1 for _, m in chunk_metas if not any(m.get(k) for k in
                ("entities", "resources", "settings", "statuses", "interventions")))
    nodes, edges, cons = aggregate(chunk_metas)
    rep = kb_store.merge_run(run, nodes, edges, source=MD.name)
    print(f"[{run}] абзацы {lo}-{hi-1}: пустых {empty}/{len(chunk_metas)}, fail {len(fails)} | "
          f"KB: +{len(rep['new_nodes'])} узлов, ~{rep['bumped_nodes']} подкреплено, "
          f"+{len(rep['new_edges'])} рёбер, связки {dict(cons)}, споры {len(set(rep['type_disputes']))}",
          flush=True)
    for i, e in fails[:3]:
        print(f"    fail[{i}]: {e[:100]}", flush=True)
print("DONE", flush=True)
