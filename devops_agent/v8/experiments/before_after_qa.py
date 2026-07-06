"""Q&A до/после второй книги: тот же вопрос — двум состояниям KB.

«До» = текущая KB, отфильтрованная по провенансу (только dummies-источники,
freq пересчитан). Механика: kb_store.load_kb подменяется, kb_qa не меняется.
"""
import copy
import json
import os
import sys

for line in open('/private/tmp/ccc/.env'):
    line = line.strip()
    if '=' not in line or line.startswith('#'):
        continue
    k, v = line.split('=', 1)
    os.environ[{'AZURE_OPENAI_ENDPROINT': 'AZURE_OPENAI_ENDPOINT',
                'AZURE_OPENAI_KEY': 'AZURE_API_KEY'}.get(k.strip(), k.strip())] = v.strip().strip('"\'')

sys.path.insert(0, '/private/tmp/ccc/devops_agent/v8')
sys.path.insert(0, '/private/tmp/ccc')
import kb_qa
import kb_store

FULL = kb_store.load_kb()

def filtered(prefix):
    kb = {"nodes": {}, "edges": {}}
    for nid, n in FULL["nodes"].items():
        src = [s for s in n["sources"] if s["run"].startswith(prefix)]
        if not src:
            continue
        m = copy.deepcopy(n)
        m["sources"] = src
        m["freq"] = sum(s["freq"] for s in src)
        kb["nodes"][nid] = m
    for k, e in FULL["edges"].items():
        src = [r for r in e["sources"] if r.startswith(prefix)]
        if not src or e["from"] not in kb["nodes"] or e["to"] not in kb["nodes"]:
            continue
        m = copy.deepcopy(e)
        m["sources"] = src
        kb["edges"][k] = m
    return kb

BEFORE = filtered("dummies")

QUESTIONS = [
    "что ты знаешь про infrastructure?",
    "что влияет на configuration?",
    "что нельзя делать одновременно с automation?",
]

for q in QUESTIONS:
    print("=" * 74)
    print(f"❓ {q}\n")
    for label, kb in (("ДО (только Dummies)", BEFORE), ("ПОСЛЕ (+ IaC)", FULL)):
        kb_store.load_kb = lambda kb=kb: kb
        print(f"--- {label} ---")
        try:
            print(kb_qa.ask(q))
        except Exception as ex:
            print(f"[fail: {ex}]")
        print()
