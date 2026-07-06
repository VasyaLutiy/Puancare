"""
kb_qa.py — Q&A-слой поверх kb_store: человек спрашивает — агент отвечает СВОИМ ГРАФОМ.

Архитектура зеркальна NLU-входу (решение 6 июл, спарринг с форком):

  [вопрос человека]
      │ LLM-переводчик: вопрос → формальный QUERY (конечная ISA вопросов)
      ▼
  QUERY = {op, target}          ← op из закрытого списка, target = узел KB
      │ МЕХАНИЧЕСКИЙ обход графа (ноль LLM — сгаллюцинировать НЕВОЗМОЖНО:
      ▼  что в графе лежит, то и вернётся, с layer и провенансом)
  FRAGMENT = кусок графа
      │ LLM-пересказчик: «перескажи РОВНО это, ничего не добавляй»
      ▼
  [ответ человеку, с пометками проверено/вычитано]

Правда живёт в среднем шаге. LLM на краях — только переводчики.
Вопрос вне ISA → честное «не понимаю» (unanswerable), не болтовня.

  python -m devops_agent.v8.kb_qa "что влияет на delivery_pipeline?"
  python -m devops_agent.v8.kb_qa --raw "..."     # показать QUERY и FRAGMENT (без пересказа)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_root = HERE.parents[1]
for p in (str(_root), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import kb_store
from utils_azure import AzureJSON

# ---------------------------- ISA вопросов (закрытый список) --------------------

OPS = {
    "affects":    "what influences/affects TARGET (incoming edges: deps, establishes, requires)",
    "impact":     "what breaks / is affected if TARGET is touched (outgoing edges from TARGET)",
    "about":      "everything known about TARGET (its card: type, layer, edges, constraints)",
    "exclusive":  "what cannot hold together (mutex/atmost), globally or around TARGET",
    "provenance": "where the knowledge about TARGET comes from (runs, probes)",
    "verified":   "what portion of knowledge is probe-verified vs read-from-text",
}
TRANSLATOR_SYS = (
    "You translate a human question into a formal query for a knowledge-graph agent. "
    "The agent ONLY answers queries from this fixed list:\n"
    + "\n".join(f"- {k}: {v}" for k, v in OPS.items()) +
    "\nReturn JSON {op, target, unanswerable, reason}. target = the thing asked about as a "
    "snake_case English id (or null for global ops like verified/exclusive-global). "
    "If the question does not fit ANY op (opinion, advice, chit-chat), set unanswerable=true "
    "with a short reason. Do NOT invent ops. JSON only."
)
TRANSLATOR_SCHEMA = {
    "op": f"one of {list(OPS)} or null",
    "target": "snake_case id of the thing asked about, or null",
    "unanswerable": "true if the question fits no op",
    "reason": "short reason if unanswerable, else empty",
}

RETELLER_SYS = (
    "Ты пересказчик ответа агента человеку. Тебе дают ВОПРОС и FRAGMENT — кусок графа знаний "
    "агента (JSON). Перескажи человеку РОВНО то, что в FRAGMENT, по-русски, коротко. "
    "ЖЁСТКИЕ ПРАВИЛА: ничего не добавлять от себя, не советовать, не интерпретировать, "
    "не использовать внешние знания. layer='testimony' => обязательно пометь «вычитано, не проверено». "
    "Слова «проверено»/«grounded» РАЗРЕШЕНЫ ТОЛЬКО если в fragment буквально стоит layer='grounded'. "
    "Если fragment пуст — так и скажи: в знаниях этого нет. "
    "JSON: {answer}"
)


# ---------------------------- механический обход (ноль LLM) ---------------------

def _find_node(kb, target):
    """target -> реальный id узла: точное совпадение, иначе подстрока (механика, не смысл)."""
    if not target:
        return None, []
    if target in kb["nodes"]:
        return target, []
    cand = sorted((n for n in kb["nodes"] if target in n or n in target),
                  key=lambda n: -kb["nodes"][n]["freq"])
    return (cand[0], cand[1:6]) if cand else (None, [])


def _edges_around(kb, nid, direction):
    out = []
    for e in kb["edges"].values():
        if direction == "in" and e["to"] == nid:
            out.append(e)
        elif direction == "out" and e["from"] == nid:
            out.append(e)
    return sorted(out, key=lambda e: -e["freq"])


def answer_query(q: dict) -> dict:
    """QUERY -> FRAGMENT. Чистый обход kb.json; никакого творчества."""
    kb = kb_store.load_kb()
    op = q.get("op")
    nid, near = _find_node(kb, q.get("target"))

    if op in ("affects", "impact", "about", "provenance") and nid is None:
        return {"op": op, "error": "no_such_node", "asked": q.get("target"),
                "nearest_ids": near or sorted(kb["nodes"], key=lambda n: -kb["nodes"][n]["freq"])[:5]}

    if op == "affects":
        return {"op": op, "node": nid, "resolved_from": q.get("target"),
                "incoming": _edges_around(kb, nid, "in")[:15]}
    if op == "impact":
        return {"op": op, "node": nid, "resolved_from": q.get("target"),
                "outgoing": _edges_around(kb, nid, "out")[:15]}
    if op == "about":
        n = kb["nodes"][nid]
        cons = [e for e in kb["edges"].values()
                if e["kind"] in ("mutex", "atmost") and nid in (e["from"], e["to"])]
        return {"op": op, "node": n, "resolved_from": q.get("target"),
                "incoming": _edges_around(kb, nid, "in")[:8],
                "outgoing": _edges_around(kb, nid, "out")[:8], "constraints": cons}
    if op == "exclusive":
        cons = [e for e in kb["edges"].values() if e["kind"] in ("mutex", "atmost")]
        if nid:
            cons = [e for e in cons if nid in (e["from"], e["to"])]
        return {"op": op, "node": nid, "constraints": sorted(cons, key=lambda e: -e["freq"])[:20]}
    if op == "provenance":
        n = kb["nodes"][nid]
        return {"op": op, "node": nid, "layer": n.get("layer"),
                "sources": n.get("sources", []), "probes": n.get("probes", [])}
    if op == "verified":
        from collections import Counter
        ln = Counter(n.get("layer") for n in kb["nodes"].values())
        le = Counter(e.get("layer") for e in kb["edges"].values())
        return {"op": op, "nodes_by_layer": dict(ln), "edges_by_layer": dict(le),
                "note": "grounded = заверено пробой мира-судьи; testimony = вычитано из текста"}
    return {"op": op, "error": "unknown_op"}


# ---------------------------- обвязка -------------------------------------------

def _is_empty(frag: dict) -> bool:
    """Пустой фрагмент = в графе НИЧЕГО нет по запросу. Определяется механикой.

    Урок эксперимента до/после (6 июл): пересказчик, получив constraints=[],
    сочинил и ответ, и метку «проверено пробой». Пустоту нельзя доверять LLM —
    пустой фрагмент вообще не должен до неё доезжать.
    """
    if frag.get("error"):
        return False                                     # ошибки пересказываем честно
    payload = {k: v for k, v in frag.items()
               if k not in ("op", "node", "resolved_from", "layer", "note")}
    lists = [v for v in payload.values() if isinstance(v, (list, dict))]
    return bool(lists) and all(not v for v in lists)


def ask(question: str, raw: bool = False) -> str:
    az = AzureJSON()
    q = az.ask(system=TRANSLATOR_SYS, user=question, schema=TRANSLATOR_SCHEMA)
    if q.get("unanswerable"):
        return f"[не понимаю: вне ISA вопросов] {q.get('reason', '')}"
    frag = answer_query(q)
    if not raw and _is_empty(frag):                      # мимо LLM — шаблон, не пересказ
        nid = frag.get("node")
        return (f"в знаниях про {nid!r} по этому вопросу — ПУСТО (op={frag['op']}); "
                "это не «нет связи в мире», а «в KB такой связи нет»")
    if raw:
        return ("QUERY:    " + json.dumps(q, ensure_ascii=False) + "\n"
                "FRAGMENT: " + json.dumps(frag, ensure_ascii=False, indent=1))
    r = az.ask(system=RETELLER_SYS,
               user=f"ВОПРОС: {question}\n\nFRAGMENT: {json.dumps(frag, ensure_ascii=False)}",
               schema={"answer": "короткий русский пересказ fragment"})
    return r.get("answer", "")


def main():
    args = [a for a in sys.argv[1:] if a != "--raw"]
    raw = "--raw" in sys.argv
    if not args:
        print(__doc__)
        return
    print(ask(" ".join(args), raw=raw))


if __name__ == "__main__":
    main()
