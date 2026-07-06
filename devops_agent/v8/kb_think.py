"""
kb_think.py — минимальная имитация «чанков мысли»: мысль = обход СВОЕГО графа.

Чанк мысли = ОДИН шаг обхода с эпистемической пометкой. Ноль LLM — думание
механическое, как и ответы kb_qa: галлюцинация невозможна, обрыв честен.

Схема шага (обратный вывод от цели):
  ХОЧУ X → кто устанавливает X? (establishes)     → чанк «есть действие A»
         → что требует A? (requires)              → рекурсия: ХОЧУ каждое R
         → от чего X зависит? (dependency)        → рекурсия: сначала Y
         → что несовместимо с X? (mutex/atmost)   → чанк-предупреждение
         → ничего из этого?                       → чанк «ОБРЫВ: пути в KB нет»

На testimony-KB мысль будет рваться быстро — это не дефект имитации, а честное
свойство знания-слухов: думать можно только по тем рёбрам, которые есть.

  python -m devops_agent.v8.kb_think <цель> [--depth N]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import kb_store


def _mark(obj) -> str:
    lay = obj.get("layer", "testimony")
    return "проверено" if lay == "grounded" else "слухи"


def think(goal: str, max_depth: int = 4):
    """Обратный вывод от цели. Возвращает список чанков-строк."""
    kb = kb_store.load_kb()
    nid, near = None, []
    if goal in kb["nodes"]:
        nid = goal
    else:
        cand = sorted((n for n in kb["nodes"] if goal in n),
                      key=lambda n: -kb["nodes"][n]["freq"])
        if cand:
            nid, near = cand[0], cand[1:4]

    chunks, visited = [], set()
    if nid is None:
        return [f"[0] ОБРЫВ на старте: узла «{goal}» в KB нет"]
    if near:
        chunks.append(f"[·] уточнил цель: «{goal}» → {nid} (рядом были: {', '.join(near)})")

    estab = {}                                       # to -> [интервенции]
    reqs = {}                                        # intervention -> [предусловия-состояния]
    uses = {}                                        # intervention -> [инструменты-entity]
    acts = {}                                        # intervention -> [patient-entity]
    deps = {}                                        # from -> [to]
    cons = {}                                        # узел -> [(kind, другой)]
    for e in kb["edges"].values():
        if e["kind"] == "establishes":
            estab.setdefault(e["to"], []).append(e)
        elif e["kind"] == "requires":
            reqs.setdefault(e["from"], []).append(e)
        elif e["kind"] == "uses":
            uses.setdefault(e["from"], []).append(e)
        elif e["kind"] == "acts_on":
            acts.setdefault(e["from"], []).append(e)
        elif e["kind"] in ("dependency", "depends_on"):     # v1/v2 словари
            deps.setdefault(e["from"], []).append(e)
        elif e["kind"] in ("mutex", "atmost"):
            cons.setdefault(e["from"], []).append((e["kind"], e["to"]))
            cons.setdefault(e["to"], []).append((e["kind"], e["from"]))

    def want(x: str, depth: int):
        pad = "  " * depth
        i = len(chunks)
        if x in visited:
            chunks.append(f"[{i}] {pad}{x}: уже в плане — не хожу по кругу")
            return
        visited.add(x)
        node = kb["nodes"].get(x)
        if node is None:
            chunks.append(f"[{i}] {pad}ОБРЫВ: {x} упомянут в ребре, но узла нет")
            return
        for kind, other in cons.get(x, [])[:2]:
            chunks.append(f"[{len(chunks)}] {pad}⚠ помню: {x} {kind} {other} — несовместимы ({_mark(node)})")
        if depth >= max_depth:
            chunks.append(f"[{len(chunks)}] {pad}{x}: глубина вышла — дальше не думаю")
            return
        acted = False
        for e in sorted(estab.get(x, []), key=lambda e: -e["freq"])[:2]:
            acted = True
            iv = e["from"]
            chunks.append(f"[{len(chunks)}] {pad}чтобы получить {x} — есть действие {iv} "
                          f"({_mark(e)}, freq={e['freq']})")
            for a in sorted(acts.get(iv, []), key=lambda e: -e["freq"])[:2]:
                chunks.append(f"[{len(chunks)}] {pad}  {iv} совершается НАД {a['to']} ({_mark(a)})")
            for u in sorted(uses.get(iv, []), key=lambda e: -e["freq"])[:3]:
                chunks.append(f"[{len(chunks)}] {pad}  {iv} использует инструмент {u['to']} "
                              f"({_mark(u)}; проба: уронить {u['to']} и смотреть)")
            for r in sorted(reqs.get(iv, []), key=lambda e: -e["freq"])[:3]:
                chunks.append(f"[{len(chunks)}] {pad}  {iv} требует состояния {r['to']} → хочу {r['to']}")
                want(r["to"], depth + 2)
        for e in sorted(deps.get(x, []), key=lambda e: -e["freq"])[:2]:
            acted = True
            chunks.append(f"[{len(chunks)}] {pad}{x} зависит от {e['to']} ({_mark(e)}) → сначала {e['to']}")
            want(e["to"], depth + 1)
        if not acted:
            chunks.append(f"[{len(chunks)}] {pad}ОБРЫВ мысли на {x}: в KB нет ни действия, "
                          f"ни зависимости — {_mark(node)} без пути")

    chunks.append(f"[{len(chunks)}] ЦЕЛЬ: {nid} — думаю по графу (всё testimony, пока проб не было)")
    want(nid, 1)
    return chunks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    depth = 4
    for a in sys.argv[1:]:
        if a.startswith("--depth"):
            depth = int(a.split("=")[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
    if not args:
        print(__doc__)
        return
    for c in think(args[0], max_depth=depth):
        print(c)


if __name__ == "__main__":
    main()
