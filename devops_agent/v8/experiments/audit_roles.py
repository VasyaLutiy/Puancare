"""Протокол ручной разметки ролей (введён Кириллом 6-7 июл): детерминированная
выборка рёбер uses/acts_on из ТЕКУЩЕЙ KB + цитаты-источники. Кирилл судит сам.

Критерии (объявлены ДО просмотра текстов, менять нельзя):
  TOOL     — текст говорит «действие делается ПОСРЕДСТВОМ вещи» (with/using/via);
  PATIENT  — текст говорит «действие совершается НАД вещью» (объект изменения);
  STATE    — текст подаёт вещь в состоянии/готовности (running X, installed, must exist);
  MISPARSE — ребро из текста не следует (агент, аудитория, домен, компонент...).

Выборка: sorted(edges) → каждое (len//n)-е. Воспроизводимо одной командой.
  python -m devops_agent.v8.experiments.audit_roles [n_per_kind]
"""
import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V8))
sys.path.insert(0, str(V8.parents[1]))

from book_graph import paragraphs  # noqa: E402

BOOKS = {
    "dummies": "/private/tmp/ccc/docspdf/output/eBook_DevOps_for-Dummies.md",
    "iac": "/private/tmp/ccc/docspdf/output/IaC Thales.md",
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    kb = json.loads((V8 / "kb" / "kb.json").read_text())
    texts = {b: paragraphs(Path(p)) for b, p in BOOKS.items()}

    edge2para = {}
    for rf in sorted((V8 / "kb" / "runs").glob("*.json")):
        d = json.loads(rf.read_text())
        book = "dummies" if rf.name.startswith("dummies") else "iac"
        for e in d["edges"]:
            if e["kind"] in ("uses", "acts_on"):
                edge2para.setdefault(f'{e["kind"]}|{e["from"]}|{e["to"]}', (book, e["para"]))

    for kind in ("uses", "acts_on"):
        keys = sorted(k for k, e in kb["edges"].items() if e["kind"] == kind)
        if not keys:
            print(f"— рёбер {kind} нет —")
            continue
        step = max(1, len(keys) // n)
        sample = keys[::step][:n]
        print(f"\n{'='*70}\n{kind}: всего {len(keys)}, шаг {step}, выборка {len(sample)}\n")
        for i, key in enumerate(sample):
            _, frm, to = key.split("|")
            book, para = edge2para.get(key, ("?", -1))
            p = texts.get(book, [""])[para] if para >= 0 else "(источник не найден)"
            word = to.replace("_", " ").split()[0]
            pos = p.lower().find(word[:8].lower())
            lo, hi = max(0, pos - 120), min(len(p), pos + 180)
            print(f"[{i:2d}] {frm}  --{kind}-->  {to}   ({book}[{para}])")
            print(f"     «…{p[lo:hi]}…»\n")


if __name__ == "__main__":
    main()
