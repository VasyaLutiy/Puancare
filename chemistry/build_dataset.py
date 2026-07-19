"""build_dataset — из пулов станка (stamp_qa) в SFT-датасет (chat jsonl).

Правила сборки:
  поплыло — вон             (истина зашумлена тиком между пробами);
  мир «этанол» — вон        (вердикт дедупа b01: mol08 ≡ mol01 «этаноль»,
                             одна форма в двух словарях — не два семпла);
  содержательный/константный -> ответ движка;
  неопределим                -> «не знаю» (калиброванная разметка движка).

Формат семпла — {"messages": [system, user, assistant]} (TRL SFT).
Промпт — общий с экзаменатором (qa_prompt): обучение и проверка говорят
одним языком.

    python3 chemistry/build_dataset.py chemistry/qa.pool.jsonl chemistry/qa.test.jsonl \
        [--out-dir chemistry/dataset] [--seed 7]
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_prompt import DONT_KNOW, SYSTEM, build_body  # noqa: E402

DUPLICATES = {"этанол"}          # вердикт дедупа b01, см. STAKE-chem-b01.md


def convert(pool_path):
    kept, cut = [], {"поплыло": 0, "дубликат": 0, "вне_статики": 0}
    for line in open(pool_path, encoding="utf-8"):
        q = json.loads(line)
        if q["мир"] in DUPLICATES:
            cut["дубликат"] += 1
            continue
        if q["поплыло"]:
            cut["поплыло"] += 1
            continue
        if q["сорт"] is None:
            cut["вне_статики"] += 1
            continue
        answer = DONT_KNOW if q["сорт"] == "неопределим" else q["ответ"]
        kept.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_body(q)},
                {"role": "assistant", "content": answer},
            ],
            "мир": q["мир"], "сорт": q["сорт"], "момент": q["момент"],
            "id": q["id"],
        })
    return kept, cut


def stats(rows):
    by = {}
    for r in rows:
        by[r["сорт"]] = by.get(r["сорт"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(by.items()))


def main():
    args = sys.argv[1:]
    train_pool, test_pool = args[0], args[1]

    def arg(flag, default):
        return args[args.index(flag) + 1] if flag in args else default

    out_dir = arg("--out-dir", "chemistry/dataset")
    seed = int(arg("--seed", "7"))
    os.makedirs(out_dir, exist_ok=True)

    for name, path, shuffle in (("train", train_pool, True),
                                ("test", test_pool, False)):
        rows, cut = convert(path)
        if shuffle:
            random.Random(seed).shuffle(rows)
        out = os.path.join(out_dir, f"{name}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} семплов ({stats(rows)}); "
              f"срезано: {', '.join(f'{k} {v}' for k, v in cut.items())}; "
              f"-> {out}")


if __name__ == "__main__":
    main()
