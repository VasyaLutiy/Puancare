"""exam_llm — baseline-экзамен LLM по штампованному пулу Q/A.

Берёт пул станка (stamp_qa --out), стратифицированно сэмплирует вопросы
и задаёт их модели через OpenAI-совместимый API. Меряет раздельно:

  определимые   — точность против ответа движка (машинная истина);
  неопределимые — долю честного «не знаю» (калибровка) и долю угадываний.

Режимы:
  закрытая книга (по умолчанию) — только вопрос: видимое, зонд→отклик;
  --context                     — в промпт кладётся yaml мира (чтение-симуляция).

Семплы «поплыло» в экзамен не идут: их истина зашумлена тиком.

    python3 chemistry/exam_llm.py chemistry/qa.pool.jsonl \
        [--api http://127.0.0.1:8000/v1] [--model auto] [--n 200] \
        [--context] [--seed 7] [--jobs 8] [--out chemistry/ledger.exam.jsonl]
"""

import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_prompt import DONT_KNOW, SYSTEM, build_body  # noqa: E402


def ask(api, model, q, world_text=None, timeout=120):
    body = build_body(q, world_text)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": body}],
        "temperature": 0.0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{api}/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]


def parse_answer(text, options):
    """Последняя непустая строка, зачистка кавычек/точек, матч по словарю."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].strip('«»"\'`.!').strip().lower()
    for o in options + [DONT_KNOW]:
        if last == o.lower():
            return o
    for o in options + [DONT_KNOW]:          # запасной: вариант внутри строки
        if o.lower() in last:
            return o
    return None


def pick(pool, n, rng):
    """Стратификация: поровну определимых и неопределимых, поплыло — вон."""
    ok = [q for q in pool if not q["поплыло"] and q["определим"] is True]
    no = [q for q in pool if not q["поплыло"] and q["определим"] is False]
    rng.shuffle(ok)
    rng.shuffle(no)
    half = n // 2
    return ok[:max(half, n - len(no))] + no[:half]


def world_texts(pool):
    """Найти yaml каждого мира по имени (для --context)."""
    paths = {}
    for root in ("chemistry/worlds", "chemistry/in/batch-01"):
        for dirp, _, files in os.walk(os.path.join(_ROOT, root)):
            for f in files:
                if f.endswith(".yaml"):
                    p = os.path.join(dirp, f)
                    name = open(p, encoding="utf-8").readline()
                    if name.startswith("мир:"):
                        paths[name.split(":", 1)[1].strip()] = p
    return {w: open(p, encoding="utf-8").read() for w, p in paths.items()}


def main():
    args = sys.argv[1:]
    pool_path = args[0]

    def arg(flag, default):
        return args[args.index(flag) + 1] if flag in args else default

    api = arg("--api", "http://127.0.0.1:8000/v1")
    model = arg("--model", "auto")
    n = int(arg("--n", "200"))
    seed = int(arg("--seed", "7"))
    jobs = int(arg("--jobs", "8"))
    out = arg("--out", None)
    use_ctx = "--context" in args

    pool = [json.loads(l) for l in open(pool_path, encoding="utf-8")]
    rng = random.Random(seed)
    exam = pick(pool, n, rng)
    ctx = world_texts(pool) if use_ctx else {}
    print(f"экзамен: {len(exam)} вопросов "
          f"({sum(q['определим'] for q in exam)} определимых), "
          f"режим {'с миром в контексте' if use_ctx else 'закрытая книга'}, "
          f"модель {model}")

    def one(q):
        try:
            raw = ask(api, model, q, ctx.get(q["мир"]))
        except Exception as e:
            return dict(q, ллм=None, сырое=f"ОШИБКА: {e}")
        return dict(q, ллм=parse_answer(raw, q["варианты"]), сырое=raw)

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(one, exam))

    det = [r for r in results if r["определим"] is True]
    und = [r for r in results if r["определим"] is False]
    d_hit = sum(1 for r in det if r["ллм"] == r["ответ"])
    d_idk = sum(1 for r in det if r["ллм"] == DONT_KNOW)
    u_idk = sum(1 for r in und if r["ллм"] == DONT_KNOW)
    u_hit = sum(1 for r in und if r["ллм"] == r["ответ"])
    junk = sum(1 for r in results if r["ллм"] is None)

    print(f"\nОПРЕДЕЛИМЫЕ  ({len(det)}): точность {d_hit}/{len(det)}"
          f"{'' if not det else f' = {d_hit / len(det):.0%}'}; "
          f"лишние «не знаю» {d_idk}")
    print(f"НЕОПРЕДЕЛИМЫЕ({len(und)}): «не знаю» {u_idk}/{len(und)}"
          f"{'' if not und else f' = {u_idk / len(und):.0%}'}; "
          f"угадала движок {u_hit}")
    print(f"мусор/ошибки парсинга: {junk}")

    per = {}
    for r in results:
        per.setdefault(r["мир"], [0, 0])
        if r["определим"] is True:
            per[r["мир"]][1] += 1
            per[r["мир"]][0] += r["ллм"] == r["ответ"]
    print("\nпо мирам (определимые):")
    for w, (h, t) in sorted(per.items()):
        if t:
            print(f"  {w}: {h}/{t}")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
