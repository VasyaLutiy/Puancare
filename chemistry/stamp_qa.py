"""stamp_qa — штамповка Q/A из мира: порции × пары действий × моменты.

Хвост цепочки датасета: замороженный мир (yaml) → вопросы с МАШИННОЙ истиной.
Ответ вычисляет движок (worldkit), не разметчик и не LLM: вопрос —
«сделал A, получил X; что даст B на той же порции сейчас?», ответ —
результат честного step(B) в том же состоянии мира. Никакой генерации
миров: только прожитые, только штамповка.

Три сигнала, которых нет в LLM-генерённых парах:
  машинная истина — ответ исполнен, не придуман;
  время           — тот же вопрос в молодом и старом мире отвечается разно
                    (старение = чистые тики динамики, без проб);
  определимость   — движок перечисляет скрытые состояния, совместимые с
                    уликами (видимое + отклик зонда, статически, без тиков);
                    если ответ по ним расходится — «неопределим»: готовая
                    разметка для калиброванного «не знаю».

Тонкость: между зондом и вопросом мир тикает; если скрытое порции
поплыло между пробами, семпл помечен «поплыло» — учтённый label noise,
не замятый.

    python3 chemistry/stamp_qa.py <мир.yaml> [ещё.yaml ...]
        [--moments 0,5,10,20] [--seeds 1,2,3] [--out qa.jsonl] [--show N]
"""

import itertools
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from worldkit import load_spec, GenericWorld  # noqa: E402


# ---------------------------------------------------- статический вывод

def _eval_static(spec, action, obj):
    """Результат действия на объекте с данными атрибутами, без мира и тиков.

    Повторяет _apply: первое подошедшее правило. Возвращает (результат,
    эффекты) или None, если тело правила не про атрибуты объекта
    (сущности, связи) — такой мир статически не разбираем.
    """
    for rule in spec.obj_actions[action]:
        ok = True
        for cl in rule["body"]:
            if len(cl) == 3 and cl[1] == "?t" and cl[0] in spec.attrs:
                if obj.get(cl[0]) != cl[2]:
                    ok = False
                    break
            else:
                return None
        if ok:
            return rule["result"], rule["effects"]
    return "ничего_не_произошло", []


def _determinable(spec, visible, probe, probe_result, question):
    """Определим ли ответ на question из улик (видимое + отклик зонда).

    Перебираем все комбинации скрытых значений; оставляем совместимые с
    откликом зонда; смотрим, один ли ответ даёт question. Возвращает
    (определим: bool | None, ответы: sorted list).
    """
    names = list(spec.hidden)
    answers = set()
    for combo in itertools.product(*(spec.hidden[a] for a in names)):
        obj = dict(visible)
        obj.update(zip(names, combo))
        got = _eval_static(spec, probe, obj)
        if got is None:
            return None, []
        if got[0] != probe_result:
            continue
        if "исчезает" in got[1]:
            answers.add("нет_такого_объекта")
            continue
        ans = _eval_static(spec, question, obj)
        if ans is None:
            return None, []
        answers.add(ans[0])
    if not answers:                       # улики вне статической модели
        return None, []
    return len(answers) == 1, sorted(answers)


def _probe_needed(spec, visible, question):
    """Нужен ли зонд вообще: расходится ли ответ question по скрытым
    состояниям без всяких улик. False — вопрос-проба константна в мире
    (или при данном видимом): ответ — факт мира, а не вывод из улик."""
    names = list(spec.hidden)
    answers = set()
    for combo in itertools.product(*(spec.hidden[a] for a in names)):
        obj = dict(visible)
        obj.update(zip(names, combo))
        got = _eval_static(spec, question, obj)
        if got is None:
            return None
        answers.add(got[0])
    return len(answers) > 1


def _vocab(spec, action):
    """Словарь возможных откликов действия — для честного скоринга."""
    out = {r["result"] for r in spec.obj_actions[action]}
    out.add("ничего_не_произошло")
    return sorted(out)


# ------------------------------------------------------------- штамповка

def stamp_world(path, moments=(0, 5, 10, 20), seeds=(1,)):
    spec = load_spec(path)
    acts = list(spec.obj_actions)
    pairs = [(a, b) for a in acts for b in acts if a != b]
    qa = []
    for seed in seeds:
        for t in moments:
            base = GenericWorld(spec, seed=seed)
            for _ in range(t):
                base._tick()              # старение без проб
            portions = sorted(base.objects)
            for name in portions:
                for a, b in pairs:
                    w = GenericWorld(spec, seed=seed)
                    for _ in range(t):
                        w._tick()
                    vis = {k: w.objects[name][k] for k in spec.visible}
                    hid0 = {k: w.objects[name][k] for k in spec.hidden}
                    ra = w.step(a, name)["result"]
                    gone = name not in w.objects
                    hid1 = (None if gone else
                            {k: w.objects[name][k] for k in spec.hidden})
                    rb = ("нет_такого_объекта" if gone else
                          w.step(b, name)["result"])
                    det, dans = _determinable(spec, vis, a, ra, b)
                    need = _probe_needed(spec, vis, b)
                    sort = ("неопределим" if det is False else
                            None if det is None else
                            "содержательный" if need else "константный")
                    qa.append({
                        "id": f"{spec.name}:s{seed}:t{t}:{name}:{a}->{b}",
                        "мир": spec.name, "сид": seed, "момент": t,
                        "порция": name, "видимое": vis,
                        "зонд": a, "отклик": ra,
                        "вопрос": b, "ответ": rb,
                        "варианты": _vocab(spec, b),
                        "определим": det,
                        "сорт": sort,
                        "ответы_совместимые": dans,
                        "поплыло": gone or hid1 != hid0,
                    })
    return spec, qa


def summarize(qa):
    n = len(qa)
    det = sum(1 for q in qa if q["определим"] is True)
    undet = sum(1 for q in qa if q["определим"] is False)
    na = n - det - undet
    drift = sum(1 for q in qa if q["поплыло"])
    honest = sum(1 for q in qa
                 if q["определим"] is True
                 and q["ответ"] in q["ответы_совместимые"])
    sorts = {}
    for q in qa:
        sorts[q["сорт"]] = sorts.get(q["сорт"], 0) + 1
    s = ", ".join(f"{k or 'вне статики'} {v}" for k, v in sorted(
        sorts.items(), key=lambda kv: str(kv[0])))
    return (f"семплов {n}; {s}; поплыло {drift}; "
            f"определим и ответ совпал со статикой {honest}/{det}")


# ------------------------------------------------------------------ CLI

def _arg(flag, default):
    return (sys.argv[sys.argv.index(flag) + 1]
            if flag in sys.argv else default)


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".yaml")]
    moments = tuple(int(x) for x in _arg("--moments", "0,5,10,20").split(","))
    seeds = tuple(int(x) for x in _arg("--seeds", "1").split(","))
    out = _arg("--out", None)
    show = int(_arg("--show", "6"))

    all_qa = []
    for path in paths:
        spec, qa = stamp_world(path, moments=moments, seeds=seeds)
        all_qa.extend(qa)
        print(f"{spec.name}: {summarize(qa)}")

    print(f"\nИТОГО: {summarize(all_qa)}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for q in all_qa:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"записано: {out}")

    if not show:
        return
    step = max(1, len(all_qa) // show)
    for i, q in enumerate(all_qa[::step][:show], 1):
        vis = ", ".join(f"{k}={v}" for k, v in q["видимое"].items())
        tag = ("✓ определим" if q["определим"] is True else
               "✗ НЕопределим" if q["определим"] is False else "— вне статики")
        tag += ", поплыло" if q["поплыло"] else ""
        print(f"\nQ{i} [{q['мир']} шаг {q['момент']}] Порция «{q['порция']}» "
              f"({vis}). Проба «{q['зонд']}» дала «{q['отклик']}». "
              f"Что даст «{q['вопрос']}»?")
        print(f"    ОТВЕТ ДВИЖКА: «{q['ответ']}»   [{tag}]")


if __name__ == "__main__":
    main()
