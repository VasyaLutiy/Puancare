"""Тест соответствия мира протоколу + слепой прогон организма.

Использование (веткой, без участия сессии-организма):
    python3 conformance.py <модуль>
где <модуль> — python-модуль, экспортирующий GLUE (dict по WORLD_PROTOCOL.md)
и EXAMS (список экзаменов). Пример: python3 conformance.py exp9

Проверяет протокол, детерминизм, требования R1-R7 (что автоматизируемо),
затем запускает стандартный прогон организма и печатает метрики.
Содержимое мира при этом никому пересказывать не нужно — held-out цел.
"""

import importlib
import random
import sys
from collections import Counter

from organism import Organism


def fail(msg):
    print(f"  ПРОВАЛ: {msg}")
    sys.exit(1)


def check_protocol(glue):
    w = glue["factory"](0)
    obs = w.observe()
    assert isinstance(obs, dict) and obs, "observe(): пустой или не dict"
    for coll, items in obs.items():
        assert isinstance(items, dict), f"коллекция {coll}: не dict"
        for name, attrs in items.items():
            assert isinstance(attrs, dict), f"{coll}/{name}: атрибуты не dict"
    space = w.action_space()
    assert space, "action_space(): пусто"
    for a, t, kw in space:
        assert isinstance(a, str) and isinstance(kw, dict), \
            f"кривой элемент action_space: {(a, t, kw)}"
    a, t, kw = space[0]
    tr = w.step(a, t, **kw)
    for key in ("obs", "action", "target", "args", "result",
                "effects", "obs_after"):
        assert key in tr, f"transition без ключа {key}"
    assert isinstance(tr["result"], str), "result не строка"
    print("  протокол: ok")


def check_determinism(glue):
    def rollout():
        w = glue["factory"](0)
        rng = random.Random(7)
        log = []
        for _ in range(60):
            sp = w.action_space()
            if not sp:
                break
            a, t, kw = rng.choice(sp)
            tr = w.step(a, t, **kw)
            log.append((a, t, tr["result"], tuple(map(tuple, tr["effects"]))))
        return log
    assert rollout() == rollout(), "два прогона с одним seed разошлись"
    print("  детерминизм: ok")


def check_glue(glue, exams):
    w = glue["factory"](0)
    obs = w.observe()
    for a, t, kw in w.action_space():
        ctx = glue["ctx_fn"](obs, a, t, kw)
        assert isinstance(ctx, frozenset), f"ctx_fn({a}) не frozenset"
    ents = glue["entities_fn"](obs)
    assert isinstance(ents, dict), "entities_fn не dict"
    for label, probe, ask, undo in exams:
        acts = {a for a, _, _ in w.action_space()}
        assert probe in acts and ask in acts, \
            f"экзамен {label}: зонд/вопрос вне action_space"
        assert probe != ask, f"экзамен {label}: зонд совпадает с вопросом"
    print("  клей и экзамены: ok")


def check_variability(glue):
    # R2: скрытая истина должна давать >= 2 значения на мир (по truth_fn)
    for ep in range(3):
        w = glue["factory"](ep)
        truths = set()
        for coll in w.observe().values():
            for name in coll:
                tv = glue["truth_fn"](w, name)
                if tv is not None:
                    truths.add(tv)
        assert len(truths) >= 2, f"мир ep={ep}: скрытая истина одноцветна"
    print("  вариативность скрытого: ok")


def run_organism(glue, exams, budget):
    org = Organism(True, glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue["entities_fn"], seed=0)
    for ep in range(10):
        w = glue["factory"](ep)
        for _ in range(budget // 10):
            org.act(w, ep)
    org.sleep()
    return org


def exam_scores(org, glue, exams):
    out = []
    for label, probe, ask, undo in exams:
        w = glue["factory"](999)
        score = total = 0
        first_coll = None
        obs0 = w.observe()
        for coll, items in obs0.items():
            names = [n for n in items
                     if any(a == ask and t == n
                            for a, t, _ in w.action_space())]
            if names:
                first_coll = names
                break
        for name in first_coll or []:
            obs = w.observe()
            ctx = glue["ctx_fn"](obs, ask, name, {})
            res = w.step(probe, name)["result"]
            if undo and res == "ok":
                w.step(probe, name)
            inferred = org.infer(probe, res)
            if inferred:
                ctx = ctx | {inferred}
            if org.R:
                ctx = ctx | {("linked_live", 0)}
            r = org.mem.predict(ctx, ask)
            pred = r["outcome"][0] if r else None
            score += pred == w.step(ask, name)["result"]
            total += 1
        out.append((label, score, total))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1].endswith((".yaml", ".yml")):
        from worldkit import load_spec, make_glue
        glue = make_glue(load_spec(sys.argv[1]))
        exams = glue["exams"]
    else:
        mod = importlib.import_module(sys.argv[1])
        glue, exams = mod.GLUE, mod.EXAMS

    print("Проверки протокола:")
    check_protocol(glue)
    check_determinism(glue)
    check_glue(glue, exams)
    check_variability(glue)

    print("Слепой прогон организма:")
    for budget in (300, 2000):
        org = run_organism(glue, exams, budget)
        n_cls = sum(len(g["clusters"]) for g in org.groups)
        res = sum(1 for (ctx, a, o) in org.mem.episodes
                  if (lambda p: p is None or p["outcome"] != o)(
                      org.mem.predict(ctx, a)))
        ex = "  ".join(f"{l}: {s}/{t}" for l, s, t in
                       exam_scores(org, glue, exams))
        print(f"  бюджет {budget:4d}: правил {len(org.mem.rules)}, "
              f"переменных {len(org.groups)} (классов {n_cls}), "
              f"необъяснённых {res}, экзамены: {ex}")
    print("\nЕсли экзамены близки к максимуму и необъяснённое мало и не")
    print("растёт с бюджетом — мир выучиваем. Если нет — см. R1-R7.")
