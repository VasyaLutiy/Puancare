"""runworld — универсальный полигон: анкета мира -> организм -> отчёт.

  python3 runworld.py мир.yaml --check   проверка анкеты + трасса 20 шагов
  python3 runworld.py мир.yaml           полный прогон (организм, экзамены)

Организм и ядро те же, что во всех экспериментах; весь клей выводится
из анкеты автоматически (worldkit.make_glue).
"""

import random
import sys
from collections import Counter

from organism import Organism
from worldkit import GenericWorld, load_spec, make_glue


def check(spec, glue):
    print(f"Мир: {spec.name}")
    print(f"  объектов {spec.n_objects}, видимое {list(spec.visible)}, "
          f"скрытое {list(spec.hidden)}")
    if spec.has_entities:
        print(f"  сущностей {spec.ent_count}, состояния {spec.states}, "
              f"связь x{spec.links_per}"
              + (f" (только {spec.link_filter[0]}={spec.link_filter[1]})"
                 if spec.link_filter else ""))
    print(f"  действия с объектом: {list(spec.obj_actions)}")
    print(f"  действия с сущностью: {list(spec.ent_actions)}")
    print(f"  вмешательства: {list(spec.interventions)}")
    print(f"  экзамены: {spec.exams}")

    w = GenericWorld(spec, seed=1)
    print("\nНачальное состояние (глазами агента — скрытое не показано):")
    obs = w.observe()
    for n, o in obs["объекты"].items():
        truth = "-".join(w.objects[n][a] for a in spec.hidden)
        print(f"  {n}: {o}   [скрыто: {truth}]")
    for n, e in obs["сущности"].items():
        print(f"  {n}: {e}   [связи: {w.entities[n]['links']}]")

    print("\nТрасса 20 случайных шагов:")
    rng = random.Random(0)
    for i in range(20):
        a, t, kw = rng.choice(w.action_space())
        tr = w.step(a, t, **kw)
        eff = f"  эффекты {tr['effects']}" if tr["effects"] else ""
        args = f" {kw}" if kw else ""
        print(f"  {i:2d}. {a}({t or ''}){args} -> {tr['result']}{eff}")
    print("\nАнкета исполнима.")


def live(glue, curious, budget, worlds=10, seed=0):
    org = Organism(curious, glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue["entities_fn"], seed=seed)
    for ep in range(worlds):
        w = glue["factory"](ep)
        for _ in range(budget // worlds):
            org.act(w, ep)
    org.sleep()
    return org


def exam(org, glue, probe, ask, undo=False):
    w = glue["exam_world"]()
    score = total = 0
    for name in sorted(w.objects):
        obs = w.observe()
        ctx = glue["ctx_fn"](obs, ask, name, {})
        res = w.step(probe, name)["result"]
        if undo:
            w.step(probe, name)   # зонд-переключатель откатывается повтором
        inferred = org.infer(probe, res)
        if inferred:
            ctx = ctx | {inferred}
        if org.R:
            ctx = ctx | {("linked_live", 0)}
        r = org.mem.answer(ctx, ask)
        pred = r["outcome"][0] if r else None
        score += pred == w.step(ask, name)["result"]
        total += 1
    return score, total


def residue(org):
    return sum(1 for (ctx, a, o) in org.mem.episodes
               if (lambda p: p is None or p["outcome"] != o)(
                   org.mem.predict(ctx, a)))


def full_run(spec, glue):
    print("=" * 76)
    print(f"МИР: {spec.name} (из анкеты)")
    print("=" * 76)
    last = None
    for budget in (300, 2000):
        for curious in (False, True):
            org = live(glue, curious, budget)
            exams = "  ".join(
                f"{label} {exam(org, glue, p, a, u)[0]}/{spec.n_objects}"
                for label, p, a, u in glue["exams"]) or "нет экзаменов"
            ok_links = sum(glue["link_truth"](o[0], o[1], e[1])
                           for (o, e, st) in org.R)
            name = "любопытный" if curious else "случайный "
            n_cls = sum(len(g["clusters"]) for g in org.groups)
            print(f"  бюджет {budget:4d} {name}: правил "
                  f"{len(org.mem.rules):2d}, классов {n_cls} в "
                  f"{len(org.groups)} перем., связей {ok_links}/{len(org.R)},"
                  f" необъясн. {residue(org):2d}, экз: {exams}")
            last = org

    print("\nПеременные (любопытный, 2000):")
    for g in last.groups:
        print(f"  {g['attr']} <- симптомы {sorted(g['actions'])}")
        for k, (ms, objs) in enumerate(g["clusters"]):
            truths = Counter(r["truth"] for r in last.records
                             if r["obj"] in set(objs) and r["truth"])
            sig = ", ".join(f"{a}={v}" for a, v in sorted(ms.items()))
            print(f"    c{k} [{sig}]  истина: {dict(truths)}")

    print("\nСмыслы (правила с class/linked_live):")
    for r in sorted(last.mem.rules,
                    key=lambda r: (r["action"], -r["support"])):
        if any(att.startswith("class") or att == "linked_live"
               for att, _ in r["conds"]):
            conds = " & ".join(f"{a}={v}" for a, v in sorted(r["conds"]))
            res, effs = r["outcome"]
            eff = f" + {list(effs)}" if effs else ""
            print(f"  {r['action']}: ЕСЛИ {conds} ТО {res}{eff}  "
                  f"[объясняет {r['support']}, искл. {r['exceptions']}]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if not args:
        sys.exit("использование: python3 runworld.py мир.yaml [--check]")
    path = args[0]
    try:
        spec = load_spec(path)
    except (ValueError, OSError) as e:
        sys.exit(f"АНКЕТА НЕ ПРИНЯТА: {e}")
    glue = make_glue(spec)
    if "--check" in args:
        check(spec, glue)
    else:
        full_run(spec, glue)
