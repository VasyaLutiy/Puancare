"""trigger — мера знания по ОТКЛИКУ, а не по нутру.

Одну выученную форму дёргаем разными триггерами: вопрос через Δ шагов
после зонда. Форму не учили на Δ>1 — если её кубик верен, она проявит
верный ответ на невиданный триггер; где триггер далёк — честно
воздержится (belief подтаял ниже ½), а не соврёт.

Профиль (верно / воздержался / соврал) по Δ = «на сколько триггеров форма
осаждает верное» = сколько в ней знания, измеренное откликом.
"""

import sys

import runworld
from organism2 import Organism
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def infer_delayed(org, probe, res, dt):
    """Метка скрытого через Δ шагов после зонда: belief от зонда,
    подтаявший на dt по выученному кубику; MAP если уверенность > ½."""
    for i, ax in enumerate(org.axes):
        if probe not in ax.actions:
            continue
        bel = ax.see(list(ax.pi), probe, res)
        bel = ax.evolve(bel, dt)
        m = max(bel)
        if m > 0.5:
            return (f"class{i}", f"c{bel.index(m)}")
    return None


def battery(org, glue, probe, ask, dt):
    w = glue["exam_world"]()
    correct = wrong = abstain = 0
    for name in sorted(w.objects):
        obs = w.observe()
        ctx0 = glue["ctx_fn"](obs, ask, name, {})
        res = w.step(probe, name)["result"]         # чтение t0, тик -> t1
        for _ in range(max(0, dt - 1)):             # догнать до t0+Δ
            w.step(probe, name)                     # зонд безвреден
        label = infer_delayed(org, probe, res, dt)
        ctx = ctx0 | ({label} if label else set())
        if org.R:
            ctx = ctx | {("linked_live", 0)}
        r = org.mem.answer(ctx, ask)
        actual = w.step(ask, name)["result"]        # чтение t0+Δ
        if r is None:
            abstain += 1
        elif r["outcome"][0] == actual:
            correct += 1
        else:
            wrong += 1
    return correct, abstain, wrong


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["batteries", "bird"]
    for world in worlds:
        glue = make_glue(load_spec(f"worlds/{world}.yaml"))
        org = runworld.live(glue, True, 2000)
        label, probe, ask, undo = glue["exams"][0]
        n = len(glue["exam_world"]().objects)
        haz = max((h for ax in org.axes for h in ax.haz), default=0.0)
        print(f"\n{world}: форма k={max((a.k for a in org.axes), default=1)}, "
              f"кубик≈{haz:.0%}/шаг, объектов {n}")
        print(f"  Δ шагов | верно | воздерж | соврал")
        for dt in (1, 2, 3, 5, 8, 13):
            c, a, w_ = battery(org, glue, probe, ask, dt)
            print(f"    {dt:>5} | {c:>5} | {a:>7} | {w_:>6}")
