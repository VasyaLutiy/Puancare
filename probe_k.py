"""probe_k — быстрый зонд идентифицируемости k, В ОБХОД structure-поиска.

Собираем данные как best_axis (случайные действия, curious=False), но БЕЗ
единого sleep()/структурного перебора — потом зовём fit_axis напрямую при
k=2,3,4. ~1с на мир вместо ~40. Вопрос: на трёх идентичных по истине мирах
(baloon1≡bact≡baloon2 — лестница вниз) даёт ли k_max=3 УСТОЙЧИВУЮ третью
моду через сиды и бюджеты, или k плавает (тогда чинить идентифицируемость).

Это диагностика: тут разрешено фитить при принудительном k. Организм так
не делает — он голосует за k, и вот это голосование мы и проверяем.
"""

import sys
import random
from collections import defaultdict

from organism2 import fit_axis
from koopman import spectrum
from worldkit import load_spec, make_glue


def collect(glue, budget, worlds=10, seed=0):
    """История obj->a->[(result, global_step)] случайными действиями,
    точь-в-точь как Organism.act записывает, но без sleep/структуры."""
    rng = random.Random(seed)
    obj_acts = set(glue["object_actions"])
    hist = defaultdict(lambda: defaultdict(list))
    step = 0
    per = budget // worlds
    for ep in range(worlds):
        w = glue["factory"](ep)
        for _ in range(per):
            a, t, kw = rng.choice(w.action_space())
            tr = w.step(a, t, **kw)
            if a in obj_acts:
                hist[(ep, t)][a].append((tr["result"], step))
            step += 1
    return hist, obj_acts


def timelines(hist, actions):
    tls = {}
    for obj, by_a in hist.items():
        tl = [(t, a, r) for a in actions for (r, t) in by_a.get(a, ())]
        if tl:
            tl.sort()
            tls[obj] = tl
    return tls


def evstr(ev):
    return "{" + " ".join(f"{z.real:.3f}" for z in ev) + "}"


MARGIN = 100.0   # бит: добавить состояние, только если прирост score выше


def select_margin(fits):
    """Правило отбора k с ЗАПАСОМ: расти, пока добавление состояния даёт
    прирост score > MARGIN. Фиктивное состояние даёт знакопеременную
    мелочь < MARGIN и не проходит; реальное даёт устойчивый большой скачок."""
    ks = sorted(fits)
    k = ks[0]
    for kk in ks[1:]:
        if fits[kk].score - fits[k].score > MARGIN:
            k = kk
        else:
            break
    return k


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["last/baloon1", "last/bact", "last/baloon2"]
    budgets = [1000, 2000]
    seeds = [0, 1, 2]
    KS = [2, 3, 4]

    picks = defaultdict(list)
    for w in worlds:
        glue = make_glue(load_spec(f"worlds/{w}.yaml"))
        print(f"\n=== {w} ===  (истина: лестница, 3 ступени → ждём k=3)")
        print(f"{'бюдж':>5} {'сид':>3} | {'argmax':>6} {'ЗАПАС':>6} | "
              f"{'Δ2→3':>7} {'Δ3→4':>7} | k=3 спектр")
        for b in budgets:
            for s in seeds:
                hist, obj_acts = collect(glue, b, seed=s)
                acts = tuple(sorted(obj_acts))
                tls = timelines(hist, acts)
                fits = {k: fit_axis(acts, tls, k) for k in KS}
                argmax = max(fits, key=lambda k: fits[k].score)
                pick = select_margin(fits)
                picks[w].append(pick)
                d23 = fits[3].score - fits[2].score
                d34 = fits[4].score - fits[3].score
                print(f"{b:>5} {s:>3} | {argmax:>6} {pick:>6} | "
                      f"{d23:>7.1f} {d34:>7.1f} | {evstr(spectrum(fits[3]))}")

    print(f"\n=== выбор k правилом ЗАПАС (порог {MARGIN:.0f} бит) ===")
    for w in worlds:
        ps = picks[w]
        ok = "✓ стабильно k=3" if set(ps) == {3} else f"плавает: {ps}"
        print(f"  {w:14s}: {ps}  {ok}")
