"""transfer — первый тест переноса: помогает ли форма из прошлой жизни
выучить новый мир БЫСТРЕЕ, чем с нуля.

Механизм минимальный и без флип-риска: узнали новый мир как форму F из
библиотеки → даём структуре с k=F.k приорную СКИДКУ в битах (не пересаживаем
эмиссии — EM находит их сам). Судья покупает ту онтологию на меньших данных.

Метрика — СКОРОСТЬ: на каком бюджете впервые выучен верный k, голый vs
засеянный. (Экзамен голый и так сдаёт рано — там мерить нечего.)
"""

import sys

import runworld
import sig
import recognize
from organism2 import Organism
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def live_autonomous(glue, lib, budget=2000, orient_frac=0.3, worlds=10):
    """Автономная петля: ориентация вслепую → узнать себя по библиотеке
    lib={имя: подпись} → seed_k и сила приора ИЗ узнавания (не руками) →
    дожить с засевом. Возвращает (организм, инфо-о-засеве)."""
    org = Organism(False, glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue.get("entities_fn"))
    per = budget // worlds
    n_orient = max(1, int(worlds * orient_frac))
    for ep in range(n_orient):                       # фаза 1: вслепую
        w = glue["factory"](ep)
        for _ in range(per):
            org.act(w, ep)
    info = "оси нет — засев не задан"
    if org.axes:
        q = sig.signature(max(org.axes, key=lambda a: a.n_obs))
        # узнавание ДЛЯ ЗАСЕВА — слепое к k (k бутстрапим, матчить по нему
        # циклично); по нему же и радиус новизны
        names = list(lib)
        ranked = sorted((sig.dist(q, lib[n], blind_k=True), n) for n in names)
        radius = max(min(sig.dist(lib[a], lib[b], blind_k=True)
                         for b in names if b != a) for a in names)
        d1, near = ranked[0]
        if d1 <= radius and near in lib:
            close = max(0.0, 1.0 - d1 / radius)
            org._seed_k = lib[near]["k"]
            org._seed_bonus = 6.0 * close
            info = (f"узнан как {near} (d={d1:.2f}, радиус={radius:.2f}) → "
                    f"засев k={org._seed_k}, сила={org._seed_bonus:.1f}")
        else:
            info = f"новая форма (d={d1:.2f}>{radius:.2f}) → без засева"
    for ep in range(n_orient, worlds):               # фаза 2: с засевом
        w = glue["factory"](ep)
        for _ in range(per):
            org.act(w, ep)
    org.sleep()
    return org, info


def live_seeded(glue, curious, budget, seed_k=None, bonus=0.0, worlds=10):
    org = Organism(curious, glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue.get("entities_fn"))
    org._seed_k = seed_k
    org._seed_bonus = bonus
    for ep in range(worlds):
        w = glue["factory"](ep)
        for _ in range(budget // worlds):
            org.act(w, ep)
    org.sleep()
    return org


def learned_k(org):
    return max((ax.k for ax in org.axes), default=1)


def exam_score(org, glue):
    label, probe, ask, undo = glue["exams"][0]
    s = runworld.exam(org, glue, probe, ask, undo)
    return f"{s[0]}/{s[1]}"


if __name__ == "__main__" and "--auto" in sys.argv:
    # автономная петля: библиотека + новый мир, k и сила приора сами
    worlds_q = [a for a in sys.argv[1:] if not a.startswith("--")]
    world = worlds_q[0] if worlds_q else "last/baloon2"
    print("=== библиотека ===")
    lib = recognize.build_sigs(["batteries", "car", "bird"])
    glue = make_glue(load_spec(f"worlds/{world}.yaml"))
    print(f"\n=== автономно на {world} ===")
    for b in [600, 1000, 2000]:
        naked = live_seeded(glue, False, b, seed_k=None)
        auto, info = live_autonomous(glue, lib, budget=b)
        print(f"бюджет {b:>4}: ГОЛЫЙ k={learned_k(naked)} {exam_score(naked,glue)}"
              f"  |  АВТО k={learned_k(auto)} {exam_score(auto,glue)}   [{info}]")

elif __name__ == "__main__":
    world = sys.argv[1] if len(sys.argv) > 1 else "last/baloon2"
    seed_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    glue = make_glue(load_spec(f"worlds/{world}.yaml"))
    budgets = [300, 600, 1000, 2000]

    print(f"мир {world}, истина = лестница (k=3). Засев k={seed_k} от птицы.")
    print(f"{'бюджет':>7} | {'ГОЛЫЙ k / экз':>16} | "
          f"{'ЗАСЕЯН+6 k / экз':>18} | {'ЗАСЕЯН+12 k / экз':>18}")
    for b in budgets:
        naked = live_seeded(glue, False, b, seed_k=None)
        s6 = live_seeded(glue, False, b, seed_k=seed_k, bonus=6.0)
        s12 = live_seeded(glue, False, b, seed_k=seed_k, bonus=12.0)
        print(f"{b:>7} | {f'k={learned_k(naked)} {exam_score(naked,glue)}':>16} | "
              f"{f'k={learned_k(s6)} {exam_score(s6,glue)}':>18} | "
              f"{f'k={learned_k(s12)} {exam_score(s12,glue)}':>18}")
