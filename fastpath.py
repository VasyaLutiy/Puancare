"""fastpath — быстрая дорога + репертуар: разделить ДОРОГОЕ ОТКРЫТИЕ формы
(редко, полный organism2, ~секунды-десятки) и ДЕШЁВУЮ подгонку при известной
форме (часто, одна fit_axis при навязанном k, ~сотни мс).

    открытие(мир)     -> форма F в репертуар          [медленно, O(форм)]
    узнать_и_подогнать(мир, репертуар) -> организм     [быстро,   O(миров)]

Узнавание = быстрая подгонка при k∈{2,3,4} → спектральная подпись (koopman)
→ ближайшая форма репертуара → НАВЯЗАТЬ её k (не переголосовывать локально).
Это чинит и скорость (нет structure-поиска), и идентифицируемость k
(репертуар решает за шумный локальный голос).

Композиция поверх: карта использования донора ввозится в цель, выровненная
по РОЛИ состояний (compose.import_usemap). organism2 не тронут.
"""

import sys
import time

import runworld
import koopman
from compose import import_usemap, axis_with, role_order
from organism2 import Organism, fit_axis
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def glue_of(world):
    return make_glue(load_spec(f"worlds/{world}.yaml"))


# ----------------------------------------------------- медленная дорога

def discover(world, budget):
    """Полный organism2: открыть форму нового мира. Медленно, редко."""
    glue = glue_of(world)
    org = runworld.live(glue, False, budget)
    ax = max(org.axes, key=lambda a: a.n_obs) if org.axes else None
    return {"name": world, "org": org, "ax": ax,
            "sig": koopman.signature(ax) if ax else None,
            "k": ax.k if ax else 1}


# ----------------------------------------------------- быстрая дорога

def _collect(glue, budget, worlds=10):
    """Прожить мир, СОБРАВ records/hist, но БЕЗ дорогого sleep (открытия)."""
    org = Organism(False, glue["object_actions"], glue["ctx_fn"],
                   glue["truth_fn"], glue.get("entities_fn"))
    org.sleep = lambda: None                      # глушим открытие
    for ep in range(worlds):
        w = glue["factory"](ep)
        for _ in range(budget // worlds):
            org.act(w, ep)
    return org


def select_k(fits, margin=100.0):
    """Отбор k с ЗАПАСОМ: расти, пока добавление состояния даёт прирост
    score > margin. Фиктивное состояние даёт знакопеременную мелочь и не
    проходит — стабильный k, где argmax-score плавает (см. probe_k)."""
    ks = sorted(fits)
    k = ks[0]
    for kk in ks[1:]:
        if fits[kk].score - fits[k].score > margin:
            k = kk
        else:
            break
    return k


def quick_signature(glue, budget, ks=(2, 3, 4), margin=100.0):
    """Только подпись нового мира (быстро): собрать → подогнать k∈ks →
    отобрать k с запасом → подпись. Для узнавания/новизны, без навязывания."""
    org = _collect(glue, budget)
    acts = tuple(sorted(org.object_actions))
    tls = org._axis_timelines(acts)
    fits = {k: fit_axis(acts, tls, k) for k in ks}
    k = select_k(fits, margin)
    return koopman.signature(fits[k]), fits[k]


def recognize_and_fit(glue, budget, repertoire, ks=(2, 3, 4)):
    """Быстрая дорога: собрать данные → подписать → матч с репертуаром →
    НАВЯЗАТЬ k формы → одна подгонка. Возвращает (организм, форма, dist)."""
    org = _collect(glue, budget)
    acts = tuple(sorted(org.object_actions))
    tls = org._axis_timelines(acts)
    fits = {k: fit_axis(acts, tls, k) for k in ks}
    prov = max(fits.values(), key=lambda a: a.score)   # для подписи
    q = koopman.signature(prov)
    scored = sorted((koopman.dist(q, f["sig"]), i, f)
                    for i, f in enumerate(repertoire) if f["sig"])
    dist, _, form = scored[0]
    k = form["k"]                                       # навязываем k формы
    org.axes = [fits[k]] if k >= 2 else []
    org.mem = org._relabel(org.axes)
    org._bel = {}
    return org, form, dist, q


# ----------------------------------------------------- демонстрация

if __name__ == "__main__":
    probe, ask = "просветить", "нагрузить"

    print("=== ОТКРЫТИЕ (медленная дорога): донор ===")
    t0 = time.time()
    donor = discover("comp/donor", 1000)
    t_disc = time.time() - t0
    print(f"донор открыт за {t_disc:.1f}с: k={donor['k']}, "
          f"роли(rank->state)={role_order(donor['ax'])}, "
          f"подпись spec={donor['sig']['spec']}")
    repertoire = [donor]

    # ЦЕЛЬ живёт ТОЛЬКО зондом (target_probe) → карты использования нет;
    # экзамен спрашивает нагрузить на ПОЛНОМ мире (target).
    glue_learn = glue_of("comp/target_probe")
    glue_exam = glue_of("comp/target")
    print("\n=== ЦЕЛЬ живёт только зондом; экзамен спрашивает нагрузить ===")
    print(f"{'бюдж':>5} | {'время':>7} | {'узнан(d)':>10} | "
          f"{'ГОЛАЯ':>7} | {'+ВВОЗ СЫРОЙ':>11} | {'+ВВОЗ ВЫРАВН':>12}")
    for b in [300, 600]:
        t0 = time.time()
        org, form, d, q = recognize_and_fit(glue_learn, b, repertoire)
        dt = (time.time() - t0) * 1000
        bare = runworld.exam(org, glue_exam, probe, ask)
        org.mem = import_usemap(org, form["org"], ask, False)[0]     # сырой
        raw = runworld.exam(org, glue_exam, probe, ask)
        org2, form2, d2, _ = recognize_and_fit(glue_learn, b, repertoire)
        org2.mem = import_usemap(org2, form2["org"], ask, True)[0]    # по роли
        al = runworld.exam(org2, glue_exam, probe, ask)
        print(f"{b:>5} | {dt:>5.0f}мс | {d:>10.2f} | "
              f"{bare[0]:>3}/{bare[1]:<3} | {raw[0]:>7}/{raw[1]:<3} | "
              f"{al[0]:>8}/{al[1]:<3}")
