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


def _fit_ks(acts, tls, ks):
    return {k: fit_axis(acts, tls, k) for k in ks}


def select_k(acts, tls, ks=(2, 3, 4)):
    """Отбор k БЕЗ рукодельного порога (арка линзы: margin=100 удалён).

    Болезнь константы (досье конвейера, batch-00..04): настоящие приросты
    score при верном k+1 — 39..137 бит и РАСТУТ с данными, а порог стоял
    на месте → один и тот же мир по-разному глубок на 300 и 600 (b04-m04:
    ввоз@300, ложный ОТКАЗ@600 с d=4.66), волк f0006 (k-флип линзы одной
    карточки раздул радиус полки до 3.26 и выключил детектор новизны).

    Правило из данных: объекты делятся на две независимые половины; шаг
    k-1→k принимается, если (а) ОБЕ половины за (прирост > 0 у каждой) и
    (б) прирост на полных данных больше межполовинного разброса прироста
    (сигнал > выборочный шум). Фиктивное состояние проваливает (а) —
    штраф Оккама уже в score даёт ему знакопеременную мелочь; пограничное
    настоящее состояние принимается, как только данные тянут, и решение
    перестаёт зависеть от бюджета скачком. Возвращает (k, fits_full)."""
    # рассадка половин — НЕМАЯ: по времени первого появления объекта в
    # дневнике (уникально — за шаг трогается один объект), не по имени;
    # sorted(key=repr) ронял немоту: обращённый алфавит тасовал половины
    # и k-флипал select_k при тех же самых данных
    keys = sorted(tls, key=lambda o: tls[o][0][0])
    ha = {o: tls[o] for i, o in enumerate(keys) if i % 2 == 0}
    hb = {o: tls[o] for i, o in enumerate(keys) if i % 2 == 1}
    full = _fit_ks(acts, tls, ks)
    fa, fb = _fit_ks(acts, ha, ks), _fit_ks(acts, hb, ks)
    kk = sorted(ks)
    k = kk[0]
    for k2 in kk[1:]:
        if any(f[x] is None for f in (full, fa, fb) for x in (k, k2)):
            break
        dA = fa[k2].score - fa[k].score
        dB = fb[k2].score - fb[k].score
        dF = full[k2].score - full[k].score
        if min(dA, dB) > 0 and dF > abs(dA - dB):
            k = k2
        else:
            break
    return k, full, fa, fb


def quick_signature(glue, budget, ks=(2, 3, 4), acts=None):
    """Подпись нового мира (быстро): МУЛЬТИ-K. Прежняя одиночная подпись
    прыгала разрывно при пограничном выборе k (волк f0006, ложный отказ
    b04-m04, раскол семьи лестниц A/B) — любое правило выбора одного k
    иногда колет семьи. Мульти-подпись несёт срезы при ВСЕХ k∈ks; сравнение
    (koopman._pair_d) идёт только при совпадающем k — паддинг-взрыв
    исчезает по построению.

    Поля подписи: верхний уровень = срез при устойчивом k (select_k) —
    legacy-совместимо; "ks" = все срезы; "noise" = внутримировая болтанка
    компонент между половинами данных (шумовой пол для calibrate).

    acts — ЛИНЗА: ограничить подпись поднабором действий (улов v2)."""
    org = _collect(glue, budget)
    acts = tuple(sorted(acts if acts is not None else org.object_actions))
    tls = org._axis_timelines(acts)
    k, fits, fa, fb = select_k(acts, tls, ks)
    sig = dict(koopman.signature(fits[k]))
    sig["ks"] = {kk: koopman.signature(fits[kk])
                 for kk in ks if fits.get(kk) is not None}
    noise = {}
    for c in koopman.COMPS:
        ds = []
        for kk in ks:
            trio = [f[kk] for f in (fits, fa, fb) if f.get(kk) is not None]
            sgs = [koopman.signature(x) for x in trio]
            # болтанка = МАКС разброс компоненты между фитами полных данных
            # и половин: артефакты (комплекс-пары EM) гуляют между фитами
            # разного объёма, настоящий сигнал — нет
            ds.append(max((koopman._comp_d(a, b, c)
                           for i, a in enumerate(sgs) for b in sgs[i + 1:]),
                          default=0.0))
        noise[c] = sum(ds) / len(ds) if ds else 0.0
    sig["noise"] = noise
    return sig, fits[k]


def recognize_and_fit(glue, budget, repertoire, ks=(2, 3, 4)):
    """Быстрая дорога: собрать данные → подписать → матч с репертуаром →
    НАВЯЗАТЬ k формы → одна подгонка. Возвращает (организм, форма, dist)."""
    org = _collect(glue, budget)
    acts = tuple(sorted(org.object_actions))
    tls = org._axis_timelines(acts)
    k_prov, fits, fa, fb = select_k(acts, tls, ks)    # тот же устойчивый
    prov = fits[k_prov]                               # отбор, что в quick_signature
    q = dict(koopman.signature(prov))                 # мульти-подпись, как там же
    q["ks"] = {kk: koopman.signature(fits[kk])
               for kk in ks if fits.get(kk) is not None}
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
