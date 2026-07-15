"""exam_heldout2 — held-out экзамен композиции С ПОРОГОМ НОВИЗНЫ.

Исправление после провала comp4: v1 (exam_heldout.py) навязывал цели форму
донора принудительно — половина логики узнавания. Полная логика живёт в
ОРИГИНАЛЬНОМ repertoire.py (замечание Кирилла: «я вообще думал что ты
будешь использовать оригинальный repertoire.py») — она и используется,
импортом, не пересказом: библиотека форм = REPERTOIRE (4 формы) + донор,
метрика калибруется по библиотеке (koopman.calibrate), порог знакомого =
радиус библиотеки (repertoire.radius). Организм получает право отказа:

  d(цель, донор) <= радиус  → узнан как донор → ввоз карты, критерий >=7/8
  d_min <= радиус, но не донор → «узнан как X, карты у X нет» → без ввоза
  d_min > радиус            → «НОВАЯ ФОРМА» → ОТКАЗ ОТ ВВОЗА

Отказ — не провал раннера: правильность отказа решает автор мира (он
знает, была ли цель формой донора). Уверенно-неверный ввоз (comp4 v1:
4/7 на б.600) эпистемически хуже честного отказа.

Протокол объявлен до прогона, файл после результатов не редактируется.
Прогоняются все comp1..comp4 (перепроверка v2 на сданных — обязательна:
порог не должен отобрать уже честно сданное).

Запуск: python3 exam_heldout2.py <донор> <цель-зонд> <цель>
"""

import sys
import time

import runworld
import koopman
from compose import import_usemap, axis_with, role_gap, role_order
from fastpath import recognize_and_fit, quick_signature, glue_of
from repertoire import discover_lib, radius
from organism2 import Organism

runworld.Organism = Organism

DONOR_BUDGETS = (1000, 1500, 2000, 3000)
TARGET_BUDGETS = (300, 600)
SIG_BUDGET = 1000            # бюджет подписи, как у запросов repertoire.py


if __name__ == "__main__":
    donor_w, probe_w, exam_w = sys.argv[1:4]
    glue_exam = glue_of(exam_w)
    _, probe, ask, undo = glue_exam["exams"][0]
    print(f"held-out v2: донор={donor_w}, цель-зонд={probe_w}, цель={exam_w}")

    # донор: медленная жизнь ради КАРТЫ (правило v1 без изменений)
    donor_org, dax, tried = None, None, []
    for db in DONOR_BUDGETS:
        org = runworld.live(glue_of(donor_w), False, db)
        _, ax = axis_with(org, ask)
        tried.append(f"{db}{'+' if ax is not None else '-'}")
        if ax is not None:
            donor_org, dax = org, ax
            break
    if dax is None:
        print(f"донор: ось не завелась ({' '.join(tried)}) — ПРОВАЛ")
        sys.exit(1)
    form = {"org": donor_org, "ax": dax, "sig": koopman.signature(dax),
            "k": dax.k}

    # библиотека узнавания = оригинальный репертуар + донор (подпись донора
    # строится тем же quick_signature, что у всех членов библиотеки)
    lib = discover_lib(SIG_BUDGET)
    dsig, _ = quick_signature(glue_of(donor_w), SIG_BUDGET)
    lib["донор"] = dsig
    cal = koopman.calibrate(lib.values())
    rad = radius(lib, cal)
    print(f"донор в библиотеке: k={dsig['k']} spec={dsig['spec']}; "
          f"радиус знакомого = {rad:.2f} (метрика калибрована, "
          f"{ {c: round(v, 2) for c, v in cal.items()} })")

    glue_probe = glue_of(probe_w)
    print(f"\n{'бюдж':>5} | {'узнавание':>32} | {'решение':>12} | "
          f"{'ГОЛАЯ':>6} | {'ВВОЗ+ВЫРАВН':>11}")
    verdicts = []
    for tb in TARGET_BUDGETS:
        q, _ = quick_signature(glue_probe, tb)
        ranked = sorted((koopman.dist(q, lib[n], cal), n) for n in lib)
        d1, near = ranked[0]
        d_donor = koopman.dist(q, dsig, cal)
        org, f, _, _ = recognize_and_fit(glue_probe, tb, [form])
        bare = runworld.exam(org, glue_exam, probe, ask)
        if d_donor <= rad:
            decision = "ВВОЗ"
            org.mem = import_usemap(org, f["org"], ask, True)[0]
            al = runworld.exam(org, glue_exam, probe, ask)
            verdicts.append(("import", al[0] / al[1] if al[1] else 0.0))
            score = f"{al[0]:>7}/{al[1]:<3}"
        elif d1 <= rad:
            decision = "нет карты"
            verdicts.append(("no_map", None))
            score = f"{'—':>11}"
        else:
            decision = "ОТКАЗ"
            verdicts.append(("refuse", None))
            score = f"{'—':>11}"
        print(f"{tb:>5} | ближ={near:<10s} d={d1:5.2f} dдон={d_donor:5.2f}"
              f" | {decision:>12} | {bare[0]:>3}/{bare[1]:<2} | {score}")

    kinds = {k for k, _ in verdicts}
    if kinds == {"import"}:
        ok = all(s >= 7 / 8 for _, s in verdicts)
        print(f"\nВЕРДИКТ: {'СДАН' if ok else 'ПРОВАЛЕН'} "
              f"(ввоз состоялся; критерий >= 7/8 на всех бюджетах)")
    elif "import" not in kinds:
        print("\nВЕРДИКТ: ОТКАЗ ОТ ВВОЗА (форма цели не узнана как донор) — "
              "правильность отказа решает автор мира")
    else:
        print("\nВЕРДИКТ: РАСЩЕПЛЁН по бюджетам (узнавание нестабильно) — "
              "публикуется как есть")
