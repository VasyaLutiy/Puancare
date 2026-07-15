"""exam_heldout3 — held-out экзамен композиции: порог новизны ЧЕРЕЗ ЛИНЗУ.

Улов v2: порог из repertoire.py, вшитый в лоб, отверг всё, включая
изоморфы донора (d=3.7 при радиусе 0.40). Причина — подписи снимались
разными линзами: библиотека полной жизнью (зонд+использование), цель —
только зондом. Разные длины sharp/ncards и другой фит спектра дают d,
меряющее ширину обзора, а не форму. (Этот же артефакт всегда стоял в
fastpath как «узнан(d=4.89)» — его прятало принудительное навязывание.)

v3: все подписи проецируются на линзу запроса — ТОЛЬКО ЗОНД. Донор
пере-подписывается по своему зонду, члены REPERTOIRE — по своим (зонд =
первое действие мира; порядок из yaml сохраняется). Логика решения —
оригинальная repertoire.py (радиус библиотеки, калиброванная метрика),
без изменений:

  d(цель, донор) <= радиус  → ВВОЗ карты, критерий >= 7/8
  d_min <= радиус, не донор → «узнан как X, карты нет» → без ввоза
  d_min > радиус            → ОТКАЗ (правильность решает автор мира)

Протокол заморожен до прогона; comp1..comp4 прогоняются все; смоук на
comp (не held-out) — до заморозки.

Запуск: python3 exam_heldout3.py <донор> <цель-зонд> <цель>
"""

import sys

import runworld
import koopman
from compose import import_usemap, axis_with
from fastpath import recognize_and_fit, quick_signature, glue_of
from repertoire import REPERTOIRE, radius
from repertoire import glue_of as glue_of_text
from organism2 import Organism

runworld.Organism = Organism

DONOR_BUDGETS = (1000, 1500, 2000, 3000)
TARGET_BUDGETS = (300, 600)
SIG_BUDGET = 1000


def probe_of(glue):
    return glue["object_actions"][0]        # зонд = первое действие мира


if __name__ == "__main__":
    donor_w, probe_w, exam_w = sys.argv[1:4]
    glue_exam = glue_of(exam_w)
    _, probe, ask, undo = glue_exam["exams"][0]
    print(f"held-out v3 (линза=зонд): донор={donor_w}, цель={exam_w}")

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

    # библиотека: REPERTOIRE + донор, ВСЕ подписи через линзу своего зонда
    lib = {}
    for name, text in REPERTOIRE.items():
        g = glue_of_text(text)
        lib[name], _ = quick_signature(g, SIG_BUDGET, acts=[probe_of(g)])
    dsig, _ = quick_signature(glue_of(donor_w), SIG_BUDGET, acts=[probe])
    lib["донор"] = dsig
    cal = koopman.calibrate(lib.values())
    rad = radius(lib, cal)
    print(f"донор-через-зонд: k={dsig['k']} spec={dsig['spec']}; "
          f"радиус знакомого = {rad:.2f}")

    glue_probe = glue_of(probe_w)
    print(f"\n{'бюдж':>5} | {'узнавание':>34} | {'решение':>9} | "
          f"{'ГОЛАЯ':>6} | {'ВВОЗ+ВЫРАВН':>11}")
    verdicts = []
    for tb in TARGET_BUDGETS:
        q, _ = quick_signature(glue_probe, tb)      # своя линза и есть зонд
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
        print(f"{tb:>5} | ближ={near:<12s} d={d1:5.2f} dдон={d_donor:5.2f}"
              f" | {decision:>9} | {bare[0]:>3}/{bare[1]:<2} | {score}")

    kinds = {k for k, _ in verdicts}
    if kinds == {"import"}:
        ok = all(s >= 7 / 8 for _, s in verdicts)
        print(f"\nВЕРДИКТ: {'СДАН' if ok else 'ПРОВАЛЕН'} "
              f"(ввоз состоялся; критерий >= 7/8 на всех бюджетах)")
    elif "import" not in kinds:
        print("\nВЕРДИКТ: ОТКАЗ ОТ ВВОЗА — правильность решает автор мира")
    else:
        print("\nВЕРДИКТ: РАСЩЕПЛЁН по бюджетам — публикуется как есть")
