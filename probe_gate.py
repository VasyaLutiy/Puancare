"""probe_gate — развёртка решения ворот полки по бюджету цели.

Болезнь-кандидат (вскрыта A/B-прогоном batch-01..04 после фикса немоты):
изоморфная пара получает ВВОЗ на 300 и «нет карты» на 600+ — парный тест
pair_import сравнивает расстояние пары с СУММОЙ ШУМОВ (межполовинная
болтанка), а шум сжимается с данными быстрее, чем расстояние пары →
ворота наказывают за богатство данных (зеркало «казни за бедность»
b05-m01, уже вылеченной дрожью в другой ветке ворот).

Зонд: донор открывается ОДИН раз (карточка, как в exam_shelf), затем
цель-зонд проживается на сетке бюджетов; на каждом бюджете — вердикт
ворот и ВСКРЫТИЕ парного теста покомпонентно:

    d_c  <=  noise_c(цель) + noise_c(донор)     — по каждой c из COMPS

Печатает, какая компонента на каком бюджете ломает ввоз, и как ведут
себя обе стороны неравенства с ростом данных.

  python3 probe_gate.py [тройка=../worldforge/in/batch-01/m01] [бюджеты]
  python3 probe_gate.py --sweep   # три патологические + одна здоровая
"""

import sys

import card as card_mod
import exam_shelf
import koopman
from fastpath import glue_of, quick_signature

BUDGETS = (300, 600, 1000, 1500, 2000, 3000)


def open_donor_card(triad):
    donor_w = f"{triad}/donor"
    glue_exam = glue_of(f"{triad}/target")
    _, probe, ask, _ = glue_exam["exams"][0]
    org, db, tried = exam_shelf.open_donor(donor_w, ask)
    if org is None:
        raise SystemExit(f"донор {donor_w} не завёлся: {' '.join(tried)}")
    card = card_mod.build(org, glue_of(donor_w), origin=donor_w,
                          discovery={"budget": db, "seed": 0})
    return card, db


def dissect_pair(q, dsig):
    """Покомпонентное вскрытие pair_import: (c, d_c, lim_c, прошла?, в тесте?).
    Печатаются ВСЕ компоненты подписи; в решение входят только PAIR_COMPS
    (sharp исключён из парного теста — вскрытие probe_gate это и показало)."""
    from shelf import Shelf
    rows = []
    for c in koopman.COMPS:
        d = koopman.comp_dist(q, dsig, c)
        lim = (q["noise"].get(c, 0.0) + dsig["noise"].get(c, 0.0))
        rows.append((c, d, lim, d <= lim, c in Shelf.PAIR_COMPS))
    return rows


def sweep(triad, budgets=BUDGETS, sh=None):
    print(f"\n{'=' * 74}\nТРОЙКА {triad}")
    if sh is None:
        sh = exam_shelf.ensure_shelf()
    card, db = open_donor_card(triad)
    dsig = card_mod.recog_sig(card)
    glue_probe = glue_of(f"{triad}/target_probe")
    print(f"донор: k={card['k']} @бюдж{db}, "
          f"spec={dsig['spec']}, noise={ {c: round(v, 3) for c, v in dsig['noise'].items()} }")
    print(f"полка v{sh.version} (только чтение: донор эфемерен, add не зовём)")

    hdr = " | ".join(f"{c:>17s}" for c in koopman.COMPS)
    print(f"\n{'бюдж':>5} | {'ворота':>9} | {'d_дон':>6} {'d1':>6} "
          f"{'рад+дрожь':>9} | {hdr}")
    print(f"{'':>5} | {'':>9} | {'':>6} {'':>6} {'':>9} | "
          + " | ".join(f"{'d_c <= шумQ+шумД':>17s}" for _ in koopman.COMPS))

    flips = []
    prev = None
    for tb in budgets:
        q, _ = quick_signature(glue_probe, tb)
        vd = sh.recognize(q, donor=card)
        rows = dissect_pair(q, dsig)
        cells = " | ".join(
            f"{d:6.3f}{'<=' if ok else ' >'}{lim:6.3f}"
            f"{('✓' if ok else '✗') if used else '·'}"
            for (_c, d, lim, ok, used) in rows)
        print(f"{tb:>5} | {vd.decision:>9} | {vd['d_donor']:6.2f} "
              f"{vd['d']:6.2f} {vd['radius']:9.2f} | {cells}")
        if prev is not None and prev != vd.decision:
            broke = [c for (c, _d, _l, ok, used) in rows if used and not ok]
            flips.append((tb, prev, vd.decision, broke))
        prev = vd.decision

    if flips:
        print("\nпереломы ворот:")
        for tb, was, now, broke in flips:
            why = (f"компоненты сломали парный тест: {', '.join(broke)}"
                   if broke and now != "ВВОЗ" else "парный тест собрался")
            print(f"  @{tb}: {was} -> {now}  ({why})")
    else:
        print("\nпереломов нет — решение стабильно по бюджету")
    return flips


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--sweep" in argv:
        sh = exam_shelf.ensure_shelf()
        # патологические (изоморф теряет ВВОЗ на 600+) + здоровая (контроль)
        for t in ("../worldforge/in/batch-01/m01",
                  "../worldforge/in/batch-03/m06",
                  "../worldforge/in/batch-03/m05",
                  "../worldforge/in/batch-02/m06"):
            sweep(t, sh=sh)
    else:
        triad = argv[0] if argv else "../worldforge/in/batch-01/m01"
        budgets = tuple(int(b) for b in argv[1:]) or BUDGETS
        sweep(triad, budgets)
