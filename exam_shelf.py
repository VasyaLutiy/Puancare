"""exam_shelf — раннер экзамена композиции v4: библиотека = ШКАФ (shelf.py).

То же, что замороженный v3 (exam_heldout3.py), с тремя отличиями (ТЗ §6):
  1. Репертуар узнавания = шкаф, не «4 архетипа + донор ad-hoc». Донор
     открывается медленной дорогой (первый из 1000/1500/2000/3000 с осью),
     кладётся ЭФЕМЕРНОЙ карточкой: участвует в узнавании и ввозе; на полку —
     только если нов (recognize == ОТКАЗ).
  2. Каждый вердикт → строка в ведомость exams_ledger.jsonl (машиночит.).
  3. Точностные счётчики по ведомости (истина дописывается автором позже).

Семантика решения не меняется (v3): линза зонда, ВВОЗ только от донора.
Бит-в-бит с v3 держится тем, что recog_sig архетипа = quick_signature(
acts=[probe]) — та же формула, cal/радиус совпадают при тех же членах.

Запуск:
  python3 exam_shelf.py <донор> <цель-зонд> <цель>      # одна тройка
  python3 exam_shelf.py --t1                            # приёмка T1 (7 троек)
"""

import os
import sys
import json
import time

import runworld
import koopman
import card as card_mod
from shelf import Shelf
from compose import import_usemap, axis_with
from fastpath import recognize_and_fit, quick_signature, glue_of
from organism2 import Organism

runworld.Organism = Organism

DONOR_BUDGETS = (1000, 1500, 2000, 3000)
TARGET_BUDGETS = (300, 600)
SHELF_PATH = "shelf/shelf.json"
LEDGER = "exams_ledger.jsonl"


def ensure_shelf(path=SHELF_PATH):
    sh = Shelf.open(path)
    if not sh.cards:
        print("шкаф пуст — засев архетипов (одноразово, медленно)...")
        sh.seed_archetypes()
        print(f"засеяно, версия={sh.version}")
    return sh


def open_donor(donor_w, ask):
    """Медленная дорога, правило бюджетов v3: первый бюджет с заведшейся осью."""
    tried = []
    for db in DONOR_BUDGETS:
        org = runworld.live(glue_of(donor_w), False, db)
        _, ax = axis_with(org, ask)
        tried.append(f"{db}{'+' if ax is not None else '-'}")
        if ax is not None:
            return org, db, tried
    return None, None, tried


def run_triad(sh, donor_w, probe_w, exam_w, ledger=LEDGER, verbose=True):
    glue_exam = glue_of(exam_w)
    _, probe, ask, _ = glue_exam["exams"][0]
    glue_donor = glue_of(donor_w)

    donor_org, db, tried = open_donor(donor_w, ask)
    if donor_org is None:
        if verbose:
            print(f"{exam_w}: донор не завёлся ({' '.join(tried)}) — ПРОВАЛ")
        return {"name": exam_w, "verdict": "ПРОВАЛ_ДОНОРА"}

    donor_card = card_mod.build(donor_org, glue_donor, origin=donor_w,
                                discovery={"budget": db, "seed": 0})
    donor_form = {"sig": donor_card["sig"], "k": donor_card["k"]}  # для k-навяз.

    # на полку — только если донор НОВ (иначе изоморф раздует полку, ТЗ реш.#1)
    probe_lens = card_mod.recog_sig(donor_card)
    nov = sh.recognize(probe_lens).decision == "ОТКАЗ"
    if nov:
        sh.add(donor_card)

    glue_probe = glue_of(probe_w)
    rows, results = [], []
    if verbose:
        print(f"\n{exam_w}: донор k={donor_card['k']} @бюдж{db}, "
              f"полка v{sh.version}{' (+донор нов)' if nov else ''}")
        print(f"{'бюдж':>5} | {'решение':>9} | {'d1':>5} {'dдон':>5} "
              f"{'рад':>5} | {'ГОЛАЯ':>6} | {'ВВОЗ':>7}")

    for tb in TARGET_BUDGETS:
        t0 = time.time()
        q, _ = quick_signature(glue_probe, tb)              # линза цели = зонд
        vd = sh.recognize(q, donor=donor_card)
        org, _, _, _ = recognize_and_fit(glue_probe, tb, [donor_form])
        bare = runworld.exam(org, glue_exam, probe, ask)
        score = None
        if vd.decision == "ВВОЗ":
            org.mem = import_usemap(org, card_mod.donor_shim(donor_card),
                                    ask, True)[0]
            al = runworld.exam(org, glue_exam, probe, ask)
            score = [al[0], al[1]]
            sc = f"{al[0]}/{al[1]}"
        else:
            sc = "—"
        elapsed = round(time.time() - t0, 2)
        row = {"name": exam_w, "budget": tb, "decision": vd.decision,
               "score": score, "bare": [bare[0], bare[1]],
               "d1": round(vd["d"], 3), "d_donor": round(vd["d_donor"], 3),
               "radius": round(vd["radius"], 3), "shelf_v": vd["version"],
               "donor_budget": db, "seed": 0, "elapsed_s": elapsed,
               "truth": None}                               # истину впишет автор
        rows.append(row)
        results.append((vd.decision, score))
        if verbose:
            print(f"{tb:>5} | {vd.decision:>9} | {vd['d']:5.2f} "
                  f"{vd['d_donor']:5.2f} {vd['radius']:5.2f} | "
                  f"{bare[0]:>3}/{bare[1]:<2} | {sc:>7}")

    with open(ledger, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisions = {d for d, _ in results}
    if decisions == {"ВВОЗ"}:
        verdict = "СДАН" if all(s and s[0] / s[1] >= 7 / 8
                                for _, s in results) else "ПРОВАЛЕН"
    elif "ВВОЗ" not in decisions:
        verdict = "ОТКАЗ"
    else:
        verdict = "РАСЩЕПЛЁН"
    return {"name": exam_w, "verdict": verdict, "results": results}


# --------------------------------------------------------------- приёмка T1

T1 = [
    ("comp1/donor", "comp1/target_probe", "comp1/target", "ВВОЗ"),
    ("comp2/donor", "comp2/target_probe", "comp2/target", "ВВОЗ"),
    ("comp3/donor", "comp3/target_probe", "comp3/target", "ОТКАЗ"),
    ("comp4/donor", "comp4/target_probe", "comp4/target", "ОТКАЗ"),
    ("hard1/donor", "hard1/target_probe", "hard1/target", "ВВОЗ"),
    ("hard2/donor", "hard2/target_probe", "hard2/target", "ВВОЗ"),
    ("hard3/donor", "hard3/target_probe", "hard3/target", "ВВОЗ"),
]


def t1():
    sh = ensure_shelf()
    print(f"\n=== T1: воспроизведение вердиктов v3 (полка v{sh.version}) ===")
    ok = 0
    for donor_w, probe_w, exam_w, expect in T1:
        res = run_triad(sh, donor_w, probe_w, exam_w)
        got = {d for d, _ in res.get("results", [])}
        # ожидаем: ВВОЗ-тройка => все бюджеты ВВОЗ; ОТКАЗ => ни одного ВВОЗ
        hit = (got == {"ВВОЗ"}) if expect == "ВВОЗ" else ("ВВОЗ" not in got)
        ok += hit
        mark = "✓" if hit else "✗"
        print(f"  {mark} {exam_w}: {res['verdict']} (ждали {expect}) "
              f"{res.get('results')}")
    print(f"\nИТОГ T1: {ok}/{len(T1)} решений совпали с ведомостью")
    return ok


if __name__ == "__main__":
    if "--t1" in sys.argv:
        t1()
    elif len(sys.argv) >= 4:
        sh = ensure_shelf()
        r = run_triad(sh, sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"\nВЕРДИКТ: {r['verdict']}")
    else:
        print(__doc__)
