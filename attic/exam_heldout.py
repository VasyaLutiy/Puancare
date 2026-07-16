"""exam_heldout — held-out экзамен композиции. Миры — от Кирилла.

ПРОТОКОЛ ЗАМОРОЖЕН ДО ПОЯВЛЕНИЯ МИРОВ. После их появления этот файл не
редактируется; прогон один; результат публикуется как есть, включая провал.
(Правило из green-tests-are-cheap-for-me: правка кода или миров после
увиденного результата — сама результат, и она запрещена молча.)

Контракт трёх файлов (как worlds/comp/{donor,target,target_probe}):
  донор        — полный мир: зонд + использование, живёт медленной дорогой
  цель-зонд    — тот же мир формой, но доступен ТОЛЬКО зонд
  цель         — полный, с разделом «экзамены: - зонд X вопрос Y»
Уступка первого кирпича (слой 1): имена действий и токены исходов общие
МЕЖДУ донором и целью; слова скрытого/видимого/объектов — любые свои.
Заявка под проверкой: КАСКАДЫ (для тумблеров окно пусто — уже измерено).

Правила, объявленные заранее:
  бюджет донора — первый из (1000, 1500, 2000, 3000), где судья завёл ось
  с действием-использованием; бюджеты цели — 300 и 600.
  Успех: ГОЛАЯ проваливает (иначе экзамен не про композицию — репортим),
  ВВОЗ+ВЫРАВН сдаёт (>=7/8) на обоих бюджетах цели.

Запуск:  python3 exam_heldout.py <донор> <цель-зонд> <цель>
         (имена без worlds/ и .yaml, напр. exam/k_donor exam/k_probe exam/k)
"""

import sys
import time

import runworld
from compose import import_usemap, axis_with, role_order, role_gap
from fastpath import recognize_and_fit, glue_of
from organism2 import Organism

runworld.Organism = Organism

DONOR_BUDGETS = (1000, 1500, 2000, 3000)
TARGET_BUDGETS = (300, 600)


if __name__ == "__main__":
    donor_w, probe_w, exam_w = sys.argv[1:4]
    glue_exam = glue_of(exam_w)
    _, probe, ask, undo = glue_exam["exams"][0]
    print(f"held-out: донор={donor_w}, цель-зонд={probe_w}, цель={exam_w}")
    print(f"экзамен: зонд «{probe}» → вопрос «{ask}»")

    donor_org, dax, tried = None, None, []
    for db in DONOR_BUDGETS:
        t0 = time.time()
        org = runworld.live(glue_of(donor_w), False, db)
        _, ax = axis_with(org, ask)
        tried.append(f"{db}{'+' if ax is not None else '-'}")
        if ax is not None:
            donor_org, dax = org, ax
            print(f"донор: ось принята на бюджете {db} за {time.time()-t0:.0f}с"
                  f" (приёмка: {' '.join(tried)}), k={dax.k}, "
                  f"role_gap={role_gap(dax):.2f}, роли={role_order(dax)}")
            break
    if dax is None:
        print(f"донор: ось не завелась ни на одном бюджете ({' '.join(tried)})"
              f" — ПРОВАЛ ЭКЗАМЕНА (композиции не из чего собраться)")
        sys.exit(1)

    import koopman
    form = {"org": donor_org, "ax": dax, "sig": koopman.signature(dax),
            "k": dax.k}
    glue_probe = glue_of(probe_w)
    print(f"\n{'бюдж':>5} | {'время':>7} | {'ГОЛАЯ':>6} | {'ВВОЗ СЫРОЙ':>10} | "
          f"{'ВВОЗ+ВЫРАВН':>11}")
    ok = True
    for tb in TARGET_BUDGETS:
        t0 = time.time()
        org, f, d, _ = recognize_and_fit(glue_probe, tb, [form])
        dt = (time.time() - t0) * 1000
        bare = runworld.exam(org, glue_exam, probe, ask)
        org.mem = import_usemap(org, f["org"], ask, False)[0]
        raw = runworld.exam(org, glue_exam, probe, ask)
        org2, f2, _, _ = recognize_and_fit(glue_probe, tb, [form])
        org2.mem = import_usemap(org2, f2["org"], ask, True)[0]
        al = runworld.exam(org2, glue_exam, probe, ask)
        print(f"{tb:>5} | {dt:>5.0f}мс | {bare[0]:>3}/{bare[1]:<2} | "
              f"{raw[0]:>6}/{raw[1]:<3} | {al[0]:>7}/{al[1]:<3}")
        if bare[1] and bare[0] / bare[1] >= 0.9:
            print("  ! ГОЛАЯ сдаёт сама — экзамен не различает композицию")
        if not al[1] or al[0] / al[1] < 7 / 8:
            ok = False
    print(f"\nВЕРДИКТ: {'СДАН' if ok else 'ПРОВАЛЕН'} "
          f"(критерий: ввоз+выравн >= 7/8 на всех бюджетах цели)")
