"""lab_m01diff — минимальный дифф-эксперимент: comp1 → форма m01 по одной оси.

comp1 (ручной изоморф, ввоз 8/8) кумулятивно мутирует в форму m01
(станочный изоморф, ввоз 0/8). Экзамен после каждого шага; ось, на
которой счёт падает, — имя дыры композиции.

Лаборатория, не экзамен: полка и ведомость — подменные копии в
experiments/labm01/, боевые файлы не трогаются. Запуск из корня репо:
  python3 experiments/lab_m01diff.py
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldforge"))

import exam_shelf
from triad import make_probe_text

LAB = "experiments/labm01"
LEDGER = f"{LAB}/lab_ledger.jsonl"

DONOR = open("worlds/comp1/donor.yaml").read()
TARGET = open("worlds/comp1/target.yaml").read()
PROBE_DISK = open("worlds/comp1/target_probe.yaml").read()


def rep(text, old, new):
    assert text.count(old) == 1, f"якорь не уникален: {old!r}"
    return text.replace(old, new)


# кумулятивные оси: (имя, мутация донора, мутация цели)
AXES = [
    ("ось1 видимое 3 токена",
     lambda d: rep(d, "индикатор: [горит, тускл]", "индикатор: [горит, тускл, чёрен]"),
     lambda t: rep(t, "лампа: [светит, темно]", "лампа: [светит, темно, мутна]")),
    ("ось2 вторая ловушка на последней",
     lambda d: rep(d, "- если заряд=полный то индикатор=горит (80%)",
                   "- если заряд=полный то индикатор=горит (80%)\n"
                   "    - если заряд=пустой то индикатор=чёрен (70%)"),
     lambda t: rep(t, "- если режим=активный то лампа=светит (80%)",
                   "- если режим=активный то лампа=светит (80%)\n"
                   "    - если режим=стоп то лампа=мутна (70%)")),
    ("ось3 [исчезает] на иначе вопроса",
     lambda d: rep(d, "- иначе глохнет", "- иначе глохнет [исчезает]"),
     lambda t: rep(t, "- иначе глохнет", "- иначе глохнет [исчезает]")),
    ("ось4 вмешательство выбираешь",
     lambda d: rep(d, "зарядить:\n    фиксировано: заряд=полный",
                   "зарядить:\n    выбираешь: [индикатор]\n    фиксировано: заряд=полный"),
     lambda t: rep(t, "сброс:\n    фиксировано: режим=активный",
                   "сброс:\n    выбираешь: [лампа]\n    фиксировано: режим=активный")),
    ("ось5 гарантия с последней на среднюю",
     lambda d: rep(d, "минимум 2 объектов с заряд=пустой",
                   "минимум 2 объектов с заряд=средний"),
     lambda t: rep(t, "минимум 2 объектов с режим=стоп",
                   "минимум 2 объектов с режим=вялый")),
    ("ось6 темп 4/5",
     lambda d: rep(d, "шанс_стать средний (5%)", "шанс_стать средний (4%)"),
     lambda t: rep(t, "шанс_стать вялый (5%)", "шанс_стать вялый (4%)")),
]


def build_step(name, donor, target, probe=None):
    d = f"{LAB}/{name}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/donor.yaml", "w").write(donor)
    open(f"{d}/target.yaml", "w").write(target)
    open(f"{d}/target_probe.yaml", "w").write(
        probe if probe is not None else make_probe_text(target))
    return d


def run_step(name, step_dir):
    shelf_copy = f"{step_dir}/shelf.json"
    shutil.copy("shelf/shelf.json", shelf_copy)
    sh = exam_shelf.ensure_shelf(shelf_copy)
    w = f"../{step_dir}"
    r = exam_shelf.run_triad(sh, f"{w}/donor", f"{w}/target_probe", f"{w}/target",
                             ledger=LEDGER)
    rows = [json.loads(l) for l in open(LEDGER)]
    mine = [x for x in rows if x["name"] == f"{w}/target"]
    scores = {x["budget"]: (x["decision"], x["score"]) for x in mine}
    return r["verdict"], scores


if __name__ == "__main__":
    os.makedirs(LAB, exist_ok=True)
    if os.path.exists(LEDGER):
        os.remove(LEDGER)
    results = []

    steps = [("step0-контроль", DONOR, TARGET, PROBE_DISK),
             ("step0b-зонд-реген", DONOR, TARGET, None)]
    d, t = DONOR, TARGET
    for i, (axis, fd, ft) in enumerate(AXES, 1):
        d, t = fd(d), ft(t)
        steps.append((f"step{i}-{axis.split()[0]}", d, t, None))

    for i, (name, dn, tg, pr) in enumerate(steps):
        print(f"\n######## {name} ########")
        step_dir = build_step(name, dn, tg, pr)
        verdict, scores = run_step(name, step_dir)
        results.append((name, verdict, scores))

    print("\n================ СВОДКА ================")
    for name, verdict, scores in results:
        s = "  ".join(f"@{b} {d} " + (f"{sc[0]}/{sc[1]}" if sc else "—")
                      for b, (d, sc) in sorted(scores.items()))
        print(f"{name:24} | {verdict:10} | {s}")
