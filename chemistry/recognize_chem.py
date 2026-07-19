"""recognize_chem — узнавание миров МОЛЕКУЛ против полки CHEM-01.

Тем же способом, что cards_chem01.py, снимаем карточку формы с КАЖДОГО
прогнанного мира молекулы (полный organism2 на анкете chemistry/in/<партия>/
molNN/мир.yaml -> самая наблюдаемая ось -> koopman.signature). Дальше —
дистанции до 9 карточек полки CHEM-01 (cards_chem01.json), радиус полки
(repertoire-идиома: макс по членам от расстояния до своего ближайшего),
вердикт: РОДНЯ ближайшей карточки, если дистанция <= радиуса, иначе НОВОЕ.

Прибор замороженный: берём только штатный вывод koopman/sig; ни строки в
ядре не трогаем. Полка сравнения и ядро — read-only; ничего не пишем в
боевую полку/леджер.

ДВУХОСЕВЫЕ МИРЫ (молекула из двух динамических групп): прибор отдаёт ОДНУ
подпись на мир (самая наблюдаемая ось — max n_obs). Мы это честно печатаем
(колонка «осей»): расщепления не изобретаем, показываем, какую одну ось
организм счёл главной и как она узналась. Факторизацию (2 переменные в
отчёте) меряет полный прогон runworld — не эта ступень.

    python3 chemistry/recognize_chem.py [chemistry/in/batch-01]
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runworld                                     # noqa: E402
import koopman                                      # noqa: E402
import sig                                          # noqa: E402
from organism2 import Organism                      # noqa: E402
from worldkit import load_spec, make_glue           # noqa: E402

runworld.Organism = Organism

SHELF_JSON = os.path.join(_HERE, "cards_chem01.json")
BUDGET = 2000


# ----------------------------------------------------------------- полка
def load_shelf():
    """9 карточек CHEM-01 (koopman-подписи) из cards_chem01.json."""
    with open(SHELF_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    lib = {}
    for key, card in data["cards"].items():
        if card.get("koopman"):
            lib[key] = card["koopman"]
    return lib


# --------------------------------------------------------- снятие карточки
def discover(world_yaml):
    """Полный organism2 на анкете мира -> koopman/sig самой наблюдаемой оси.
    Возвращает подпись формы или None (ось не выучена)."""
    glue = make_glue(load_spec(world_yaml))
    org = runworld.live(glue, False, BUDGET)
    ax = max(org.axes, key=lambda a: a.n_obs) if org.axes else None
    if ax is None:
        return None
    return {"koopman": koopman.signature(ax), "sig": sig.signature(ax),
            "naxes": len(org.axes)}


# ------------------------------------------------------------- вердикты
def _topology(card):
    """Словесная форма из штатных чисел (как cards_chem01._topology)."""
    s, ss = card["koopman"], card["sig"]
    osc = any(x > 0.05 for x in s["osc"])
    if osc:
        shape = "кольцо/осц"
    elif ss["directed"] >= 0.5:
        shape = "сток →"
    else:
        shape = "тумблер ⇄"
    return f"k={s['k']} {shape} напр={ss['directed']:.2f}"


def batch_worlds(batch_dir):
    """Список (имя_mol, путь_к_миру) партии, отсортированный."""
    out = []
    for name in sorted(os.listdir(batch_dir)):
        wpath = os.path.join(batch_dir, name, "мир.yaml")
        if os.path.isdir(os.path.join(batch_dir, name)) and os.path.exists(wpath):
            out.append((name, wpath))
    return out


def main():
    batch_dir = (sys.argv[1] if len(sys.argv) > 1
                 else os.path.join(_HERE, "in", "batch-01"))
    if not os.path.isdir(batch_dir):
        sys.exit(f"нет партии: {batch_dir}")

    lib = load_shelf()
    weights = koopman.calibrate(list(lib.values()))
    radius = _radius(lib, weights)

    print("=" * 82)
    print("УЗНАВАНИЕ МОЛЕКУЛ ПРОТИВ ПОЛКИ CHEM-01 (штатный прибор, без слов)")
    print("=" * 82)
    print(f"полка: {len(lib)} карточек; радиус знакомого = {radius:.2f} "
          f"(веса калиброваны по полке)\n")
    hdr = (f"{'мир':7} {'осей':4} {'форма мира':22} "
           f"{'ближайшая':10} {'дист':>7} {'радиус':>7} вердикт")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, wpath in batch_worlds(batch_dir):
        card = discover(wpath)
        if card is None:
            print(f"{name:7} {'—':4} {'(ось не выучена)':22}")
            rows.append({"world": name, "verdict": "невыучена"})
            continue
        ranked = sorted((koopman.dist(card["koopman"], lib[n], weights), n)
                        for n in lib)
        d, near = ranked[0]
        kin = d <= radius
        verdict = "РОДНЯ" if kin else "НОВОЕ"
        naxes = card["naxes"]
        print(f"{name:7} {naxes:>4} {_topology(card):22} "
              f"{near:10} {d:>7.2f} {radius:>7.2f} {verdict}")
        rows.append({"world": name, "naxes": naxes, "near": near,
                     "dist": round(d, 3), "radius": round(radius, 3),
                     "verdict": verdict})

    print("\nПримечание: колонка «осей» — сколько скрытых осей у мира; прибор "
          "\nотдаёт ОДНУ подпись (самая наблюдаемая ось). Двухосевые "
          "(осей=2) —\nоткрытый вопрос B4: расщепление не мерим здесь.")
    return rows


def _radius(lib, w):
    """Радиус знакомого = макс по членам от расстояния до своего ближайшего
    (repertoire.radius / novelty_declared.radius — та же идиома)."""
    names = list(lib)
    r = 0.0
    for a in names:
        others = [koopman.dist(lib[a], lib[b], w) for b in names if b != a]
        if others:
            r = max(r, min(others))
    return r


if __name__ == "__main__":
    main()
