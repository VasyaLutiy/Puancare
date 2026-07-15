"""harness — петля слепого противника: приём → валидация → упаковка тройки →
гейт новизны → маршрут (экзамен / регресс) → манифест заморозки.

Форж НЕ придумывает миры (это подгонка). Он принимает тройки, сгенерённые
ВСЛЕПУЮ внешней моделью (prompts/), и механически:
  1. компилирует worldkit.Spec — брак ловит компилятор;
  2. выводит target_probe (triad.build);
  3. судит по ЗАЯВЛЕННОЙ структуре (novelty_declared.classify_triad) —
     БЕЗ запуска ядра, метрикой, ОТЛИЧНОЙ от узнавателя ядра (разрыв круга);
  4. маршрутизирует: novel → EXAM (frontier, замораживается),
                     family → REGRESSION (быстрый CI ядра);
  5. печатает МАНИФЕСТ ЗАМОРОЗКИ — что закоммитить ДО прогона, чтобы автор не
     подстроил ядро под свежий мир (протокол EXAMS.md).

Порядок соблюдения слепоты — на совести оператора: не читать содержимое до
вердикта. Форж лишь делает это дешёвым: маршрут строится по структуре, а не
по смыслу слов.

Приём:  worldforge/harness.py <входной_каталог> [--out <каталог>]
  <входной_каталог>/<имя>/donor.yaml + target.yaml   (target_probe выведется)
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import triad                                  # noqa: E402
import novelty_declared as nd                 # noqa: E402


def library():
    """Библиотека «что ядро знает» = архетипы repertoire.py, ЗАЯВЛЕННЫЕ подписи.
    Импорт repertoire ленивый (тянет ядро) — сами подписи ядро не запускают."""
    import repertoire
    return {name: nd.sig_of_text(text)
            for name, text in repertoire.REPERTOIRE.items()}


def assess(donor_text, target_text, lib=None, w=None, rad=None):
    """Один кандидат: собрать тройку и вынести маршрут. Возвращает dict-отчёт
    или бросает ValueError, если тройка не компилируется/несогласована."""
    if lib is None:
        lib = library()
    if w is None:
        w = nd.calibrate(lib.values())
    if rad is None:
        rad = nd.radius(lib, w)
    tri = triad.build(donor_text, target_text)       # валидирует всё
    ds = nd.declared_signature(tri["donor"][1])
    ts = nd.declared_signature(tri["target"][1])
    novel, why, d, near = nd.classify_triad(ds, ts, lib, w, rad)
    return {"triad": tri, "donor_sig": ds, "target_sig": ts,
            "novel": novel, "route": "EXAM" if novel else "REGRESSION",
            "why": why, "d": d, "near": near,
            "exam": tri["exam"]}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _clean(e):
    """Санитизация ошибки для панели (ТЗ §8.1): только до первой кавычки —
    str(e) компилятора цитирует слова миров = утечка слепоты."""
    msg = str(e)
    for q in ("'", '"', "«"):
        i = msg.find(q)
        if i != -1:
            msg = msg[:i]
    return msg.strip().rstrip(":") or type(e).__name__


def run(in_dir, out_dir):
    lib = library()
    w = nd.calibrate(lib.values())
    rad = nd.radius(lib, w)
    names = sorted(n for n in os.listdir(in_dir)
                   if os.path.isdir(os.path.join(in_dir, n)))
    exams, regress, rejects = [], [], []

    print(f"=== приём {len(names)} кандидатов из {in_dir} ===")
    print(f"радиус знакомого = {rad:.2f} | веса = "
          f"{ {c: round(v, 2) for c, v in w.items()} }\n")

    for name in names:
        d = os.path.join(in_dir, name)
        dp, tp = os.path.join(d, "donor.yaml"), os.path.join(d, "target.yaml")
        if not (os.path.exists(dp) and os.path.exists(tp)):
            print(f"  ✗ {name}: нет donor.yaml/target.yaml")
            rejects.append((name, "нет донора/цели"))
            continue
        try:
            rep = assess(_read(dp), _read(tp), lib, w, rad)
        except ValueError as e:
            print(f"  ✗ {name}: БРАК компиляции — {_clean(e)}")
            rejects.append((name, _clean(e)))
            continue
        route = rep["route"]
        dst = os.path.join(out_dir, route.lower(), name)
        triad.write(rep["triad"], dst)
        (exams if rep["novel"] else regress).append((name, rep, dst))
        # роли, не слова: имена зонда/вопроса — утечка смысла (ТЗ §8.1)
        print(f"  {'◆' if rep['novel'] else '·'} {name}: {route:10s} "
              f"({rep['why']}) экзамен зонд→вопрос d={rep['d']:.2f}")

    _manifest(exams, regress, rejects, out_dir)
    return {"exam": exams, "regression": regress, "rejects": rejects}


def _manifest(exams, regress, rejects, out_dir):
    print(f"\n=== СВОДКА ===")
    print(f"  экзамен (frontier): {len(exams)}  |  регресс: {len(regress)}"
          f"  |  брак: {len(rejects)}")
    if exams:
        print(f"\n=== МАНИФЕСТ ЗАМОРОЗКИ (протокол EXAMS.md) ===")
        print("Слепые frontier-тройки. ЗАМОРОЗЬ раннер и тройки коммитом ДО "
              "прогона —\nне читай содержимое, не правь ядро под них:")
        rel = os.path.relpath(os.path.join(out_dir, "exam"), _ROOT)
        print(f"    git add {rel}")
        print(f'    git commit -m "held-out тройки заморожены ДО прогона: '
              f'{", ".join(n for n, *_ in exams)}"')
        print("Затем прогон замороженным ядром exam_shelf.py на каждой тройке "
              "(действия экзамена ядро возьмёт из мира само):")
        for name, rep, dst in exams:
            rl = os.path.relpath(dst, _ROOT)
            print(f"    # {name}: заморожен в {rl}/ — прогнать exam_shelf.py "
                  f"на этой тройке")
        print("Вскрытие содержимого и веса — ТОЛЬКО после вердиктов.")
    if regress:
        print(f"\nРегресс-набор (ядро уже умеет — быстрый CI, содержимое "
              f"читать можно): {', '.join(n for n, *_ in regress)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = "worldforge/out"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if not args:
        print(__doc__)
        sys.exit(0)
    run(args[0], out)
