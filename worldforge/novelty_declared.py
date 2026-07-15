"""novelty_declared — новизна мира по ЗАЯВЛЕННОЙ структуре, БЕЗ запуска ядра.

Зачем отдельная метрика, если есть koopman.dist? Круг. Узнаватель ядра судит
кандидата тем же koopman.dist. Отбирая экзаменационных кандидатов этой
метрикой, гарантируешь, что ядро назовёт их новыми — экзамен становится
зеркалом фильтра, зелёное задаром. Тот же грех, что «прогноз+код в одной
руке», переодетый в метрику.

Развязка — ДВЕ дистанции на РАЗНЫХ измерениях:
  * ЗАЯВЛЕННАЯ (здесь): читается из yaml через worldkit.Spec, ядро не бежит.
        k         число скрытых состояний ведущей оси
        tempo     средняя вероятность перехода за шаг (динамика)
        directed  1 однонаправленный поток (лестница), 0 обратимый (тумблер),
                  дробь между — кольцо/смешанный
        cards     кардинальность лучшего зонда (макс. исходов у действия)
  * ВЫУЧЕННАЯ (koopman, в экзамене): спектр оси, которую ядро реально нашло.

Фильтр — грубый гейт по ЗАЯВЛЕННОЙ: отсеять изоморфы, которые ядро уже жрёт.
Вердикт — по ВЫУЧЕННОЙ замороженным ядром, вскрытие после заморозки.

Веса дистанции — калиброваны из библиотеки (как koopman.calibrate), не руками.
Порог «знакомого» — радиус библиотеки (как repertoire.radius), без констант.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from worldkit import Spec, parse_text, load_spec   # noqa: E402


COMPS = ("k", "tempo", "directed", "cards")


# ----------------------------------------------------- подпись из Spec

def declared_signature(spec):
    """Структурная подпись, вычитанная из объявленного мира. Ядро не бежит."""
    if not spec.hidden:
        return None
    attr = _primary_axis(spec)
    states = spec.hidden[attr]
    k = len(states)

    dyn = [(fr, to, p) for a, fr, to, p in spec.dynamics if a == attr]
    tempo = sum(p for _, _, p in dyn) / len(dyn) if dyn else 0.0

    # Направленность имеет смысл лишь при k>=3: у двух состояний детальный
    # баланс выполняется всегда, направление НЕ выучиваемо из поведения
    # (sig.py:41). Гейт не должен видеть то, чего не видит ядро, — иначе
    # ложно разведёт батарейку и тумблер, которые ядро при k=2 не различает.
    if k >= 3:
        idx = {s: i for i, s in enumerate(states)}
        fwd = sum(1 for fr, to, _ in dyn if idx[to] > idx[fr])
        back = sum(1 for fr, to, _ in dyn if idx[to] < idx[fr])
        directed = abs(fwd - back) / (fwd + back) if (fwd + back) else 0.0
    else:
        directed = 0.0

    cards = max((len({r["result"] for r in rules})
                 for rules in spec.obj_actions.values()), default=0)

    return {"k": float(k), "tempo": round(tempo, 3),
            "directed": round(directed, 3), "cards": float(cards)}


def _primary_axis(spec):
    """Ведущая скрытая ось = та, чью динамику видно, иначе с макс. состояний.
    При нескольких скрытых берём носитель динамики (у него есть темп/поток)."""
    in_dyn = [a for a, *_ in spec.dynamics if a in spec.hidden]
    pool = in_dyn or list(spec.hidden)
    return max(pool, key=lambda a: len(spec.hidden[a]))


def sig_of_text(text):
    return declared_signature(Spec(parse_text(text)))


def sig_of_file(path):
    return declared_signature(load_spec(path))


# ----------------------------------------------------- дистанция и порог

def calibrate(sigs):
    """Веса из библиотеки: 1/средний попарный зазор компоненты. Компонента,
    не различающая членов, веса не получает. Как koopman.calibrate — без
    ручных констант. k тоже калибруется (не привилегирован жёстким 2.0)."""
    sigs = [s for s in sigs if s]
    pairs = [(a, b) for i, a in enumerate(sigs) for b in sigs[i + 1:]]
    w = {}
    for c in COMPS:
        m = (sum(abs(a[c] - b[c]) for a, b in pairs) / len(pairs)
             if pairs else 0.0)
        w[c] = 1.0 / m if m > 1e-9 else 0.0
    return w


def dist(a, b, w=None):
    if w is None:
        w = {c: 1.0 for c in COMPS}
    return sum(w[c] * abs(a[c] - b[c]) for c in COMPS)


def radius(lib, w=None):
    """Радиус знакомого = макс по членам от расстояния до СВОЕГО ближайшего.
    Чужак дальше радиуса = форма, которой ядро ещё не жило (repertoire.radius)."""
    names = list(lib)
    r = 0.0
    for a in names:
        others = [dist(lib[a], lib[b], w) for b in names if b != a]
        if others:
            r = max(r, min(others))
    return r


def gate(sig, lib, w=None, rad=None):
    """Грубый гейт одной подписи: NEW если дальше радиуса, иначе ISO.
    Возвращает (novel: bool, d, near, rad). Для одиночного мира; для тройки
    донор+цель см. classify_triad."""
    if w is None:
        w = calibrate(lib.values())
    if rad is None:
        rad = radius(lib, w)
    d, near = min((dist(sig, lib[n], w), n) for n in lib)
    return d > rad, d, near, rad


def openable(donor_sig, lib):
    """Есть ли форма донора НА ПОЛКЕ (в репертуаре)? Пороги выведены из
    библиотеки, не из констант:
      * k за глубочайшей известной цепью (k > max_k);
      * топология, которой в репертуаре нет (значение directed отсутствует
        у членов) — кольцо (дробное directed) против лестниц/тумблеров.
    Нет на полке => открытие будет ВПЕРВЫЕ => frontier (экзамен).

    Историческая правка (после фикса судьи 07935fe, тот же день): раньше
    здесь стояло «судья не заводит лишнюю ступень» — это было верно для
    СТАРОГО судьи (по-строчный min-кап остатка продавал незнание по цене
    знания; hard1 отвергался при +270 битах форс-фита). Групповой остаток
    это вылечил: ядро открывает и k=4 (hard1: 10/10 на 600), и кольца
    (hard3: k=2-аппроксимация; ввоз честно валится 3/8 — аппроксимация не
    тянет 3-исходную карту). Маршрутизация НЕ изменилась — «нет на полке»
    по-прежнему значит frontier — изменилось обоснование: это свойство
    РЕПЕРТУАРА, а не потолок судьи."""
    max_k = max(s["k"] for s in lib.values())
    if donor_sig["k"] > max_k:
        return False, f"k={donor_sig['k']:.0f} > глубины репертуара {max_k:.0f}"
    known_dir = {round(s["directed"], 2) for s in lib.values()}
    if round(donor_sig["directed"], 2) not in known_dir:
        return False, (f"топология directed={donor_sig['directed']:.2f} "
                       f"вне репертуара {sorted(known_dir)}")
    return True, "форма на полке"


def classify_triad(donor_sig, target_sig, lib, w=None, rad=None):
    """Вердикт тройки по модели ядра (fastpath.recognize_and_fit): открыть
    донора → положить форму в репертуар → узнать цель против {репертуар ∪ донор}.

    novel  = frontier: провал открытия ЛИБО отказ ввоза (цель ≠ форме донора)
    family = регресс:  донор открыт И цель узнана как знакомая форма
    Возвращает (novel: bool, причина: str, d, near)."""
    if w is None:
        w = calibrate(lib.values())
    if rad is None:
        rad = radius(lib, w)
    ok, why = openable(donor_sig, lib)
    if not ok:
        return True, f"формы нет на полке: {why}", 0.0, "—"
    union = {**lib, "·донор·": donor_sig}     # донор открыт → в репертуаре
    d, near = min((dist(target_sig, union[n], w), n) for n in union)
    if d > rad:
        return True, f"отказ ввоза: цель разошлась (ближ «{near}»)", d, near
    return False, f"узнана как «{near}»", d, near


# ----------------------------------------------------- ретро-тест

def _library():
    """«Что ядро уже знает» = архетипы repertoire.py: лестницы k=3, тумблеры
    k=2. Импорт ленивый — тянет koopman/organism2, тяжело; ядро не бежит,
    берём только тексты миров."""
    import repertoire
    return {name: sig_of_text(text)
            for name, text in repertoire.REPERTOIRE.items()}


# ожидаемые вердикты held-out серии (EXAMS.md, вскрытие 15.07). Гейт судит
# ДОНОРА — есть ли его форма на полке; на полке -> регресс (ввоз),
# нет на полке -> frontier (экзамен). Это и есть исход ведомости.
# NB: ведомость снята СТАРЫМ судьёй; после 07935fe hard1 сдаёт 10/10 на 600,
# но маршрут hard1 -> frontier остаётся верным (k=4 на полке нет).
#   family = ядро открыло и верно ввезло (изоморф/возмущённый родич)
#   novel  = новая форма (иная кардинальность/темп/топология)
RETRO = {
    "comp1": ("family", "изоморф донора k=3"),
    "comp2": ("family", "изоморф донора k=3"),
    "comp3": ("novel",  "глубокая лестница k=4"),
    "comp4": ("novel",  "быстрая лестница, чужой темп"),
    "hard1": ("novel",  "лестница k=4 медленная"),
    "hard2": ("family", "батарейка k=2 быстрая; ядро ввезло (k=2 навигир. по k)"),
    "hard3": ("novel",  "лестница-кольцо быстрая"),
}

# hard2 — задокументированное КОНСЕРВАТИВНОЕ расхождение. Цель ближе всего к
# СВОЕМУ донору (near=·донор·) — метрика верно ранжирует её как почти-семью.
# Промах чисто калибровочный: возмущение темпа 25->22 (Δ=0.03) × вес темпа 24
# = 0.72 > радиус 0.48, где 0.48 задаёт тощая библиотека из 4 близнецов
# (внутренний разброс темпа тумблеров 0.10..0.12 = 0.02 < перекос донора).
# Ядро же открыло k=2 и ввезло. Промах в безопасную сторону (регресс->экзамен),
# НЕ в опасную. Веса/радиус под 7/7 не крутим — это была бы подгонка под
# ведомость (см. память: зелёные тесты даром). Богаче библиотека — уйдёт сам.
KNOWN_CONSERVATIVE = {"hard2"}


def _retro():
    lib = _library()
    w = calibrate(lib.values())
    rad = radius(lib, w)
    print("=== библиотека (что ядро знает) — ЗАЯВЛЕННЫЕ подписи ===")
    for n, s in lib.items():
        print(f"  {n:12s}: k={s['k']:.0f} темп={s['tempo']:.2f} "
              f"направл={s['directed']:.2f} карт={s['cards']:.0f}")
    print(f"\nвеса (из библиотеки): "
          f"{ {c: round(v, 2) for c, v in w.items()} }")
    print(f"радиус знакомого = {rad:.2f}\n")

    print("=== ретро-тест: тройки held-out против ведомости "
          "(модель ядра: донор→репертуар→узнать цель) ===")
    ok = 0
    for name, (expect, note) in RETRO.items():
        ddir = os.path.join(_ROOT, "worlds", name)
        dp = os.path.join(ddir, "donor.yaml")
        tp = os.path.join(ddir, "target.yaml")
        if not (os.path.exists(dp) and os.path.exists(tp)):
            print(f"  ? {name}: нет донора/цели")
            continue
        ds, ts = sig_of_file(dp), sig_of_file(tp)
        novel, why, d, near = classify_triad(ds, ts, lib, w, rad)
        got = "novel" if novel else "family"
        hit = got == expect
        ok += hit
        mark = "✓" if hit else ("~" if name in KNOWN_CONSERVATIVE else "✗")
        verdict = "НОВАЯ" if novel else "семья"
        print(f"  {mark} {name}: {verdict:5s} — {why:40s} "
              f"[донор k={ds['k']:.0f} т={ds['tempo']:.2f} "
              f"напр={ds['directed']:.2f}; цель k={ts['k']:.0f} "
              f"т={ts['tempo']:.2f}] ждали {expect}")
    print(f"\nитог: {ok}/{len(RETRO)} совпадений с EXAMS.md")
    return ok


if __name__ == "__main__":
    _retro()
