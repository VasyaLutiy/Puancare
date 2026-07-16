"""recognize — первый акт узнавания: узнать форму нового мира по библиотеке
уже прожитых, БЕЗ единого слова из yaml.

Библиотека = подписи прожитых миров (sig.py, форма без объектов/слов).
recognize(подпись, библиотека) -> ближайший шаблон + уверенность + вердикт
«узнан / новая форма». Порог новизны — НЕ константа: он выведен из самой
библиотеки (насколько её члены далеки друг от друга); чужак дальше этого
радиуса = форма, которой организм ещё не жил.

Тест — leave-one-out: прячем каждый мир, узнаём по остальным. Falsifiable:
проверяем не только «попал в семью», но и сбылся ли структурный прогноз
(совпало ли k предсказанного шаблона с k спрятанного мира) — розетка на
мета-уровне: узнавание есть ставка, которую спрятанный мир может опровергнуть.
"""

import sys

import sig


def familiar_radius(names, sigs):
    """Насколько далеко члены библиотеки от СВОИХ ближайших — предел
    «знакомого». Дальше него = новая форма. Никаких магических констант."""
    r = 0.0
    for a in names:
        others = [b for b in names if b != a]
        if others:
            r = max(r, min(sig.dist(sigs[a], sigs[b]) for b in others))
    return r


def recognize(query, library):
    """library = {имя: подпись}. -> (ранжирование, радиус)."""
    ranked = sorted(((sig.dist(query, s), name) for name, s in library.items()))
    radius = familiar_radius(list(library), library)
    return ranked, radius


def report(name, query, library, expect, tag="спрятан"):
    """expect: множество приемлемых ближайших имён, или None если ждём
    «новую форму»."""
    ranked, radius = recognize(query, library)
    d1, near = ranked[0]
    d2 = ranked[1][0] if len(ranked) > 1 else d1 + 1.0
    conf = (d2 - d1) / d2 if d2 > 0 else 1.0
    novel = d1 > radius
    verdict = "НОВАЯ ФОРМА" if novel else f"узнан как «{near}»"
    pred_k = library[near]["k"]      # falsifiable: k шаблона vs факт запроса
    k_ok = "" if novel else ("k совпал" if pred_k == query["k"]
                             else f"k ПРОМАХ ({pred_k}≠{query['k']})")
    if expect is None:
        family_ok = "верно(новая)" if novel else "мимо(ждали новую)"
        ok = novel
    else:
        family_ok = "верно" if (not novel and near in expect) else "мимо"
        ok = not novel and near in expect
    print(f"  {tag} {name:10s}: {verdict:22s} "
          f"(ближ={d1:.2f}, увер={conf:.0%}, радиус={radius:.2f})  "
          f"семья:{family_ok}  {k_ok}")
    return ok


def build_sigs(worlds):
    out = {}
    for w in worlds:
        ax = sig.best_axis(w)
        if ax is None:
            print(f"  {w}: ось не выучена"); continue
        out[w] = sig.signature(ax)
        s = out[w]
        print(f"  {w:10s}: k={s['k']} устойч={s['persist']:.3f} "
              f"направл={s['directed']:.3f} резк={s['sharp']} исх={s['ncards']}")
    return out


TANK = {"batteries", "car"}   # семья-бак 2 состояния


def main_loo(worlds):
    expect = {"batteries": TANK, "car": TANK, "bike": TANK, "bird": None}
    print("=== жизни -> подписи ===")
    sigs = build_sigs(worlds)
    print("\n=== leave-one-out узнавание ===")
    ok = 0
    for w in worlds:
        if w not in sigs:
            continue
        lib = {n: s for n, s in sigs.items() if n != w}
        ok += report(w, sigs[w], lib, expect.get(w, TANK))
    print(f"\nсовпало с прогнозом: {ok}/{len(sigs)}")


def main_novel(lib_worlds, q_worlds, expect):
    print("=== библиотека -> подписи ===")
    lib = build_sigs(lib_worlds)
    print("=== запросы (held-out) -> подписи ===")
    qs = build_sigs(q_worlds)
    print("\n=== узнавание пришельцев по фиксированной библиотеке ===")
    ok = 0
    for w in q_worlds:
        if w in qs:
            ok += report(w, qs[w], lib, expect.get(w, TANK), tag="запрос")
    print(f"\nсовпало с прогнозом: {ok}/{len(qs)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--lib" in a:
        lib_worlds = a[a.index("--lib") + 1].split(",")
        q_worlds = a[a.index("--query") + 1].split(",")
        # зарегистрированный прогноз: batt1/batt2 -> семья-бак; batt3 (3
        # ступени) -> лестница = bird (форма бьёт имя)
        expect = {"batt1": TANK, "batt2": TANK, "batt3": {"bird"}}
        main_novel(lib_worlds, q_worlds, expect)
    else:
        main_loo(a or ["batteries", "car", "bike", "bird"])
