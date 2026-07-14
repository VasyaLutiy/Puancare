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


def report(name, query, library, expect):
    ranked, radius = recognize(query, library)
    d1, near = ranked[0]
    d2 = ranked[1][0] if len(ranked) > 1 else d1 + 1.0
    conf = (d2 - d1) / d2 if d2 > 0 else 1.0
    novel = d1 > radius
    verdict = "НОВАЯ ФОРМА" if novel else f"узнан как «{near}»"
    # falsifiable: прогноз шаблона о k против факта спрятанного мира
    pred_k = library[near]["k"]
    true_k = query["k"]
    k_ok = "" if novel else ("k совпал" if pred_k == true_k
                             else f"k ПРОМАХ ({pred_k}≠{true_k})")
    family_ok = ("НОВАЯ" if expect is None else
                 ("верно" if (not novel and near == expect) else "мимо"))
    print(f"  спрятан {name:10s}: {verdict:20s} "
          f"(ближ={d1:.2f}, увер={conf:.0%}, радиус={radius:.2f})  "
          f"семья:{family_ok}  {k_ok}")
    ok = (novel and expect is None) or (not novel and near == expect)
    return ok


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["batteries", "car", "bike", "bird"]
    # ожидаемая семья каждого (мой зарегистрированный прогноз до прогона):
    # batteries<->car; bike -> к семье-баку (batteries); bird -> новая форма
    expect = {"batteries": "car", "car": "batteries",
              "bike": "batteries", "bird": None}

    print("=== жизни -> подписи ===")
    sigs = {}
    for w in worlds:
        ax = sig.best_axis(w)
        if ax is None:
            print(f"{w}: ось не выучена"); continue
        sigs[w] = sig.signature(ax)
        s = sigs[w]
        print(f"  {w:10s}: k={s['k']} устойч={s['persist']:.3f} "
              f"направл={s['directed']:.3f} резк={s['sharp']} исх={s['ncards']}")

    print("\n=== leave-one-out узнавание ===")
    ok = 0
    for w in worlds:
        if w not in sigs:
            continue
        lib = {n: s for n, s in sigs.items() if n != w}
        ok += report(w, sigs[w], lib, expect.get(w))
    print(f"\nсовпало с прогнозом: {ok}/{len(sigs)}")
