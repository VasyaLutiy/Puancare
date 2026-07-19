"""cards_chem01 — карточки форм CHEM-01 для ставок S6/S7 (STAKE-chem01.md).

Прибор — замороженный (organism2 + worldkit + koopman/sig + fastpath),
ни строки в нём не меняется. Отсюда берём ТОЛЬКО штатный вывод:
  koopman.signature(ax) -> k, спектр |λ| оси, осцилляция, τ, резкость, исходы;
  sig.signature(ax)     -> persist (темп) и directed (0 обратимо ⇄, 1 сток →).

Ось добывается тем же путём, что fastpath.discover: полный organism2
(runworld.live) на анкете chemistry/worlds/<имя>.yaml, затем самая
наблюдаемая ось. Карточки печатаются и складываются в
chemistry/cards_chem01.json. В боевую полку/леджер НИЧЕГО не пишется.

Только динамические миры (у статических темп 0, спектр вырожден —
открытый вопрос Q1 ставки): m01 m02 m04 m05 m06 m07 m08 m09 m10.

    <python> chemistry/cards_chem01.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runworld                                    # noqa: E402
import koopman                                     # noqa: E402
import sig                                         # noqa: E402
from organism2 import Organism                     # noqa: E402
from worldkit import load_spec, make_glue          # noqa: E402

runworld.Organism = Organism

WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")

# Динамические миры ставки (порядок — как в ORACLE §4).
DYNAMIC = ["m01", "m02", "m04", "m05", "m06", "m07", "m08", "m09", "m10"]

# Пары ставок: S6 — разные длины лестниц; S7 — тумблер против стока.
PAIR_S6 = ("m01", "m02")
PAIR_S7 = ("m06", "m07")

BUDGET = 2000


def _file(key):
    for n in os.listdir(WORLDS_DIR):
        if n.startswith(key + "_") and n.endswith(".yaml"):
            return n[:-5]
    raise FileNotFoundError(f"нет анкеты для {key}")


def discover(key):
    """Открыть форму мира полным organism2 (как fastpath.discover, но по
    анкете из chemistry/worlds). Возвращает подпись формы или None-карту."""
    glue = make_glue(load_spec(os.path.join(WORLDS_DIR, _file(key) + ".yaml")))
    org = runworld.live(glue, False, BUDGET)
    ax = max(org.axes, key=lambda a: a.n_obs) if org.axes else None
    if ax is None:
        return {"key": key, "k": 1, "sig": None, "sig_sig": None}
    ksig = koopman.signature(ax)                    # штатный спектр формы
    ssig = sig.signature(ax)                         # штатные темп/направленность
    return {"key": key, "k": ksig["k"], "sig": ksig, "sig_sig": ssig}


def _topology(card):
    """Словесная форма из штатных чисел: k + направленность/осцилляция."""
    if card["sig"] is None:
        return "статичная/невыучена"
    s, ss = card["sig"], card["sig_sig"]
    osc = any(x > 0.05 for x in s["osc"])
    directed = ss["directed"]
    if osc:
        shape = "кольцо/осцилляция"
    elif directed >= 0.5:
        shape = "сток →"
    else:
        shape = "тумблер ⇄"
    return f"k={s['k']} {shape} (напр={directed:.2f})"


# ------------------------------------------------------------------ печать

def main():
    cards = {k: discover(k) for k in DYNAMIC}

    print("=" * 78)
    print("КАРТОЧКИ ФОРМ CHEM-01 (штатный прибор, без слов из yaml)")
    print("=" * 78)
    hdr = (f"{'мир':6} {'k':>2} {'спектр|λ|':>16} {'осц':>10} {'τ':>6} "
           f"{'темп':>6} {'напр':>6} {'резкость':>14} топология")
    print(hdr)
    print("-" * len(hdr))
    for k in DYNAMIC:
        c = cards[k]
        if c["sig"] is None:
            print(f"{k:6} {'—':>2} {'(ось не выучена)':>16}")
            continue
        s, ss = c["sig"], c["sig_sig"]
        print(f"{k:6} {s['k']:>2} {str(s['spec']):>16} {str(s['osc']):>10} "
              f"{s['tau']:>6} {ss['persist']:>6.3f} {ss['directed']:>6.3f} "
              f"{str(s['sharp']):>14} {_topology(c)}")

    # --- попарные koopman-дистанции (веса из библиотеки, штатный calibrate) ---
    valid = [k for k in DYNAMIC if cards[k]["sig"] is not None]
    w = koopman.calibrate([cards[k]["sig"] for k in valid])
    metric = lambda a, b: koopman.dist(cards[a]["sig"], cards[b]["sig"], w)

    print("\n" + "=" * 78)
    print("KOOPMAN: попарные дистанции формы (веса калиброваны по библиотеке)")
    print("=" * 78)
    print(f"веса: { {c: round(v, 2) for c, v in w.items()} }")
    print(" " * 7 + "".join(f"{k:>8}" for k in valid))
    for a in valid:
        row = "".join(f"{metric(a, b):8.2f}" for b in valid)
        print(f"{a:6} {row}")

    def pair_report(pair, stake, claim):
        a, b = pair
        print("\n" + "-" * 78)
        print(f"{stake}: {claim}  —  ({a} × {b})")
        if cards[a]["sig"] is None or cards[b]["sig"] is None:
            print("  одна из осей не выучена — вердикт невозможен")
            return None
        d = metric(a, b)
        ka, kb = cards[a]["k"], cards[b]["k"]
        da = cards[a]["sig_sig"]["directed"]
        db = cards[b]["sig_sig"]["directed"]
        print(f"  {a}: {_topology(cards[a])}")
        print(f"  {b}: {_topology(cards[b])}")
        print(f"  koopman-дистанция = {d:.3f};  k: {ka} vs {kb};  "
              f"направленность: {da:.2f} vs {db:.2f}")
        return {"pair": [a, b], "dist": d, "k": [ka, kb],
                "directed": [da, db]}

    s6 = pair_report(PAIR_S6, "S6", "разные длины лестниц -> разные карточки")
    s7 = pair_report(PAIR_S7, "S7", "тумблер ⇄ против стока -> разная форма")

    # --- JSON (подписи + матрица дистанций + пары ставок) ---
    out = {
        "budget": BUDGET,
        "cards": {k: {"key": k, "k": cards[k]["k"],
                      "koopman": cards[k]["sig"],
                      "sig": cards[k]["sig_sig"]} for k in DYNAMIC},
        "koopman_weights": w,
        "distances": {a: {b: metric(a, b) for b in valid} for a in valid},
        "stakes": {"S6": s6, "S7": s7},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cards_chem01.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nзаписано {os.path.relpath(path, ROOT)} "
          f"({len(valid)} карточек, боевая полка не тронута)")


if __name__ == "__main__":
    main()
