"""
GD1 — «ОБОБЩЕНИЕ» РАСПОЗНАВАНИЯ: интеллект или артефакт грануляции сигнатуры?

Claim M1: «агент формирует КОНЕЧНОЕ число категорий из растущего потока → обобщает».
Атака: сигнатура = (виды ресурсов, виды настроек, статус) над КРОШЕЧНЫМ словарём kind'ов.
  → кодомен сигнатуры мал → «плато» ГАРАНТИРОВАНО для ЛЮБОГО входа (хеш в маленький кодомен),
    это не обучение.
  → сделай сигнатуру реалистично-тонкой (деталь инстанса) → категорий ≈ инстансов (мемоизация).
Вывод-проверка: число категорий — функция МОЕГО выбора грануляции, не свойство данных/агента.
"""

import random
import sys

from devops_agent.v6.recognizer import RES_KINDS, SET_KINDS, signature

N = 200
_LABELS = ["db", "cache", "queue", "worker", "gw", "proxy", "stream", "lb"]


def _codomain():
    return (2 ** len(RES_KINDS)) * (2 ** len(SET_KINDS)) * 2   # все подмн-ва kind'ов × статус


def _stream(n, seed=1):
    rng = random.Random(seed)
    res, sett = list(RES_KINDS), list(SET_KINDS)
    out = []
    for i in range(n):
        out.append({
            "id": f"e{i}",
            "label": rng.choice(_LABELS),
            "resources": rng.sample(res, rng.randint(0, len(res))),
            "settings": rng.sample(sett, rng.randint(0, len(sett))),
            "status": rng.random() < 0.8,
            "deps": [f"e{j}" for j in rng.sample(range(i), min(i, rng.randint(0, 3)))] if i else [],
            "version": rng.randint(1, 100),     # реалистичная деталь инстанса (версия/имя/значение)
        })
    return out


def _coarse(f):
    return signature(f)                                            # как в v6 (виды + статус)


def _fine(f):
    return (signature(f), len(f["deps"]), f["label"])              # +связи +ярлык


def _finest(f):
    return (signature(f), len(f["deps"]), f["label"], f["version"])  # +деталь инстанса (реализм)


def main() -> None:
    print("=== GD1: «обобщение» — интеллект или артефакт грануляции сигнатуры? ===\n")
    stream = _stream(N)
    cod = _codomain()
    rows = [("coarse (как в v6: виды+статус)", _coarse),
            ("fine (+связи+ярлык)", _fine),
            ("finest (+деталь инстанса)", _finest)]
    counts = {}
    for name, fn in rows:
        c = len({fn(f) for f in stream})
        counts[name] = c
        print(f"  {name:34} → {c:3d} категорий из {N} инстансов")

    print(f"\n  кодомен coarse-сигнатуры = {cod} → категорий ВСЕГДА ≤ {cod}, при ЛЮБОМ потоке.")
    print("  (то есть «плато 5 из 30» в M1 — гарантировано малым кодоменом, а не выучено.)")

    coarse_c = counts["coarse (как в v6: виды+статус)"]
    finest_c = counts["finest (+деталь инстанса)"]
    print("=" * 64)
    if coarse_c <= cod and finest_c >= 0.8 * N:
        print("ВЕРДИКТ GD1: ПАДЁТ ✗ — «обобщение» — артефакт грануляции сигнатуры, не интеллект.")
        print(f"  coarse даёт ≤{cod} (хеш в крошечный кодомен → плато тривиально);")
        print(f"  реалистично-тонкая сигнатура → {finest_c}/{N} ≈ мемоизация (категорий ≈ инстансов).")
        print("  Число категорий = МОЙ выбор грубости хеша, а не свойство данных. Нужен FUZZY-матч")
        print("  по СХОДСТВУ (расстояние), а не точный хеш — иначе это либо тривиальное плато, либо зубрёжка.")
    else:
        print(f"ВЕРДИКТ GD1: устоял? coarse={coarse_c} finest={finest_c} cod={cod} — перепроверить.")


if __name__ == "__main__":
    main()
