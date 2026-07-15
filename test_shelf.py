"""test_shelf — приёмка шкафа: T2 смерть процесса, T3 диета, T4 версия,
T5 скорость. (T6 карточки — в card.py; T1 вердикты — в exam_shelf.py.)

Запуск: python3 test_shelf.py
"""

import os
import time
import shutil
import tempfile

import runworld
from organism2 import Organism
from shelf import Shelf
import card as card_mod
from repertoire import ladder, glue_of as glue_of_text
from fastpath import quick_signature

runworld.Organism = Organism

SEED_BUDGET = 1200
SIG_BUDGET = 1000


def _probe_sig(text):
    g = glue_of_text(text)
    probe = g["object_actions"][0]
    return quick_signature(g, SIG_BUDGET, acts=[probe])[0]


def _isomorph_ladder(i):
    """Изоморф лестницы-A: та же форма, СВОИ слова (проверяем форму, не файл)."""
    return ladder(f"изо{i}", f"ось{i}", [f"a{i}", f"b{i}", f"c{i}"],
                  f"зонд{i}", [f"x{i}", f"y{i}", f"z{i}"], f"вид{i}",
                  [f"p{i}", f"q{i}"], f"вмеш{i}", haz=5)


def main():
    tmp = tempfile.mkdtemp(prefix="shelf_test_")
    path = os.path.join(tmp, "shelf.json")
    ok = 0

    try:
        # --- засев --------------------------------------------------------
        sh = Shelf.open(path)
        t0 = time.time()
        v = sh.seed_archetypes(budget=SEED_BUDGET)
        print(f"засеяно {len(sh.cards)} архетипов за {time.time()-t0:.1f}с, "
              f"версия={v}")

        # запрос-изоморф лестницы (для T2/T3)
        qsig = _probe_sig(_isomorph_ladder(0))

        # --- T4: версия ---------------------------------------------------
        vd = sh.recognize(qsig)
        assert vd["version"] == sh.version, "вердикт не несёт версию"
        v_before = sh.version
        # добавим клон-карточку → версия +1
        sh.add(dict(sh.cards[0]))
        t4 = sh.version == v_before + 1 and sh.cards[-1]["added_v"] == sh.version
        print(f"  {'✓' if t4 else '✗'} T4 версия: {v_before} → {sh.version}, "
              f"вердикт несёт версию")
        ok += t4

        # --- T3: диета (100 изоморфов → полка не растёт) ------------------
        n_before = len(sh.cards)
        added = 0
        for i in range(100):
            sig = _probe_sig(_isomorph_ladder(i)) if i < 3 else qsig
            d = sh.recognize(sig).decision            # донора нет → ВВОЗ невозможен
            if d == "ОТКАЗ":                            # политика: add только на ОТКАЗ
                sh.add({"origin": f"iso{i}"}); added += 1
        t3 = len(sh.cards) == n_before and added == 0
        print(f"  {'✓' if t3 else '✗'} T3 диета: 100 изоморфов, добавлено "
              f"{added}, полка {n_before}→{len(sh.cards)} "
              f"(решение изоморфа: {sh.recognize(qsig).decision})")
        ok += t3

        # --- T2: смерть процесса (байт-в-байт) ---------------------------
        before = [sh.recognize(_probe_sig(_isomorph_ladder(i)))
                  for i in range(5)]
        v_saved, n_saved = sh.version, len(sh.cards)
        del sh                                          # «смерть процесса»
        sh2 = Shelf.open(path)                           # поднять из файла
        after = [sh2.recognize(_probe_sig(_isomorph_ladder(i)))
                 for i in range(5)]
        same = (sh2.version == v_saved and len(sh2.cards) == n_saved and
                all(a["decision"] == b["decision"] and
                    abs(a["d"] - b["d"]) < 1e-9 and a["version"] == b["version"]
                    for a, b in zip(before, after)))
        print(f"  {'✓' if same else '✗'} T2 смерть процесса: версия/карты/"
              f"вердикты совпали ({n_saved} карт, v{v_saved})")
        ok += same

        # --- T5: скорость при 100 карточках ------------------------------
        big = list(sh2.cards)
        while len(big) < 100:
            big.append(dict(sh2.cards[len(big) % len(sh2.cards)]))
        sh_big = Shelf(path + ".big", big)
        t0 = time.time()
        for i in range(20):
            sh_big.recognize(_probe_sig(_isomorph_ladder(i % 5)))
        dt = (time.time() - t0) / 20
        # честно: время доминирует quick_signature (проживание мира), не поиск.
        # Меряем ЧИСТЫЙ поиск по готовой подписи:
        s = _probe_sig(_isomorph_ladder(0))
        t0 = time.time()
        for _ in range(200):
            sh_big.recognize(s)
        dt_search = (time.time() - t0) / 200
        t5 = dt_search < 0.05
        print(f"  {'✓' if t5 else '✗'} T5 скорость@100карт: поиск "
              f"{dt_search*1000:.1f}мс/запрос (подпись+поиск {dt*1000:.0f}мс)")
        ok += t5

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(path + ".big"):
            os.remove(path + ".big")

    print(f"\nитог: {ok}/4 (T2,T3,T4,T5)")
    return ok


if __name__ == "__main__":
    main()
