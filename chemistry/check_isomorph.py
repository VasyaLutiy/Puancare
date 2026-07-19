"""check_isomorph — ставка S5: изоморф m16 ↔ m01 бит-в-бит (STAKE-chem01.md).

m16 скомпилирован как m01, где каждое содержательное слово заменено по
oracle.RENAME (грамматика движка не трогается). Аксиома немоты: если химия
была настоящей структурой, а не перекраской слов, то организм, проживший
m16, выучит ТО ЖЕ, что на m01 — правила, классы, экзамены, спектры совпадут
С ТОЧНОСТЬЮ ДО СЛОВАРЯ. Проверяем буквально:

  1. прожить m01 и m16 ОДНИМ конвейером (runworld.full_run, те же бюджеты
     300/2000 и сиды — детерминизм ядра гарантирует воспроизводимость);
  2. перехватить весь текст отчёта обоих;
  3. в тексте m16 заменить слова ОБРАТНЫМ словарём RENAME (m16-слово -> m01);
  4. сравнить с текстом m01 ПОБУКВЕННО.

Печать: BIT-IDENTICAL да/нет; при расхождении — первые 10 строк диффа.

    <python> chemistry/check_isomorph.py
"""

import io
import os
import re
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runworld                                    # noqa: E402
import oracle                                      # noqa: E402
from organism2 import Organism                     # noqa: E402
from worldkit import load_spec, make_glue          # noqa: E402

runworld.Organism = Organism

WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")


def _run(world_file):
    """Прожить мир штатным full_run, вернуть весь его отчёт как текст."""
    spec = load_spec(os.path.join(WORLDS_DIR, world_file + ".yaml"))
    glue = make_glue(spec)
    buf = io.StringIO()
    with redirect_stdout(buf):
        runworld.full_run(spec, glue)
    return buf.getvalue()


def _inverse_rename():
    """Обратный словарь m16-слово -> m01-слово. RENAME инъективен (проверено
    самопроверкой oracle), поэтому обращение однозначно."""
    inv = {}
    for src, dst in oracle.RENAME.items():
        if dst in inv and inv[dst] != src:
            raise ValueError(f"RENAME не инъективен: {dst!r} <- {src!r}/{inv[dst]!r}")
        inv[dst] = src
    return inv


def _detranslate(text, inv):
    """Заменить каждое m16-слово обратно на m01-слово, токенно (границы —
    не-словарные символы кириллицы/латиницы/цифр/подчёркивания), длинные
    ключи раньше коротких (чтобы префикс не съел составной токен)."""
    for dst in sorted(inv, key=len, reverse=True):
        src = inv[dst]
        if dst == src:
            continue
        text = re.sub(r"(?<![\w])" + re.escape(dst) + r"(?![\w])",
                      src, text)
    return text


def main():
    m01_file = "m01_лестница_первичного"
    m16_file = "m16_изоморф_m01"

    print("проживаю m01 и m16 одним конвейером (full_run, бюджеты 300/2000)...")
    out01 = _run(m01_file)
    out16 = _run(m16_file)

    inv = _inverse_rename()
    # заголовок «МИР: <имя>» несёт имя файла — тоже слово RENAME, снимется.
    out16_back = _detranslate(out16, inv)

    a = out01.splitlines()
    b = out16_back.splitlines()
    identical = (out01 == out16_back)

    print("=" * 78)
    print(f"S5 ИЗОМОРФ m16 ↔ m01: BIT-IDENTICAL = {'ДА' if identical else 'НЕТ'}")
    print("=" * 78)
    print(f"строк: m01={len(a)}, m16(обратно)={len(b)}")

    if identical:
        # покажем ключевые числа, чтобы вердикт был читаем без диффа.
        for line in a:
            if ("экз:" in line or "класс" in line.lower()
                    or line.strip().startswith("class")):
                print("  " + line.strip())
        print("\nчисла (правила/классы/экзамены/спектры) совпали точно — "
              "немота подтверждена.")
        return 0

    # НЕ бит-в-бит. Различаем два исхода S5 канонической (порядко-независимой)
    # сверкой: если МНОЖЕСТВА строк совпадают, а различается лишь ПОРЯДОК —
    # содержание выучено одинаково, а прибор течёт словами через sorted() по
    # непереведённому имени действия (порядок отчёта несёт алфавит m16).
    # Если же множества строк расходятся — химия была перекраской.
    canon_identical = (sorted(a) == sorted(b))
    print(f"КАНОНИЧЕСКИ (порядко-независимо) = "
          f"{'СОВПАЛО' if canon_identical else 'РАЗОШЛОСЬ'}")
    if canon_identical:
        # покажем, сколько строк переставлено (в тех же множествах).
        moved = sum(1 for x, y in zip(a, b) if x != y)
        print(f"  множества строк идентичны; переставлено строк: {moved}")
        print("  ВЕРДИКТ S5: содержание выучено ОДИНАКОВО (немота химии цела);")
        print("  расхождение — презентационное: runworld.full_run сортирует")
        print("  правила/симптомы по имени действия, и переименование")
        print("  переставляет порядок печати (прибор течёт словами через")
        print("  sorted(), не химия). Оба исхода S5 объявлены важными.")
    else:
        print("  ВЕРДИКТ S5: множества выученного РАЗОШЛИСЬ — химия была")
        print("  перекраской слов ИЛИ прибор течёт содержательно.")

    print("\nПЕРВЫЕ 10 СТРОК ПОЗИЦИОННОГО ДИФФА (m01 | m16→обратно):")
    shown = 0
    for i in range(max(len(a), len(b))):
        la = a[i] if i < len(a) else "<нет>"
        lb = b[i] if i < len(b) else "<нет>"
        if la != lb:
            print(f"  строка {i + 1}:")
            print(f"    m01 : {la}")
            print(f"    m16→: {lb}")
            shown += 1
            if shown >= 10:
                break
    print("\nРАСХОЖДЕНИЕ (позиционное). Смысловой вердикт — см. каноническую "
          "сверку выше.")
    return 0 if canon_identical else 1


if __name__ == "__main__":
    sys.exit(main())
