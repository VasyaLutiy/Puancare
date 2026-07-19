"""check_isomorph_canon — S5, каноничный судья поверх check_isomorph.

Первый прогон показал: все ЧИСЛА совпали, разошёлся только ПОРЯДОК строк
и элементов в строках — печать runworld сортирует по именам действий, а
алфавитный порядок слов у изоморфа другой. Порядок печати — не знание.
Канон: (1) внутри строки элементы, разделённые '  ' и ', ', сортируются;
(2) сами строки сравниваются как мультимножество. Совпадение канонов =
организм выучил то же самое; несовпадение = настоящая утечка.
Сырые отчёты сохраняются рядом (iso_m01.txt, iso_m16back.txt) для вскрытия.
"""

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_isomorph import _run, _inverse_rename, _detranslate  # noqa: E402


def canon_line(line):
    parts = re.split(r"(?:\s\s+|, )", line.strip())
    return " | ".join(sorted(p.strip() for p in parts if p.strip()))


def main():
    out01 = _run("m01_лестница_первичного")
    out16 = _run("m16_изоморф_m01")
    out16b = _detranslate(out16, _inverse_rename())

    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, "iso_m01.txt"), "w", encoding="utf-8").write(out01)
    open(os.path.join(here, "iso_m16back.txt"), "w", encoding="utf-8").write(out16b)

    raw = (out01 == out16b)
    c01 = Counter(canon_line(l) for l in out01.splitlines() if l.strip())
    c16 = Counter(canon_line(l) for l in out16b.splitlines() if l.strip())
    canon = (c01 == c16)

    print(f"S5 сырое побуквенное: {'ДА' if raw else 'НЕТ'}")
    print(f"S5 канон (мультимножества строк/элементов): {'ДА' if canon else 'НЕТ'}")
    if not canon:
        print("\nтолько в m01:")
        for l, n in (c01 - c16).items():
            print(f"  x{n}  {l}")
        print("только в m16→обратно:")
        for l, n in (c16 - c01).items():
            print(f"  x{n}  {l}")
    return 0 if canon else 1


if __name__ == "__main__":
    sys.exit(main())
