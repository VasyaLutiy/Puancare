"""isomorph — механический изоморф анкеты: переименовать все слова мира так,
чтобы ЛЕКСИКОГРАФИЧЕСКИЙ ПОРЯДОК ОБРАТИЛСЯ, а структура (порядок строк,
позиции в списках, вероятности) осталась байт-в-байт той же.

Обращение глобального порядка обращает порядок в любом подмножестве —
одной картой закрываются все sorted() ядра (res_a, actions, атрибуты).
Ключевые слова анкеты и служебные токены движка не переименовываются.

Держи-аут F2: изоморфная жизнь обязана дать бит-в-бит ту же форму.

    python3 experiments/f2_isomorph/isomorph.py <in.yaml> <out.yaml> [map.json]
    build_map(texts) / apply_map(text, m) — для троек (общий словарь!).
"""

import json
import re
import sys

KEYWORDS = {
    # структура анкеты
    "мир", "объекты", "зовутся", "видимое", "скрытое", "ловушки", "ловушка",
    "гарантии", "действия", "динамика", "каждый_шаг", "вмешательство",
    "экзамены", "объектов", "сущности", "штук", "состояния", "связаны_с",
    "выбираешь", "фиксировано",
    # грамматика правил
    "если", "то", "иначе", "и", "с", "минимум", "шанс_стать", "зонд",
    "вопрос", "откатом", "состояние", "связанный_отсутствует",
    # эффекты и служебные исходы движка (worldkit)
    "исчезает", "каскад", "починка", "ок", "нет_такого_объекта",
    "нет_такой_сущности", "нет_такого_действия", "ничего_не_произошло",
    "непонятные_параметры", "исчез", "создан", "сущность",
}

TOKEN_RE = re.compile(r"[а-яё][а-яё0-9_]*", re.IGNORECASE)
RU = [chr(0x0430 + i) for i in range(30)]        # а..щ+ — строго возрастают


def _tokens(text):
    toks = set()
    for line in text.splitlines():
        line = line.split("#")[0]
        if ":" in line and not line.lstrip().startswith("- "):
            key = line.split(":", 1)[0].strip().lstrip("- ")
            # имя мира — косметика; ключи-слова оставляет KEYWORDS
            toks.update(TOKEN_RE.findall(line))
        else:
            toks.update(TOKEN_RE.findall(line))
    return {t for t in toks if t not in KEYWORDS and not t.isdigit()}


def build_map(texts):
    """Общая карта переименования для НЕСКОЛЬКИХ файлов (тройка обязана
    делить словарь — включая слова, встречающиеся не во всех файлах)."""
    toks = set()
    for t in texts:
        toks |= _tokens(t)
    srt = sorted(toks)
    n = len(srt)
    if n > len(RU) * len(RU):
        raise ValueError(f"слов больше {len(RU)**2}: {n}")
    m = {}
    for i, tok in enumerate(srt):
        r = n - 1 - i                            # обращённый ранг
        pref = RU[r // len(RU)] + RU[r % len(RU)]
        m[tok] = f"{pref}_{tok}"
    return m


def apply_map(text, m):
    def sub(match):
        return m.get(match.group(0), match.group(0))
    out = []
    for raw in text.splitlines(keepends=True):
        if "#" in raw:
            code, _, comment = raw.partition("#")
            out.append(TOKEN_RE.sub(sub, code) + "#" + comment)
        else:
            out.append(TOKEN_RE.sub(sub, raw))
    return "".join(out)


def check_reversal(m):
    """Порядок образов обязан быть обращением порядка прообразов."""
    srt = sorted(m)
    imgs = [m[t] for t in srt]
    assert imgs == sorted(imgs, reverse=True), "обращение порядка сломано"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)
    src = open(sys.argv[1], encoding="utf-8").read()
    m = build_map([src])
    check_reversal(m)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.write(apply_map(src, m))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"переименовано слов: {len(m)}")
