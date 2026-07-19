"""forge_mol — компилятор МОЛЕКУЛ в анкеты миров (этап 2 CHEM).

Молекула (имя + 1-2 группы из oracle.MOLECULE_GROUPS) -> мини-YAML движка.
НИКАКОЙ перекраски: пробы и исходы берутся стандартным словарём oracle §2
(OUTCOME/PROBES) с учётом КОМБИНИРОВАНИЯ проб (§6) на молекуле из нескольких
групп. Общий словарь исходов — контракт переносимости карт (T2), потому имя
молекулы и имя порции — единственные слова руки; всё поведение из таблиц.

Что делает компилятор (ORACLE.md §6):
  * скрытая ОСЬ на каждую ДИНАМИЧЕСКУЮ группу молекулы (среда этапа 2:
    DYNAMIC_IN_AIR); значения оси — стандартные слова мотива;
  * СТАТИЧНАЯ группа — без оси, её положительные пробы дают КОНСТАНТНЫЕ по
    осям исходы (шум организму, проверка B5);
  * ДЕЙСТВИЯ-ПРОБЫ: читающие ось (2-4 на ось, из OUTCOME с комбинированием)
    + константные пробы статичной группы (шум);
  * ЭКЗАМЕН на каждую ось: пара зонд->вопрос, выбранная ФАНТОМ-ДЕТЕКТОРОМ
    на КОМБИНИРОВАННЫХ исходах — полностью детерминированная, если такая
    пара существует, иначе лучшая частичная (доля различённых пар значений);
  * ВМЕШАТЕЛЬСТВО «обновить»: свежая порция с осями в начальных значениях;
  * ЛОВУШКА: видимый атрибут коррелирует с первой осью на 80% (слова
    видимого — свои, детерминированные, не химические);
  * ГАРАНТИИ: >=2 порции на каждое значение каждой оси; 8 порций.

Фантом-детектор осей — та же логика, что oracle._self_check §4 (совместная
сигнатура действий различает все пары значений оси), но на КОМБИНИРОВАННЫХ
исходах молекулы: со-группа может замаскировать пробу (пузырьки у спирта и
у фенола сливаются) — тогда ось нечитаема и мир бракуется как фантом.
"""

import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import oracle  # noqa: E402

# Видимый скелет мира — слова forge, детерминированные и не химические
# (как _ВИД этапа 1 тёмный/светлый/пёстрый, но иные, чтобы не путать полки).
_VIS = {"оттенок": ["сизый", "охристый", "мшистый"],
        "крап": ["есть", "нет"]}
_NOUN_DEFAULT = "порция"
MAX_READERS = 4          # 2-4 читающих пробы на ось (§6)
MAX_NOISE = 2            # сколько шумовых констант-проб добавить (B5)


# ------------------------------------------------------- разбор молекулы
def analyze(groups):
    """группы -> (динамические оси, статические ключи). Оси — записи
    DYNAMIC_IN_AIR; статики — внутренние ключи проб (None = немая группа)."""
    gs = [oracle.normalize_group(g) for g in groups]
    axes = [oracle.DYNAMIC_IN_AIR[g] for g in gs if g in oracle.DYNAMIC_IN_AIR]
    static_keys = [oracle.STATIC_KEY[g] for g in gs
                   if g in oracle.STATIC_KEY and oracle.STATIC_KEY[g] is not None]
    return gs, axes, static_keys


# ------------------------------------------ комбинированный исход пробы
def _combined(pid, assign, axes, static_keys):
    """Исход пробы pid на порции при назначении осей assign (axis_idx->val_idx),
    с учётом статических групп. Комбинирование — oracle.combine_probe (§6)."""
    default = oracle.PROBES[pid]["дефолт"]
    outs = [oracle.outcome_of(axes[ai]["значения"][vi][1], pid)
            for ai, vi in assign.items()]
    outs += [oracle.outcome_of(sk, pid) for sk in static_keys]
    return oracle.combine_probe(outs, default)


def _separated(pid, ai, axes, static_keys):
    """Множество пар (i<j) значений оси ai, которые проба pid РАЗЛИЧАЕТ хотя бы
    в одном контексте прочих осей (комбинированным исходом)."""
    others = [x for x in range(len(axes)) if x != ai]
    ka = len(axes[ai]["значения"])
    ranges = [range(len(axes[x]["значения"])) for x in others]
    pairs = set()
    for combo in itertools.product(*ranges):
        base = dict(zip(others, combo))
        for i in range(ka):
            for j in range(i + 1, ka):
                oi = _combined(pid, {**base, ai: i}, axes, static_keys)
                oj = _combined(pid, {**base, ai: j}, axes, static_keys)
                if oi != oj:
                    pairs.add((i, j))
    return pairs


def _all_pairs(ka):
    return {(i, j) for i in range(ka) for j in range(i + 1, ka)}


def _is_constant(pid, axes, static_keys):
    """Даёт ли проба один и тот же исход на ВСЕХ совместных состояниях осей?"""
    ranges = [range(len(ax["значения"])) for ax in axes]
    seen = set()
    for combo in itertools.product(*ranges):
        seen.add(_combined(pid, dict(enumerate(combo)), axes, static_keys))
        if len(seen) > 1:
            return False
    return True


# ----------------------------------------------- выбор проб и экзаменов
def _readers_for_axis(ai, axes, static_keys):
    """Пробы-читатели оси ai: {pid: separated}. Порядок проб P1..P14."""
    out = {}
    for pid in oracle.PROBES:
        sep = _separated(pid, ai, axes, static_keys)
        if sep:
            out[pid] = sep
    return out


def _cover(readers, ka):
    """Жадный набор проб-читателей, покрывающий все пары значений (min 2,
    max MAX_READERS). Сорт по покрытию убыв., затем по номеру пробы."""
    need = _all_pairs(ka)
    ranked = sorted(readers, key=lambda p: (-len(readers[p]), int(p[1:])))
    chosen, covered = [], set()
    for p in ranked:
        if len(chosen) >= MAX_READERS:
            break
        if not need - covered:                    # уже покрыто — добираем до 2
            if len(chosen) >= 2:
                break
        if readers[p] - covered or len(chosen) < 2:
            chosen.append(p)
            covered |= readers[p]
    return chosen, covered


def _exam_pair(chosen, readers, ka):
    """Пара (зонд, вопрос) из читателей, максимизирующая покрытие пар. Полная,
    если объединение = все пары. Возвращает (зонд, вопрос, sharpness)."""
    total = len(_all_pairs(ka)) or 1
    best = None
    for a in chosen:
        for b in chosen:
            if a == b:
                continue
            union = readers[a] | readers[b]
            s = len(union) / total
            cand = (s, -int(a[1:]), -int(b[1:]), a, b)
            if best is None or cand > best:
                best = cand
    if best is None:                              # <2 читателей — зонд один
        a = chosen[0]
        return a, a, len(readers[a]) / total
    s, _, _, a, b = best
    return a, b, s


# ------------------------------------------------------------ санитайзер
def sanitize(word, fallback):
    """Слово руки -> безопасный токен диалекта (кириллица/латынь/цифры/_)."""
    w = "".join(ch if (ch.isalnum() or ch == "_") else "_"
                for ch in str(word).strip())
    w = w.strip("_")
    return w or fallback


# --------------------------------------------------------------- сборка
def build(molecule):
    """молекула {имя, группы} -> словарь плана мира. Бросает ValueError, если
    оракул забраковал набор групп или ось фантомна (нечитаема пробами)."""
    имя = molecule.get("имя", "")
    groups = molecule.get("группы", [])
    ok, why = oracle.judge_molecule(groups)
    if not ok:
        raise ValueError(why)

    gs, axes, static_keys = analyze(groups)
    if not axes:                                   # страховка (судья ловит §D2)
        raise ValueError("пустышка: нет динамической группы")

    name = sanitize(имя, "молекула")
    noun = _NOUN_DEFAULT

    # --- читатели, покрытие, фантом-детектор, экзамены (на комбинир. исходах) ---
    world_probes, exams, axinfo = [], [], []
    phantom_bad = None
    for ai, ax in enumerate(axes):
        ka = len(ax["значения"])
        readers = _readers_for_axis(ai, axes, static_keys)
        chosen, covered = _cover(readers, ka)
        if covered != _all_pairs(ka):
            phantom_bad = (f"фантом: ось {ax['ось']} нечитаема "
                           f"(со-группа маскирует пробы)")
        зонд, вопрос, sharp = (_exam_pair(chosen, readers, ka)
                               if chosen else (None, None, 0.0))
        world_probes += chosen
        if зонд:
            exams.append((oracle.PROBES[зонд]["действие"],
                          oracle.PROBES[вопрос]["действие"], sharp))
        axinfo.append({"ось": ax["ось"], "мотив": ax["мотив"], "k": ka,
                       "sharp": sharp, "readers": chosen})
    if phantom_bad:
        raise ValueError(phantom_bad)

    # --- шумовые константные пробы статичных групп (проверка B5) ---
    noise = []
    for sk in static_keys:
        for pid in oracle.PROBES:
            if len(noise) >= MAX_NOISE:
                break
            if pid in world_probes or pid in noise:
                continue
            if oracle.outcome_of(sk, pid) == oracle.PROBES[pid]["дефолт"]:
                continue
            if _is_constant(pid, axes, static_keys):
                noise.append(pid)
    # порядок проб — устойчивый (P1..P14), без дублей
    probes = sorted(set(world_probes) | set(noise), key=lambda p: int(p[1:]))
    # экзамен требует >=2 разных действий — доберём при нужде
    if len({оракул_act for оракул_act in probes}) < 2:
        for pid in oracle.PROBES:
            if pid not in probes:
                probes.append(pid)
                break

    text = _emit(name, noun, axes, static_keys, probes, exams)
    return {
        "name": name, "noun": noun, "text": text,
        "groups": gs, "axes": axinfo, "statics": static_keys,
        "probes": probes, "noise": noise, "exams": exams,
    }


def _probe_rules(pid, axes, static_keys):
    """Decision list пробы: правила на релевантные оси (те, чьи группы вообще
    реагируют на pid) с комбинированными исходами; финал 'иначе <дефолт>'.
    <=2 условий (осей максимум 2) — держим R1."""
    default = oracle.PROBES[pid]["дефолт"]
    rel = [ai for ai, ax in enumerate(axes)
           if any(oracle.outcome_of(g, pid) != default
                  for _, g in ax["значения"])]
    lines = []
    if not rel:
        const = _combined(pid, {}, axes, static_keys)
        if const != default:
            lines.append(f"иначе {const}")
        else:
            lines.append(f"иначе {default}")
        return lines
    ranges = [range(len(axes[ai]["значения"])) for ai in rel]
    for combo in itertools.product(*ranges):
        assign = dict(zip(rel, combo))
        out = _combined(pid, assign, axes, static_keys)
        if out == default:
            continue
        conds = " и ".join(f"{axes[ai]['ось']}={axes[ai]['значения'][vi][0]}"
                           for ai, vi in assign.items())
        lines.append(f"если {conds} то {out}")
    lines.append(f"иначе {default}")
    return lines


def _emit(name, noun, axes, static_keys, probes, exams):
    L = ["мир: " + name, "объекты:", f"  зовутся: {noun}", "  видимое:"]
    for a, vals in _VIS.items():
        L.append(f"    {a}: [{', '.join(vals)}]")
    L.append("  скрытое:")
    for ax in axes:
        vals = [v for v, _ in ax["значения"]]
        L.append(f"    {ax['ось']}: [{', '.join(vals)}]")
    # ловушка: первая ось, первое значение -> оттенок=сизый 80%
    ax0 = axes[0]
    L.append("  ловушки:")
    L.append(f"    - если {ax0['ось']}={ax0['значения'][0][0]} то "
             f"оттенок=сизый (80%)")
    L.append("  гарантии:")
    for ax in axes:
        for v, _ in ax["значения"]:
            L.append(f"    - минимум 2 объектов с {ax['ось']}={v}")

    L.append("действия:")
    for pid in probes:
        L.append(f"  {oracle.PROBES[pid]['действие']}:")
        for line in _probe_rules(pid, axes, static_keys):
            L.append(f"    - {line}")

    # динамика: переходы каждой оси (метки значений + темп из TEMPO)
    L.append("динамика:")
    L.append("  каждый_шаг:")
    for ax in axes:
        vals = [v for v, _ in ax["значения"]]
        for i, j, темп in ax["переходы"]:
            L.append(f"    - если {ax['ось']}={vals[i]} то шанс_стать "
                     f"{vals[j]} ({oracle.TEMPO[темп]}%)")

    # вмешательство: свежая порция, оси в начальных значениях
    L.append("вмешательство:")
    L.append("  обновить:")
    L.append(f"    выбираешь: [{next(iter(_VIS))}]")
    fixed = ", ".join(f"{ax['ось']}={ax['значения'][0][0]}" for ax in axes)
    L.append(f"    фиксировано: {fixed}")

    L.append("экзамены:")
    for зонд, вопрос, _ in exams:
        L.append(f"  - зонд {зонд} вопрос {вопрос}")

    L.append("объектов: 8")
    return "\n".join(L) + "\n"
