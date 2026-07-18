"""forge.py — детерминированный компилятор анкет CHEM-01.

ГЛАВНЫЙ ИНВАРИАНТ (смысл этого файла):
    Секции `действия` и `динамика` выводятся ТОЛЬКО из таблиц oracle.py
    (OUTCOME/SPECIAL для действий, MOTIFS/TEMPO для динамики). Рука
    (oracle.WORLDS) выбирает исключительно ЧТО существует в мире и какими
    словами оно зовётся: имена объектов/атрибутов/значений, набор проб,
    ловушки, вмешательство, экзамен. Ни один исход и ни один переход
    здесь не задаётся напрямую — forge только переводит правду природы
    в грамматику ШАБЛОН.yaml/worldkit.parse_text.

    ОГОВОРКА (m11): исходы СУЩНОСТНЫХ действий (@сущность: осмотреть_цепь,
    добавить_ингибитор) — это разметка сущностного каскада, взятая рукой из
    WORLDS, а НЕ из OUTCOME. Инвариант «исходы из таблиц» касается проб и
    динамики; теги [каскад]/[исчезает]/[починка] сущностей — не химия проб.

Движок читает НЕ настоящий YAML, а мини-подмножество (worldkit.parse_text):
жёсткий отступ в 2 пробела, токены без пробелов в ловушках/гарантиях/
динамике, встрочные списки [a, b]. Эмиттер держит этот диалект точно.

    python3 chemistry/forge.py                 # записать 16 анкет
    python3 chemistry/forge.py --manifest      # + worlds.frozen.sha256
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle  # noqa: E402

WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOUN = "реактив"   # как зовётся один объект (для m16 переименуется)


# ------------------------------------------------------------- сборка правил

def _probe_rules(world):
    """Действия-пробы -> decision list из таблицы OUTCOME (§2).
    Строка на каждое значение с исходом != дефолт; финал 'иначе <дефолт>'."""
    out = {}
    ось = world["ось"]
    for pid in world["пробы"]:
        act = oracle.PROBES[pid]["действие"]
        default = oracle.PROBES[pid]["дефолт"]
        rules = []
        for знач, группа in world["значения"]:
            res = oracle.OUTCOME.get((группа, pid), default)
            if res != default:
                rules.append(([(ось, знач)], res, []))
        rules.append(("иначе", default, []))
        out[act] = rules
    return out


def _special_rules(world):
    """Мотив-специфичные действия (m08/m10/m11) -> decision list из SPECIAL.
    Читают ось напрямую; последнее значение оси уходит в 'иначе'."""
    out = {}
    ось = world["ось"]
    table = oracle.SPECIAL[world["ключ"]]
    values = [v for v, _ in world["значения"]]
    for act, by_value in table.items():
        rules = []
        for знач in values[:-1]:
            res, eff = by_value[знач]
            rules.append(([(ось, знач)], res, list(eff)))
        res, eff = by_value[values[-1]]
        rules.append(("иначе", res, list(eff)))
        out[act] = rules
    return out


def _entity_rules(world):
    """Действия над сущностями (@сущность) — только m11. Структура задана
    рукой в WORLDS (это разметка сущностного каскада, не химия исходов)."""
    out = {}
    for act, spec in (world["сущ_действия"] or {}).items():
        rules = []
        for cond, res, eff in spec:
            if cond == "иначе":
                rules.append(("иначе", res, list(eff)))
            elif cond == "связанный_отсутствует":
                rules.append(("связанный_отсутствует", res, list(eff)))
            else:  # dict {"состояние": "..."}
                (a, v), = cond.items()
                rules.append(([(a, v)], res, list(eff)))
        out[act] = rules
    return out


def _dynamics(world):
    """Секция динамики -> из MOTIFS/TEMPO. Статические миры (мотив None
    или пустые переходы) — пустой список, секция не печатается."""
    мотив = world["мотив"]
    if not мотив:
        return []
    values = [v for v, _ in world["значения"]]
    ось = world["ось"]
    rules = []
    for i, j, темп in oracle.MOTIFS[мотив]["переходы"]:
        rules.append((ось, values[i], values[j], oracle.TEMPO[темп]))
    return rules


# ----------------------------------------------------------------- эмиттер

def _emit(world, rename=None):
    """Печатает анкету мира в мини-YAML движка. rename — карта слов для
    изоморфа m16 (грамматические ключи не переименовываются)."""
    r = (lambda t: rename.get(t, t)) if rename else (lambda t: t)
    L = []
    ось = world["ось"]
    значения = [v for v, _ in world["значения"]]

    L.append(f"мир: {r(world['файл'])}")
    L.append("объекты:")
    L.append(f"  зовутся: {r(NOUN)}")
    L.append("  видимое:")
    for a, vals in world["видимое"].items():
        L.append(f"    {r(a)}: [{', '.join(r(v) for v in vals)}]")
    L.append("  скрытое:")
    L.append(f"    {r(ось)}: [{', '.join(r(v) for v in значения)}]")
    L.append("  ловушки:")
    for (ha, hv, va, vv, p) in world["ловушки"]:
        L.append(f"    - если {r(ha)}={r(hv)} то {r(va)}={r(vv)} ({p}%)")
    L.append("  гарантии:")
    for знач in значения:
        L.append(f"    - минимум 2 объектов с {r(ось)}={r(знач)}")

    ent = world["сущности"]
    if ent:
        L.append("сущности:")
        L.append(f"  зовутся: {r(ent['зовутся'])}")
        L.append(f"  штук: {ent['штук']}")
        L.append(f"  состояния: [{', '.join(r(s) for s in ent['состояния'])}]")
        # связаны_с: '1 (ось=знач)' — фильтр переименуем токен за токеном.
        сс = ent["связаны_с"]
        if "(" in сс:
            n, rest = сс.split("(", 1)
            key, val = rest.rstrip(")").split("=")
            сс = f"{n.strip()} ({r(key.strip())}={r(val.strip())})"
        L.append(f"  связаны_с: {сс}")

    # ---- действия: пробы + спец + сущностные ----
    L.append("действия:")
    rules = {}
    if world["спец"]:
        rules.update(_special_rules(world))
    else:
        rules.update(_probe_rules(world))
    for act, rlist in rules.items():
        L.append(f"  {r(act)}:")
        for cond, res, eff in rlist:
            L.append("    - " + _rule_line(cond, res, eff, r))
    for act, rlist in _entity_rules(world).items():
        L.append(f"  {r(act)} @сущность:")
        for cond, res, eff in rlist:
            L.append("    - " + _rule_line(cond, res, eff, r))

    # ---- динамика (только у мотивных миров) ----
    dyn = _dynamics(world)
    if dyn:
        L.append("динамика:")
        L.append("  каждый_шаг:")
        for attr, frm, to, pct in dyn:
            L.append(f"    - если {r(attr)}={r(frm)} то "
                     f"шанс_стать {r(to)} ({pct}%)")

    # ---- вмешательство (R5) ----
    name, chosen, fixed = world["вмешательство"]
    L.append("вмешательство:")
    L.append(f"  {r(name)}:")
    L.append(f"    выбираешь: [{', '.join(r(a) for a in chosen)}]")
    L.append("    фиксировано: "
             + ", ".join(f"{r(a)}={r(v)}" for a, v in fixed))

    # ---- экзамен ----
    зонд, вопрос = world["экзамен"]
    L.append("экзамены:")
    L.append(f"  - зонд {r(зонд)} вопрос {r(вопрос)}")

    L.append(f"объектов: {world['объектов']}")
    return "\n".join(L) + "\n"


def _rule_line(cond, res, eff, r):
    """Одна строка decision list: 'если a=v то исход [эффекты]' / 'иначе …'."""
    tag = f" [{', '.join(eff)}]" if eff else ""
    if cond == "иначе":
        return f"иначе {r(res)}{tag}"
    if cond == "связанный_отсутствует":
        return f"если связанный_отсутствует то {r(res)}{tag}"
    body = " и ".join(f"{r(a)}={r(v)}" for a, v in cond)
    return f"если {body} то {r(res)}{tag}"


# ------------------------------------------------------------------ вывод

def build():
    """Возвращает [(имя_файла, текст)] для 16 анкет (15 + изоморф m16)."""
    files = []
    for w in oracle.WORLDS:
        files.append((w["файл"] + ".yaml", _emit(w)))
    # m16 — изоморф m01 бит-в-бит по структуре, слова по RENAME.
    m01 = oracle.WORLD_BY_KEY["m01"]
    files.append((oracle.ISOMORPH["файл"] + ".yaml",
                  _emit(m01, rename=oracle.RENAME)))
    return files


def write_all():
    os.makedirs(WORLDS_DIR, exist_ok=True)
    paths = []
    for name, text in build():
        path = os.path.join(WORLDS_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        paths.append(path)
        print(f"написан {os.path.relpath(path, REPO)}")
    return paths


def write_manifest(paths):
    lines = []
    for path in sorted(paths):
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        lines.append(f"{digest}  {os.path.relpath(path, REPO)}")
    man = os.path.join(WORLDS_DIR, "..", "worlds.frozen.sha256")
    man = os.path.normpath(man)
    with open(man, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"манифест {os.path.relpath(man, REPO)} ({len(lines)} файлов)")


if __name__ == "__main__":
    paths = write_all()
    if "--manifest" in sys.argv[1:]:
        write_manifest(paths)
