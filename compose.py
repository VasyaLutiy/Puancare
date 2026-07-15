"""compose — первый кирпич композиции: перенести карту ИСПОЛЬЗОВАНИЯ из
прожитого мира-донора в новый мир-цель, выровняв скрытые состояния ПО РОЛИ
(без слов о состояниях). Экзамен цели сдаётся связкой [свой зонд у цели] +
[карта использования, ввезённая от донора], на бюджете, где цель с нуля не
тянет карту использования.

Уступка первого кирпича: словарь ДЕЙСТВИЙ и ТОКЕНОВ у донора и цели общий
(просветить/нагрузить, пик/полка/провал, мощно/слабо/глохнет) — так
изолируется РОВНО одна новая способность: выравнивание СОСТОЯНИЙ по роли
(EM нумерует их в каждом мире по-своему). Ролевое выравнивание самих
действий/токенов — слой 2.

organism2 заморожен: только читаем поля и мёржим эпизоды через Ruleset.
Три арки на каждом бюджете:
  ГОЛАЯ          — цель как есть
  ВВОЗ БЕЗ ВЫРАВН— правила донора вклеены по СЫРЫМ меткам (c0↔c0)
  ВВОЗ+ВЫРАВН    — по РОЛИ (стационар: источник→сток)
"""

import runworld
import sig
from collections import Counter
from organism2 import Organism, Ruleset
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def glue_of(world):
    return make_glue(load_spec(f"worlds/{world}.yaml"))


def live(world, budget, curious=False):
    glue = glue_of(world)
    return runworld.live(glue, curious, budget), glue


def axis_with(org, action):
    """(индекс, ось) первой оси, содержащей действие; иначе (None, None)."""
    for i, ax in enumerate(org.axes):
        if action in ax.actions:
            return i, ax
    return None, None


def role_order(ax):
    """Канон состояний ПО РОЛИ в каскаде: по стационарной вероятности
    (источник мал → сток велик). Возвращает rank->state.

    Граница метода: если π вырожден (симметричный тумблер — оба состояния
    структурно одинаковы), ранжирование = монета. Различимость ролей
    измеряет role_gap; выравнивание честно только при заметном зазоре."""
    pi = sig.stationary(ax)
    return sorted(range(ax.k), key=lambda s: pi[s])


def role_gap(ax):
    """Различимость ролей: минимальный зазор соседних стационарных
    вероятностей. ≈0 → роли структурно неразличимы, ролевое выравнивание
    вырождается в подбрасывание монеты (см. compose_toggle)."""
    pi = sorted(sig.stationary(ax))
    return min(b - a for a, b in zip(pi, pi[1:])) if len(pi) > 1 else 1.0


def alignment(donor_ax, target_ax):
    """donor_state -> target_state по совпадению роли (ранга)."""
    od, ot = role_order(donor_ax), role_order(target_ax)
    return {od[r]: ot[r] for r in range(min(len(od), len(ot)))}


def remap_ctx(ctx, di, ti, smap):
    """Переразметка class-меток в ctx: индекс оси di->ti, состояние по smap."""
    out = set()
    for (a, v) in ctx:
        if a == f"class{di}" and isinstance(v, str) and v.startswith("c"):
            try:
                s = int(v[1:])
            except ValueError:
                out.add((a, v)); continue
            out.add((f"class{ti}", f"c{smap.get(s, s)}"))
        else:
            out.add((a, v))
    return frozenset(out)


def import_usemap(target_org, donor_org, use_action, aligned):
    """Слить эпизоды использования донора в память цели (с переразметкой
    состояний), пересобрать правила. aligned=False → сырые метки c_i↔c_i.

    Цель может НЕ иметь действия-использования (жила только зондом): тогда
    выравниваем ось-донора на ГЛАВНУЮ ось цели (ту, что даст класс на
    экзамене через infer), а не на несуществующую ось использования."""
    di, dax = axis_with(donor_org, use_action)
    if dax is None or not target_org.axes:
        return None, None
    ti = max(range(len(target_org.axes)),
             key=lambda i: target_org.axes[i].n_obs)
    tax = target_org.axes[ti]
    smap = (alignment(dax, tax) if aligned
            else {s: s for s in range(min(dax.k, tax.k))})
    merged = Ruleset()
    merged.episodes = Counter(target_org.mem.episodes)
    for (ctx, a, o), m in donor_org.mem.episodes.items():
        if a != use_action:
            continue
        merged.episodes[(remap_ctx(ctx, di, ti, smap), a, o)] += m
    merged.consolidate()
    return merged, smap


if __name__ == "__main__":
    donor_w, target_w = "comp/donor", "comp/target"
    probe, ask = "просветить", "нагрузить"

    print("=== ДОНОР проживает свою жизнь (бюджет 2000) ===")
    donor_org, _ = live(donor_w, 2000)
    di, dax = axis_with(donor_org, ask)
    n_use = sum(1 for r in donor_org.mem.rules if r["action"] == ask)
    print(f"донор: осей={len(donor_org.axes)}, ось-использования k={dax.k}, "
          f"роли(rank->state)={role_order(dax)}, правил использования={n_use}")
    for r in donor_org.mem.rules:
        if r["action"] == ask:
            conds = " & ".join(f"{a}={v}" for a, v in sorted(r["conds"]))
            print(f"    ЕСЛИ {conds or '(нет)'} ТО {ask}={r['outcome'][0]}"
                  f"  [опора {r['support']}, искл {r['exceptions']}]")

    print("\n=== ЦЕЛЬ: голая vs с ввезённой картой (по бюджетам) ===")
    print(f"{'бюдж':>5} | {'ГОЛАЯ':>7} | {'ВВОЗ БЕЗ ВЫРАВН':>15} | "
          f"{'ВВОЗ+ВЫРАВН':>12} | детали")
    for b in [150, 300, 600, 1200]:
        target_org, glue_t = live(target_w, b)
        ti, tax = axis_with(target_org, probe)
        base = runworld.exam(target_org, glue_t, probe, ask)
        if tax is None:
            print(f"{b:>5} | {base[0]}/{base[1]} — оси у цели нет")
            continue
        # без выравнивания (сырые метки)
        target_org.mem, _ = import_usemap(target_org, donor_org, ask, False)
        raw = runworld.exam(target_org, glue_t, probe, ask)
        # с выравниванием (свежая жизнь цели, чтобы mem был чист)
        target2, glue2 = live(target_w, b)
        target2.mem, smap = import_usemap(target2, donor_org, ask, True)
        al = runworld.exam(target2, glue2, probe, ask)
        print(f"{b:>5} | {base[0]:>3}/{base[1]:<3} | {raw[0]:>7}/{raw[1]:<7} | "
              f"{al[0]:>6}/{al[1]:<5} | k_цель={tax.k}, dmap={smap}")
