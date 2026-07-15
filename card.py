"""card — перевод org ↔ карточка формы (мост для шкафа shelf.py).

Карточка = выжимка ОДНОЙ открытой формы: ось + подписи + карта использования.
Без records/hist (это сундук chest.py), без правил (выводимы), без чего-либо
про target-миры. Только прожитый донор.

Два перевода:
  build(org, glue, ...) -> card    сборка из живого организма (при открытии)
  donor_shim(card)      -> shim    тонкий псевдо-донор, который ест import_usemap
                                   БЕЗ живого org (для ввоза из полки)

Тонкость di-нормализации (подтверждено ТЗ, реш. #3): import_usemap.remap_ctx
переразмечает токены class{di}, где di = индекс оси-донора. Карточка несёт
РОВНО одну ось → после сборки di≡0. Поэтому при сборке эпизоды usemap
переписываются class{di_исходный} → class0, и шим тривиален (di=0).

Линза узнавания (v3, exam_heldout3.py): подписи сравнимы только через ОДНУ
линзу. Карточка хранит sig_lens[действие] = quick_signature(glue, acts=[a]);
узнавание идёт по линзе зонда sig_lens[probe]. Полная sig — для справки.
"""

from collections import Counter

import chest
import koopman
from compose import axis_with, role_order
from fastpath import quick_signature
from organism2 import Ruleset

SIG_BUDGET = 1000                # тот же бюджет подписи, что у v3 (детерминизм)


# ------------------------------------------------------------- сборка

def _form_axis(org, probe):
    """Ведущая ось формы = с макс. наблюдений; зонд обязан её читать."""
    if not org.axes:
        raise ValueError("у организма нет осей — форма не открыта")
    di = max(range(len(org.axes)), key=lambda i: org.axes[i].n_obs)
    dax = org.axes[di]
    if probe not in dax.actions:
        raise ValueError(f"зонд {probe!r} не читает ведущую ось "
                         f"(действия оси: {sorted(dax.actions)})")
    return di, dax


def _normalize_ctx(ctx, di):
    """class{di} → class0 в одном ctx (frozenset литералов). di=0 → тождество."""
    if di == 0:
        return ctx
    src, dst = f"class{di}", "class0"
    return frozenset((dst if a == src else a, v) for (a, v) in ctx)


def build(org, glue, origin, discovery):
    """Собрать карточку из живого донора. probe = первое действие мира (порядок
    yaml сохранён glue). use-действия = все прочие действия ведущей оси —
    карточка переносима для любого экзамена, а не только под один вопрос.

    discovery = {"budget": <бюджет открытия оси>, "seed": <база сидов>} —
    для воспроизводимости. sig_lens считаются отдельным SIG_BUDGET-проходом
    (как v3), не из org — чтобы T1 совпал с замороженным раннером."""
    probe = glue["object_actions"][0]
    di, dax = _form_axis(org, probe)
    use_actions = [a for a in sorted(dax.actions) if a != probe]

    # карта использования: эпизоды каждого use-действия, di-нормализованные
    usemap = {a: [] for a in use_actions}
    for (ctx, a, o), m in org.mem.episodes.items():
        if a in usemap:
            usemap[a].append((_normalize_ctx(ctx, di), a, o, m))

    # линзовые подписи: свежий SIG_BUDGET-проход на каждое действие оси (v3)
    sig_lens = {a: quick_signature(glue, SIG_BUDGET, acts=[a])[0]
                for a in sorted(dax.actions)}

    return {
        "id": None,                        # выдаст шкаф при add
        "added_v": None,                   # проставит шкаф
        "origin": origin,
        "discovery": dict(discovery),
        "probe": probe,                    # линза узнавания
        "axis": chest.enc(dax),            # ось кодеком сундука (без records/hist)
        "sig": koopman.signature(dax),     # полная подпись — справочно
        "sig_lens": sig_lens,              # по действию — для сравнения через линзу
        "usemap": chest.enc(usemap),       # эпизоды use-действий (frozenset ctx)
        "k": dax.k,
        "roles": role_order(dax),          # rank->state, кэш (вычислимо из оси)
    }


# ------------------------------------------------------------- ввоз из карточки

class _ShimMem:
    __slots__ = ("episodes",)

    def __init__(self, episodes):
        self.episodes = episodes


class _ShimOrg:
    """Псевдо-донор: ровно то, что читает import_usemap — .axes и .mem.episodes.
    Одна ось → axis_with вернёт di=0, а ctx уже нормализованы на class0."""
    __slots__ = ("axes", "mem")

    def __init__(self, axis, episodes):
        self.axes = [axis]
        self.mem = _ShimMem(episodes)


def donor_shim(card):
    """Из карточки собрать псевдо-донора для import_usemap (без живого org)."""
    axis = chest.dec(card["axis"])
    usemap = chest.dec(card["usemap"])
    episodes = Counter()
    for a, eps in usemap.items():
        for (ctx, act, o, m) in eps:
            episodes[(ctx, act, o)] += m
    return _ShimOrg(axis, episodes)


def recog_sig(card):
    """Подпись для узнавания = линза зонда (v3)."""
    return card["sig_lens"][card["probe"]]


# ------------------------------------------------------------- сериализация

def enc(card):
    return chest.enc(card)              # axis/usemap уже enc'нуты, но enc идемпотентен


def dec(obj):
    return chest.dec(obj)


# ------------------------------------------------------------- T6-самотест

if __name__ == "__main__":
    import json
    import runworld
    from organism2 import Organism
    from compose import import_usemap
    from fastpath import glue_of
    runworld.Organism = Organism

    print("=== T6: карточка ↔ шим даёт тот же ввоз, что живой донор ===")
    donor_w, target_w, ask = "comp1/donor", "comp1/target", "нагрузить"
    glue_d = glue_of(donor_w)
    donor_org = runworld.live(glue_d, False, 1500)

    card = build(donor_org, glue_d, origin=donor_w,
                 discovery={"budget": 1500, "seed": 0})
    # round-trip через JSON
    card2 = dec(json.loads(json.dumps(enc(card), ensure_ascii=False)))

    # ввоз через живой org vs через шим карточки — в свежую цель
    glue_t = glue_of(target_w)
    tgt_live = runworld.live(glue_t, False, 600)
    merged_live, smap_live = import_usemap(tgt_live, donor_org, ask, True)

    tgt_shim = runworld.live(glue_t, False, 600)
    shim = donor_shim(card2)
    merged_shim, smap_shim = import_usemap(tgt_shim, shim, ask, True)

    r_live = {(tuple(sorted(r["conds"])), r["action"], r["outcome"])
              for r in merged_live.rules if r["action"] == ask}
    r_shim = {(tuple(sorted(r["conds"])), r["action"], r["outcome"])
              for r in merged_shim.rules if r["action"] == ask}
    print(f"  правил ввоза (живой) = {len(r_live)}, (шим) = {len(r_shim)}")
    print(f"  smap живой={smap_live}  шим={smap_shim}")
    print(f"  {'✓ совпали' if r_live == r_shim else '✗ РАЗОШЛИСЬ'}")
    print(f"  probe={card['probe']}  k={card['k']}  роли={card['roles']}")
    print(f"  sig_lens линзы: {sorted(card['sig_lens'])}")
