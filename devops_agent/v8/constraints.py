"""
constraints.py — логический слой НАД 6 мета-типами: mutex + atmost.

Идея взята из DomiKnowS (логические связки), но без фреймворка и без авторских правил:
связка — это новый ВИД мета-предиката, гипотеза, которую агент добывает пробой мира-судьи
(см. learn_constraints.py). Здесь только ФОРМА (данные) и КОНТРАКТ (легальность), не содержание.

Две связки, ровно те, что residue гейта (devops_agent/v8/gate.py) показал как дыры:
  MUTEX(a, b)          — a и b не могут быть активны одновременно (взаимоисключение).
  ATMOST(members, B)   — сумма уровней членов ≤ B (разделяемый бюджет / ёмкость).

`if` сознательно в карантине: простой guard выражается через intervention.requires;
контекстно-временной `if` требует трогать мир-судью и откладывается отдельным разговором.

Связки ссылаются на id из того же пространства, что и граф (resource/status/setting),
графа НЕ раздувают: оверлей O(constraints), а не O(types²).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mutex:
    """a и b не могут быть активны (>0 / on) одновременно. Симметрична."""
    members: tuple                      # (id_a, id_b), оба — resource или оба — status

    def key(self) -> tuple:
        return ("mutex", tuple(sorted(self.members)))


@dataclass(frozen=True)
class AtMost:
    """Σ уровней членов ≤ bound. Разделяемый бюджет поверх ресурсов (возможно разных сущностей)."""
    members: tuple                      # ≥2 resource id
    bound: int

    def key(self) -> tuple:
        return ("atmost", tuple(sorted(self.members)), self.bound)


@dataclass
class ConstraintLayer:
    """Тонкий оверлей над графом: список заверенных пробой связок."""
    mutexes: list = field(default_factory=list)     # list[Mutex]
    atmosts: list = field(default_factory=list)      # list[AtMost]

    def keys(self) -> set:
        return {c.key() for c in self.mutexes} | {c.key() for c in self.atmosts}


def validate_constraints(layer: ConstraintLayer,
                         resource_ids: set,
                         status_ids: set,
                         setting_ids: set) -> list:
    """Контракт логического слоя. Возвращает нарушения (пусто = чисто)."""
    v = []
    decl = resource_ids | status_ids | setting_ids

    def homogeneous(members):
        """Все члены объявлены и одного метатипа (resource | setting | status)."""
        for grp in (resource_ids, setting_ids, status_ids):
            if all(x in grp for x in members):
                return True
        return False

    for m in layer.mutexes:
        if len(m.members) != 2:
            v.append(f"mutex-arity≠2:{m.members}")
            continue
        a, b = m.members
        if a == b:
            v.append(f"mutex-self:{a}")
        for x in m.members:
            if x not in decl:
                v.append(f"mutex→undeclared:{x}")
        # одинаковый метатип: булевы настройки / статусы / ресурсы — но не смесь
        if not homogeneous(m.members):
            v.append(f"mutex-mixed-kind:{m.members}")
    for am in layer.atmosts:
        if len(set(am.members)) < 2:
            v.append(f"atmost-arity<2:{am.members}")     # одночленный = bounded-resource, не связка
            continue
        if am.bound < 0:
            v.append(f"atmost-bound<0:{am.bound}")
        # либо общий КОЛИЧЕСТВЕННЫЙ бюджет (resource), либо at-most-N-активных (булевы setting/status)
        if not homogeneous(am.members):
            v.append(f"atmost-mixed-or-undeclared:{am.members}")
    return v
