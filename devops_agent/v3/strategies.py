"""
strategies.py — библиотека стратегий разрешения, индексированная типом переменной (kind).

Обобщение тезиса v3: «стратегия = функция от order-структуры неизвестного».
Порт рабочих v2-петель:
  ordered_monotone → DoublingStrategy   (порт удвоения из agent.py: last*2 до MEM_CEILING)
  categorical      → EnumerateStrategy   (порт bios._choose_config: mark_bad + следующий non-bad)
Будущее (вне MVP): bounded→bisection, unimodal(sweet-spot)→тернарный, boolean→try-both.

Контракт стратегии: on_gap(bios, svc, last_value) → следующее значение рычага, либо None
(исчерпана → агент трактует как ceiling/no_config). Начальное значение даёт MetaBios.initial_value.
"""

from devops_agent.v3.meta_bios import MEM_CEILING


class Strategy:
    def on_gap(self, bios, svc, last_value):
        raise NotImplementedError


class DoublingStrategy(Strategy):
    """ordered_monotone: удвоить ресурс. None если вышли за MEM_CEILING."""
    def on_gap(self, bios, svc, last_value):
        nxt = last_value * 2
        return nxt if nxt <= MEM_CEILING else None


class EnumerateStrategy(Strategy):
    """categorical: текущее значение known-bad → следующий непровальный кандидат (или None)."""
    def on_gap(self, bios, svc, last_value):
        bios.mark_bad_config(svc, last_value)
        return bios.choose_config(svc)


_BY_KIND = {
    "ordered_monotone": DoublingStrategy,
    "categorical": EnumerateStrategy,
}


def strategy_for(kind: str) -> Strategy:
    """kind → экземпляр Strategy. boolean и пр. — добавим за пределами MVP."""
    try:
        return _BY_KIND[kind]()
    except KeyError:
        raise ValueError(f"нет стратегии для kind={kind!r} (MVP: {sorted(_BY_KIND)})")
