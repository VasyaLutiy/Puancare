"""
strategies.py — библиотека стратегий разрешения, индексированная типом переменной (kind).

Обобщение тезиса v3: «стратегия = функция от order-структуры неизвестного».
Порт рабочих v2-петель:
  ordered_monotone → DoublingStrategy   (порт удвоения из agent.py: MEM_BASE·2^k до MEM_CEILING)
  categorical      → EnumerateStrategy   (порт bios._choose_config: incumbent → sorted non-bad alt)
Будущее (вне MVP): bounded→bisection, unimodal(sweet-spot)→тернарный/золотое сечение, boolean→try-both.
"""

from devops_agent.v3.meta_bios import MEM_BASE, MEM_CEILING


class Strategy:
    """
    Контракт: initial(bios,svc) → стартовое значение рычага;
              on_gap(bios,svc,last) → следующее значение, либо None (исчерпана → refute/no_plan).
    """
    def initial(self, bios, svc):
        raise NotImplementedError

    def on_gap(self, bios, svc, last_value):
        raise NotImplementedError


class DoublingStrategy(Strategy):
    """ordered_monotone (память/диск/реплики). TODO(v3): start = known threshold or MEM_BASE;
    on_gap = last*2, None если > MEM_CEILING. Запись порога — на стороне MetaBios.record_running."""
    def initial(self, bios, svc):
        raise NotImplementedError("TODO(v3): порт agent.py current_mem старт")

    def on_gap(self, bios, svc, last_value):
        raise NotImplementedError("TODO(v3): порт agent.py current_mem*=2 + ceiling")


class EnumerateStrategy(Strategy):
    """categorical (config/версия/флаг). TODO(v3): порт bios._choose_config —
    incumbent если не bad, иначе первый sorted non-bad; None если все bad. on_gap = mark_bad + следующий."""
    def initial(self, bios, svc):
        raise NotImplementedError("TODO(v3): порт _choose_config")

    def on_gap(self, bios, svc, last_value):
        raise NotImplementedError("TODO(v3): mark_bad_config + следующий кандидат")


def strategy_for(kind: str) -> Strategy:
    """ordered_monotone→DoublingStrategy, categorical→EnumerateStrategy. TODO(v3): + boolean позже."""
    raise NotImplementedError("TODO(v3): диспетчер kind→Strategy")
