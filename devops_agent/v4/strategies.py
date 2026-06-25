"""
strategies.py — GENERIC стратегии разрешения, индексированные по KIND (НЕ по домену).

Петля выбирает стратегию по kind, который предложил LLM. Стратегия НЕ знает, в каком домене
работает — она оперирует только (значение, симптом). Это и есть граница не-хардкода:
библиотека стратегий доменно-слепа; смысл («что лечит») приходит от LLM, проверяет docker.

consumes(symptom) — умеет ли стратегия двигаться на этом симптоме. Если симптом приписан
рычагу, а его стратегия НЕ consumes → kind неверен → петля делает рефьют (переспрос LLM).
"""


class Strategy:
    def consumes(self, symptom: str) -> bool:
        raise NotImplementedError

    def next_value(self, history: list):
        """history: list[(value, symptom)] для ЭТОГО рычага. → следующее значение | None (исчерпано)."""
        raise NotImplementedError


class Doubling(Strategy):
    """ordered_monotone: 'больше = лучше'. Удваивает до потолка. Один триггер-симптом."""
    def __init__(self, trigger: str, base: int = 64, ceiling: int = 16384):
        self.trigger, self.base, self.ceiling = trigger, base, ceiling

    def consumes(self, symptom):
        return symptom == self.trigger

    def next_value(self, history):
        if not history:
            return self.base
        nxt = history[-1][0] * 2
        return nxt if nxt <= self.ceiling else None


class Enumerate(Strategy):
    """categorical: перебор вариантов до успеха. Один триггер-симптом."""
    def __init__(self, trigger: str, options: list):
        self.trigger, self.options = trigger, list(options)

    def consumes(self, symptom):
        return symptom == self.trigger

    def next_value(self, history):
        tried = {v for v, _ in history}
        for o in self.options:
            if o not in tried:
                return o
        return None


class Bounded(Strategy):
    """non-monotone: безопасное окно. Нужны ДВА направленных симптома (мало/много). Бисекция."""
    def __init__(self, low_symptom: str, high_symptom: str, base: int = 64):
        self.low, self.high, self.base = low_symptom, high_symptom, base

    def consumes(self, symptom):
        return symptom in (self.low, self.high)

    def next_value(self, history):
        lo, hi = 0, None
        for v, s in history:
            if s == self.low:
                lo = max(lo, v)               # было слишком мало → поднять пол
            elif s == self.high:
                hi = v if hi is None else min(hi, v)   # было слишком много → опустить потолок
        if hi is None:                        # «много» ещё не видели → растём
            return max(lo, self.base) * 2 if lo else self.base
        mid = (lo + hi) // 2
        return mid if lo < mid < hi else None


_BY_KIND = {"ordered_monotone": Doubling, "categorical": Enumerate, "bounded": Bounded}


def make_strategy(kind: str, **cfg) -> Strategy:
    """kind + конфиг из предложения LLM → стратегия. Петля НЕ знает домена, только kind."""
    if kind == "ordered_monotone":
        return Doubling(trigger=cfg["trigger"])
    if kind == "categorical":
        return Enumerate(trigger=cfg["trigger"], options=cfg["options"])
    if kind == "bounded":
        return Bounded(low_symptom=cfg["low_symptom"], high_symptom=cfg["high_symptom"])
    raise ValueError(f"неизвестный kind {kind!r} (есть: {sorted(_BY_KIND)})")
