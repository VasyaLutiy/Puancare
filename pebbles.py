"""Камешки: третий мир — математика без цифр. Фаза 1 (статичная).

Кучки камешков, n = 2..30. ЧИСЕЛ АГЕНТ НЕ ВИДИТ: кучка — непрозрачный
объект. Видимое: size (на глаз), basket (происхождение), color (шум).
Скрыто: n, а с ним ДВЕ независимые переменные — чётность и простота.

Ground-truth динамика:
  split_evenly(x) -> "ok" если n чётно, иначе "leftover"     (чётность)
  pair_up(x)      -> "ok" если n чётно, иначе "one_left"     (чётность)
  make_rectangle(x) -> "ok" если n составное, иначе "impossible" (простота)
  share(x)        -> "shared" если составное, иначе "remainder_always"
  dump(x)         -> "ok": кучка исчезает; весы, где она участвует,
                     опрокидываются (каскад — чисто реляционный)
  check(s)        -> "balanced" | "tipped"
  fix(s)          -> "already_balanced" | "ok" (обе кучки существуют) |
                     "missing_pile"
  new_pile(size)  -> новая кучка случайного n из размерной корзины,
                     basket=neutral (инструмент вмешательства)

Весы: сущность-состояние. Весы держат ДВЕ кучки равного n,
status = balanced | tipped. Связь кучка-весы скрыта.

Ловушка-корреляция: в basket_a кладут простые (90%), в basket_b
составные (90%). Истинная причина "прямоугольник не выкладывается" —
простота, не корзина.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SIZES = (("small", 2, 9), ("medium", 10, 19), ("large", 20, 30))
BASKETS = ("basket_a", "basket_b", "neutral")
COLORS = ("red", "gray", "white")
PILE_ACTIONS = ("split_evenly", "pair_up", "make_rectangle", "share", "dump")
SCALE_ACTIONS = ("check", "fix")
MUTATE_ACTIONS = ("add_pebble", "remove_pebble")


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def size_of(n):
    for name, lo, hi in SIZES:
        if lo <= n <= hi:
            return name
    return "large"


@dataclass
class Pile:
    name: str
    n: int               # СКРЫТО — вместе с чётностью и простотой
    basket: str
    color: str

    def visible(self):
        d = asdict(self)
        del d["n"]
        d["size"] = size_of(self.n)
        return d


@dataclass
class Scale:
    name: str
    status: str          # balanced | tipped
    piles: tuple         # СКРЫТО: две кучки равного n

    def visible(self):
        return {"name": self.name, "status": self.status}


class Pebbles:
    def __init__(self, seed=0, n_piles=8):
        self.seed = seed
        self.n_piles = n_piles
        self.reset()

    def reset(self):
        rng = random.Random(self.seed)
        self.piles = {}
        for i in range(self.n_piles):
            n = rng.randint(2, 30)
            if rng.random() < 0.9:
                basket = "basket_a" if is_prime(n) else "basket_b"
            else:
                basket = rng.choice(("basket_a", "basket_b"))
            self.piles[f"p{i}"] = Pile(f"p{i}", n, basket, rng.choice(COLORS))
        # весы: две пары кучек уравниваются по n (вторая подгоняется)
        self.scales = {}
        names = rng.sample(sorted(self.piles), 4)
        for i in range(2):
            a, b = names[2 * i], names[2 * i + 1]
            self.piles[b].n = self.piles[a].n
            self.scales[f"w{i}"] = Scale(f"w{i}", "balanced", (a, b))
        self._new_counter = 0
        return self.observe()

    def observe(self):
        return {
            "piles": {n: p.visible() for n, p in sorted(self.piles.items())},
            "scales": {n: s.visible() for n, s in sorted(self.scales.items())},
        }

    def step(self, action, target=None, **kw):
        before = self.observe()
        result, effects = self._apply(action, target, kw)
        return {"obs": before, "action": action, "target": target, "args": kw,
                "result": result, "effects": effects,
                "obs_after": self.observe()}

    def _apply(self, action, target, kw):
        effects = []
        if action in PILE_ACTIONS:
            p = self.piles.get(target)
            if p is None:
                return "no_such_pile", effects
            if action == "split_evenly":
                return ("ok" if p.n % 2 == 0 else "leftover"), effects
            if action == "pair_up":
                return ("ok" if p.n % 2 == 0 else "one_left"), effects
            if action == "make_rectangle":
                return ("impossible" if is_prime(p.n) else "ok"), effects
            if action == "share":
                return ("remainder_always" if is_prime(p.n)
                        else "shared"), effects
            if action == "dump":
                del self.piles[p.name]
                effects.append(("pile_dumped", p.name))
                for s in self.scales.values():
                    if p.name in s.piles and s.status == "balanced":
                        s.status = "tipped"
                        effects.append(("scale_tipped", s.name))
                return "ok", effects

        if action in SCALE_ACTIONS:
            s = self.scales.get(target)
            if s is None:
                return "no_such_scale", effects
            if action == "check":
                return s.status, effects
            if action == "fix":
                if s.status == "balanced":
                    return "already_balanced", effects
                if not all(n in self.piles for n in s.piles):
                    return "missing_pile", effects
                s.status = "balanced"
                effects.append(("scale_fixed", s.name))
                return "ok", effects

        if action == "new_pile":
            size = kw.get("size", "small")
            band = next((b for b in SIZES if b[0] == size), None)
            if band is None:
                return "bad_args", effects
            rng = random.Random((self.seed, self._new_counter))
            n = rng.randint(band[1], band[2])
            name = f"new{self._new_counter}"
            self._new_counter += 1
            self.piles[name] = Pile(name, n, "neutral", rng.choice(COLORS))
            effects.append(("pile_new", name))
            return "ok", effects

        return "bad_action", effects

    def action_space(self):
        acts = []
        for n in self.piles:
            for a in PILE_ACTIONS:
                acts.append((a, n, {}))
        for n in self.scales:
            for a in SCALE_ACTIONS:
                acts.append((a, n, {}))
        for s, _, _ in SIZES:
            acts.append(("new_pile", None, {"size": s}))
        return acts
