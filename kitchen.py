"""Кухня: второй мир для карантинного экзамена ТЕОРИИ.

Тот же протокол, что у world.World (observe / action_space / step),
но другая онтология и другая скрытая структура:

Вещества, видимые атрибуты: color, form, origin.
Скрыто: kind (food | poison) — аналог owner.
Блюда: status (fresh | spoiled); скрытая связь key (ключевой ингредиент)
— аналог config -> process.

Ground-truth динамика:
  taste(x)  -> "yummy" если food, "bitter" если poison (неразрушающий зонд)
  eat(x)    -> "sick" если poison (не съедается);
               "ok" если food: x исчезает, блюда с key=x портятся (каскад)
  heat(x)   -> "boiled" если form=liquid, иначе "charred" (видимое правило)
  cut(x)    -> "splash" если form=liquid, иначе "chopped" (видимое правило)
  serve(d)  -> "applause" если fresh, "complaint" если spoiled
  remake(d) -> "already_fresh" | "ok" (если key ещё существует) |
               "missing_ingredient"
  buy(color, form) -> новое вещество, всегда food с origin=market
               (рынок безопасен — инструмент вмешательства)

Ловушка-корреляция (в генераторе, не в динамике): яд растёт в лесу (90%),
еда — из сада/с рынка (90%). Истинная причина вкуса — kind, не origin.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass

COLORS = ("red", "green", "white")
FORMS = ("solid", "liquid")
ORIGINS = ("forest", "garden", "market")
SUB_ACTIONS = ("taste", "eat", "heat", "cut")
DISH_ACTIONS = ("serve", "remake")


@dataclass
class Substance:
    name: str
    color: str
    form: str
    origin: str
    kind: str            # СКРЫТО

    def visible(self, hidden):
        return {k: v for k, v in asdict(self).items() if k not in hidden}


@dataclass
class Dish:
    name: str
    status: str          # fresh | spoiled
    key: str             # СКРЫТО: ключевой ингредиент

    def visible(self):
        return {"name": self.name, "status": self.status}


class Kitchen:
    def __init__(self, seed=0, n_subst=8, hidden=("kind",)):
        self.seed = seed
        self.n_subst = n_subst
        self.hidden = set(hidden)
        self.reset()

    def reset(self):
        rng = random.Random(self.seed)
        self.subst = {}
        for i in range(self.n_subst):
            kind = rng.choice(("food", "poison"))
            if rng.random() < 0.9:
                origin = "forest" if kind == "poison" else rng.choice(
                    ("garden", "market"))
            else:
                origin = rng.choice(ORIGINS)
            s = Substance(f"s{i}", rng.choice(COLORS), rng.choice(FORMS),
                          origin, kind)
            self.subst[s.name] = s
        # каскад "съел ключевое -> блюдо испортилось" должен быть наблюдаем:
        # гарантируем >=2 съедобных для ключей (аналог user-конфига в ОС)
        foods = [s for s in self.subst.values() if s.kind == "food"]
        while len(foods) < 2:
            victim = rng.choice([s for s in self.subst.values()
                                 if s.kind == "poison"])
            victim.kind = "food"
            if victim.origin == "forest" and rng.random() < 0.9:
                victim.origin = rng.choice(("garden", "market"))
            foods = [s for s in self.subst.values() if s.kind == "food"]
        self.dishes = {}
        for i, key in enumerate(rng.sample(foods, 2)):
            d = Dish(f"d{i}", "fresh", key.name)
            self.dishes[d.name] = d
        self._buy_counter = 0
        return self.observe()

    def observe(self):
        return {
            "subst": {n: s.visible(self.hidden)
                      for n, s in sorted(self.subst.items())},
            "dishes": {n: d.visible() for n, d in sorted(self.dishes.items())},
        }

    def step(self, action, target=None, **kw):
        before = self.observe()
        result, effects = self._apply(action, target, kw)
        return {"obs": before, "action": action, "target": target, "args": kw,
                "result": result, "effects": effects,
                "obs_after": self.observe()}

    def _apply(self, action, target, kw):
        effects = []
        if action in SUB_ACTIONS:
            s = self.subst.get(target)
            if s is None:
                return "no_such_substance", effects
            if action == "taste":
                return ("yummy" if s.kind == "food" else "bitter"), effects
            if action == "eat":
                if s.kind == "poison":
                    return "sick", effects
                del self.subst[s.name]
                effects.append(("consumed", s.name))
                for d in self.dishes.values():
                    if d.key == s.name and d.status == "fresh":
                        d.status = "spoiled"
                        effects.append(("dish_spoiled", d.name))
                return "ok", effects
            if action == "heat":
                return ("boiled" if s.form == "liquid" else "charred"), effects
            if action == "cut":
                return ("splash" if s.form == "liquid" else "chopped"), effects

        if action in DISH_ACTIONS:
            d = self.dishes.get(target)
            if d is None:
                return "no_such_dish", effects
            if action == "serve":
                return ("applause" if d.status == "fresh"
                        else "complaint"), effects
            if action == "remake":
                if d.status == "fresh":
                    return "already_fresh", effects
                if d.key not in self.subst:
                    return "missing_ingredient", effects
                d.status = "fresh"
                effects.append(("dish_fresh", d.name))
                return "ok", effects

        if action == "buy":
            color = kw.get("color", "white")
            form = kw.get("form", "solid")
            if color not in COLORS or form not in FORMS:
                return "bad_args", effects
            name = f"new{self._buy_counter}"
            self._buy_counter += 1
            self.subst[name] = Substance(name, color, form, "market", "food")
            effects.append(("bought", name))
            return "ok", effects

        return "bad_action", effects

    def action_space(self):
        acts = []
        for n in self.subst:
            for a in SUB_ACTIONS:
                acts.append((a, n, {}))
        for n in self.dishes:
            for a in DISH_ACTIONS:
                acts.append((a, n, {}))
        for c, f in itertools.product(COLORS, FORMS):
            acts.append(("buy", None, {"color": c, "form": f}))
        return acts
