"""worldkit — конструктор миров: анкета -> datalite-клаузы -> живой мир.

Анкета — текстовый файл (см. ШАБЛОН.yaml): объекты с видимыми/скрытыми
атрибутами, сущности с состояниями и скрытыми связями, действия как
таблицы "если ... то ...", ловушки-корреляции, вмешательства, экзамены.

Компиляция: каждая строка таблицы действия -> (условия-литералы, исход,
эффекты). Исполнение шага: факты состояния + производные правила ->
первое правило с матчем (decision list поверх datalite.match).

Эффекты (в квадратных скобках): исчезает | каскад | починка.
Специальные условия: состояние=... (для действий над сущностью),
связанный_отсутствует (нет живого пути починки).

Автоклей: make_glue(spec) выводит object_actions / ctx_fn / entities_fn /
truth_fn / link_truth / фабрику миров и экзамены — организм подключается
без единой строчки мирозависимого кода.
"""

from __future__ import annotations

import random
import re

from datalite import derive, holds

# ---------------------------------------------------------------- парсер

def parse_text(text):
    """Мини-подмножество YAML: словари, списки '- ...', встрочные [a, b]."""
    lines = []
    for raw in text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        line = raw.split(" #")[0].rstrip()
        if line.strip():
            lines.append(line)
    pos = 0

    def parse_val(v):
        if v.startswith("[") and v.endswith("]"):
            return [x.strip() for x in v[1:-1].split(",") if x.strip()]
        return v

    def block(indent):
        nonlocal pos
        result = None
        while pos < len(lines):
            line = lines[pos]
            ind = len(line) - len(line.lstrip())
            if ind < indent:
                break
            if ind > indent:
                raise ValueError(f"строка {pos + 1}: неожиданный отступ: {line.strip()!r}")
            s = line.strip()
            if s.startswith("- "):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise ValueError(f"строка {pos + 1}: список внутри словаря: {s!r}")
                result.append(s[2:].strip())
                pos += 1
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise ValueError(f"строка {pos + 1}: ключ внутри списка: {s!r}")
                key, sep, val = s.partition(":")
                if not sep:
                    raise ValueError(f"строка {pos + 1}: нет ':' в {s!r}")
                key, val = key.strip(), val.strip()
                pos += 1
                if val:
                    result[key] = parse_val(val)
                elif pos < len(lines):
                    nind = len(lines[pos]) - len(lines[pos].lstrip())
                    result[key] = block(nind) if nind > indent else ""
                else:
                    result[key] = ""
        return result

    return block(0)


# ------------------------------------------------------------- компилятор

RULE_RE = re.compile(r"^если\s+(.+?)\s+то\s+(.+)$")
TRAP_RE = re.compile(r"^если\s+(\S+?)=(\S+)\s+то\s+(\S+?)=(\S+)\s+\((\d+)%\)$")
GUAR_RE = re.compile(r"^минимум\s+(\d+)\s+объект\w*\s+с\s+(\S+?)=(\S+)$")
LINK_RE = re.compile(r"^(\d+)(?:\s*\((\S+?)=(\S+)\))?$")
EXAM_RE = re.compile(r"^зонд\s+(\S+)\s+вопрос\s+(\S+?)(\s+с\s+откатом)?$")
DYN_RE  = re.compile(r"^если\s+(\S+?)=(\S+)\s+то\s+шанс_стать\s+(\S+)\s+\((\d+)%\)$")
EFFECTS = ("исчезает", "каскад", "починка")


def _compile_rule(s, attrs, states, lineno=""):
    eff = []
    m = re.search(r"\[(.*)\]\s*$", s)
    if m:
        eff = [e.strip() for e in m.group(1).split(",")]
        for e in eff:
            if e not in EFFECTS:
                raise ValueError(f"{lineno}неизвестный эффект {e!r} "
                                 f"(можно: {', '.join(EFFECTS)})")
        s = s[:m.start()].strip()
    if s.startswith("иначе"):
        conds, result = [], s[len("иначе"):].strip()
    else:
        m = RULE_RE.match(s)
        if not m:
            raise ValueError(f"{lineno}не разобрал правило {s!r} "
                             f"(жду 'если a=b то исход' или 'иначе исход')")
        conds = [c.strip() for c in re.split(r"\s+и\s+", m.group(1))]
        result = m.group(2).strip()
    body = []
    for c in conds:
        if c == "связанный_отсутствует":
            body.append(("не", ("есть_живой_путь", "?t")))
            continue
        if "=" not in c:
            raise ValueError(f"{lineno}условие {c!r} — жду 'атрибут=значение'")
        a, v = (x.strip() for x in c.split("=", 1))
        if a == "состояние":
            if v not in states:
                raise ValueError(f"{lineno}состояние {v!r} не объявлено")
            body.append(("состояние", "?t", v))
        else:
            if a not in attrs:
                raise ValueError(f"{lineno}атрибут {a!r} не объявлен")
            if v not in attrs[a]:
                raise ValueError(f"{lineno}значение {v!r} нет у атрибута {a!r}")
            body.append((a, "?t", v))
    if not result:
        raise ValueError(f"{lineno}пустой исход")
    return {"body": body, "result": result, "effects": eff}


class Spec:
    def __init__(self, raw):
        obj = raw.get("объекты") or {}
        self.name = raw.get("мир", "безымянный")
        self.n_objects = int(raw.get("объектов", 8))
        self.visible = {k: list(v) for k, v in (obj.get("видимое") or {}).items()}
        self.hidden = {k: list(v) for k, v in (obj.get("скрытое") or {}).items()}
        self.attrs = {**self.visible, **self.hidden}
        if not self.attrs:
            raise ValueError("у объектов нет ни одного атрибута")

        self.traps = []
        for t in (obj.get("ловушки") or ([obj["ловушка"]] if obj.get("ловушка") else [])):
            m = TRAP_RE.match(t)
            if not m:
                raise ValueError(f"ловушка {t!r}: жду 'если h=v то a=w (90%)'")
            self.traps.append((m.group(1), m.group(2), m.group(3),
                               m.group(4), int(m.group(5)) / 100))
        self.guarantees = []
        for g in (obj.get("гарантии") or []):
            m = GUAR_RE.match(g)
            if not m:
                raise ValueError(f"гарантия {g!r}: жду 'минимум N объектов с a=v'")
            self.guarantees.append((int(m.group(1)), m.group(2), m.group(3)))

        ent = raw.get("сущности") or {}
        self.has_entities = bool(ent)
        self.ent_count = int(ent.get("штук", 2)) if ent else 0
        self.states = list(ent.get("состояния", [])) if ent else []
        if self.has_entities and len(self.states) < 2:
            raise ValueError("у сущностей должно быть >=2 состояний (живое первым)")
        self.live, self.dead = (self.states + ["", ""])[:2]
        lk = str(ent.get("связаны_с", "1")) if ent else "1"
        m = LINK_RE.match(lk)
        if not m:
            raise ValueError(f"связаны_с: {lk!r} — жду '1' или '1 (род=еда)'")
        self.links_per = int(m.group(1))
        self.link_filter = (m.group(2), m.group(3)) if m.group(2) else None

        self.obj_actions, self.ent_actions = {}, {}
        for key, rules in (raw.get("действия") or {}).items():
            if isinstance(rules, str):
                rules = [rules]
            name, _, tag = key.partition("@")
            name = name.strip()
            compiled = [_compile_rule(r, self.attrs, self.states,
                                      f"{name}: ") for r in rules]
            if tag.strip() == "сущность":
                self.ent_actions[name] = compiled
            else:
                self.obj_actions[name] = compiled
        if not self.obj_actions:
            raise ValueError("нет ни одного действия с объектом")

        self.interventions = {}
        for name, d in (raw.get("вмешательство") or {}).items():
            chosen = d.get("выбираешь", []) if isinstance(d, dict) else []
            fixed = {}
            if isinstance(d, dict) and d.get("фиксировано"):
                for pair in str(d["фиксировано"]).split(","):
                    a, v = (x.strip() for x in pair.split("="))
                    if a not in self.attrs or v not in self.attrs[a]:
                        raise ValueError(f"{name}: фиксировано {a}={v} — не объявлено")
                    fixed[a] = v
            for a in chosen:
                if a not in self.visible:
                    raise ValueError(f"{name}: выбираешь {a!r} — не видимый атрибут")
            self.interventions[name] = {"chosen": list(chosen), "fixed": fixed}

        # экзамен — четвёрка по WORLD_PROTOCOL: (метка, зонд, вопрос, откат).
        # Откат = повторить зонд после вопроса; работает только для
        # действий-переключателей — автор мира отвечает за обратимость.
        self.exams = []
        for e in (raw.get("экзамены") or []):
            m = EXAM_RE.match(e)
            if not m:
                raise ValueError(f"экзамен {e!r}: жду 'зонд A вопрос B"
                                 f" [с откатом]'")
            probe, ask, undo = m.group(1), m.group(2), bool(m.group(3))
            for a in (probe, ask):
                if a not in self.obj_actions:
                    raise ValueError(f"экзамен: действие {a!r} не объявлено")
            probe_effs = {eff for r in self.obj_actions[probe]
                          for eff in r["effects"]}
            if "исчезает" in probe_effs:
                raise ValueError(
                    f"экзамен: зонд {probe!r} уничтожает объект — вопрос "
                    f"задать будет некому; выбери наблюдающий зонд")
            if probe_effs and not undo:
                raise ValueError(
                    f"экзамен: зонд {probe!r} меняет мир "
                    f"({', '.join(sorted(probe_effs))}) — допиши 'с откатом' "
                    f"(если повтор зонда откатывает) или выбери безопасный")
            self.exams.append((f"{probe}->{ask}", probe, ask, undo))

        self.dynamics = []
        dyn = raw.get("динамика") or {}
        for rule in (dyn.get("каждый_шаг") or []):
            m = DYN_RE.match(rule)
            if not m:
                raise ValueError(
                    f"динамика {rule!r}: жду "
                    f"'если attr=from то шанс_стать to (p%)'")
            attr, from_val, to_val, pct = (m.group(1), m.group(2),
                                           m.group(3), int(m.group(4)))
            if attr not in self.hidden:
                raise ValueError(f"динамика: {attr!r} — только скрытые атрибуты")
            for v in (from_val, to_val):
                if v not in self.attrs[attr]:
                    raise ValueError(
                        f"динамика: {v!r} не объявлено у атрибута {attr!r}")
            self.dynamics.append((attr, from_val, to_val, pct / 100))


def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        return Spec(parse_text(fh.read()))


# ------------------------------------------------------------------- мир

DERIVED = [(("есть_живой_путь", "?e"),
            [("связан", "?o", "?e"), ("объект", "?o")])]


class GenericWorld:
    def __init__(self, spec, seed=0):
        self.spec = spec
        self.seed = seed
        self.reset()

    def reset(self):
        sp = self.spec
        rng = random.Random(self.seed)
        self.objects = {}
        for i in range(sp.n_objects):
            self.objects[f"x{i}"] = self._gen_object(rng)
        for k, attr, val in sp.guarantees:
            have = [o for o in self.objects.values() if o[attr] == val]
            others = [o for o in self.objects.values() if o[attr] != val]
            rng.shuffle(others)
            for o in others[:max(0, k - len(have))]:
                o[attr] = val
        self.entities = {}
        if sp.has_entities:
            pool = [n for n, o in self.objects.items()
                    if sp.link_filter is None
                    or o[sp.link_filter[0]] == sp.link_filter[1]]
            rng.shuffle(pool)
            for i in range(sp.ent_count):
                links = tuple(pool[i * sp.links_per:(i + 1) * sp.links_per])
                self.entities[f"e{i}"] = {"status": sp.live, "links": links}
        self._counter = 0
        self._rng = rng
        return self.observe()

    def _gen_object(self, rng):
        sp = self.spec
        o = {a: rng.choice(vals) for a, vals in sp.hidden.items()}
        for a, vals in sp.visible.items():
            trapped = False
            for (ha, hv, va, vv, p) in sp.traps:
                if va == a and o.get(ha) == hv and rng.random() < p:
                    o[a] = vv
                    trapped = True
                    break
            if not trapped:
                o[a] = rng.choice(vals)
        return o

    # ------------------------------------------------------- наблюдение

    def observe(self):
        sp = self.spec
        return {
            "объекты": {n: {a: o[a] for a in sp.visible}
                        for n, o in sorted(self.objects.items())},
            "сущности": {n: {"status": e["status"]}
                         for n, e in sorted(self.entities.items())},
        }

    def _facts(self):
        fs = set()
        for n, o in self.objects.items():
            fs.add(("объект", n))
            for a, v in o.items():
                fs.add((a, n, v))
        for n, e in self.entities.items():
            fs.add(("сущность", n))
            fs.add(("состояние", n, e["status"]))
            for o in e["links"]:
                fs.add(("связан", o, n))
        return derive(DERIVED, fs)

    # ------------------------------------------------------------- шаг

    def _tick(self):
        for obj in self.objects.values():
            mutations = {}
            for attr, from_val, to_val, prob in self.spec.dynamics:
                if attr not in mutations and obj.get(attr) == from_val:
                    if self._rng.random() < prob:
                        mutations[attr] = to_val
            obj.update(mutations)

    def step(self, action, target=None, **kw):
        before = self.observe()
        result, effects = self._apply(action, target, kw)
        if self.spec.dynamics:
            self._tick()
        return {"obs": before, "action": action, "target": target, "args": kw,
                "result": result, "effects": effects,
                "obs_after": self.observe()}

    def _apply(self, action, target, kw):
        sp = self.spec
        effects = []
        if action in sp.interventions:
            iv = sp.interventions[action]
            rng = self._rng
            o = {a: rng.choice(vals) for a, vals in sp.attrs.items()}
            for a in iv["chosen"]:
                v = kw.get(a)
                if v not in sp.visible[a]:
                    return "непонятные_параметры", effects
                o[a] = v
            o.update(iv["fixed"])
            name = f"n{self._counter}"
            self._counter += 1
            self.objects[name] = o
            effects.append(("создан", name))
            return "ок", effects

        if action in sp.obj_actions:
            if target not in self.objects:
                return "нет_такого_объекта", effects
            rules = sp.obj_actions[action]
        elif action in sp.ent_actions:
            if target not in self.entities:
                return "нет_такой_сущности", effects
            rules = sp.ent_actions[action]
        else:
            return "нет_такого_действия", effects

        facts = self._facts()
        for rule in rules:
            if holds(rule["body"], facts, {"?t": target}):
                return rule["result"], self._effects(rule["effects"],
                                                     action, target, effects)
        return "ничего_не_произошло", effects

    def _effects(self, tags, action, target, effects):
        sp = self.spec
        if "каскад" in tags:
            for n, e in self.entities.items():
                if target in e["links"] and e["status"] == sp.live:
                    e["status"] = sp.dead
                    effects.append((sp.dead, n))
        if "исчезает" in tags:
            del self.objects[target]
            effects.append(("исчез", target))
        if "починка" in tags:
            self.entities[target]["status"] = sp.live
            effects.append((sp.live, target))
        return effects

    # ------------------------------------------------- пространство действий

    def action_space(self):
        sp = self.spec
        acts = []
        for n in self.objects:
            for a in sp.obj_actions:
                acts.append((a, n, {}))
        for n in self.entities:
            for a in sp.ent_actions:
                acts.append((a, n, {}))
        for name, iv in sp.interventions.items():
            combos = [{}]
            for a in iv["chosen"]:
                combos = [dict(c, **{a: v}) for c in combos
                          for v in sp.visible[a]]
            for c in combos:
                acts.append((name, None, c))
        return acts


# ------------------------------------------------------------- экспортёр

def export_rules(spec):
    """Правила мира в формате памяти агента (WORLD_PROTOCOL, раздел 4):
    (действие, frozenset((атрибут, значение)), (результат, виды эффектов)).

    Механический перевод из datalite-клауз:
      (атрибут, ?t, знач)          -> условие (атрибут, знач)
      (состояние, ?t, знач)        -> условие (состояние, знач)
      (не, (есть_живой_путь, ?t))  -> условие (есть_живой_путь, 0)
      каскад     -> правило раздваивается по linked_live:
                    (…, linked_live=1) -> результат + [<мёртвое>, …]
                    (…, linked_live=0) -> результат + […]
      исчезает   -> эффект "исчез";  починка -> эффект <живое состояние>

    Диф с выученной памятью — с точностью до переименования изобретённых
    предикатов (class* <-> скрытые атрибуты)."""
    def conds_of(body):
        cs = set()
        for lit in body:
            if lit[0] == "не":
                cs.add((lit[1][0], 0))
            else:
                cs.add((lit[0], lit[2]))
        return cs

    out = []
    for name, rules in spec.obj_actions.items():
        for r in rules:
            cs = conds_of(r["body"])
            effs = set()
            if "исчезает" in r["effects"]:
                effs.add("исчез")
            if "починка" in r["effects"]:
                effs.add(spec.live)
            if "каскад" in r["effects"]:
                out.append((name, frozenset(cs | {("linked_live", 1)}),
                            (r["result"],
                             tuple(sorted(effs | {spec.dead})))))
                out.append((name, frozenset(cs | {("linked_live", 0)}),
                            (r["result"], tuple(sorted(effs)))))
            else:
                out.append((name, frozenset(cs),
                            (r["result"], tuple(sorted(effs)))))
    for name, rules in spec.ent_actions.items():
        for r in rules:
            effs = {spec.live} if "починка" in r["effects"] else set()
            out.append((name, frozenset(conds_of(r["body"])),
                        (r["result"], tuple(sorted(effs)))))
    for name in spec.interventions:
        out.append((name, frozenset(), ("ок", ("создан",))))
    return out


# --------------------------------------------------------------- автоклей

def make_glue(spec):
    def ctx_fn(obs, action, target, kw):
        if action in spec.obj_actions:
            o = obs["объекты"][target]
            return frozenset(o.items())
        if action in spec.ent_actions:
            return frozenset({("состояние", obs["сущности"][target]["status"])})
        return frozenset((a, v) for a, v in kw.items())

    def truth_fn(w, t):
        if t in w.objects:
            return "-".join(w.objects[t][a] for a in spec.hidden)
        return None

    return dict(
        object_actions=tuple(spec.obj_actions),
        ctx_fn=ctx_fn,
        truth_fn=truth_fn,
        entities_fn=lambda obs: {n: e["status"]
                                 for n, e in obs["сущности"].items()},
        factory=lambda ep: GenericWorld(spec, seed=42 + ep),
        exam_world=lambda: GenericWorld(spec, seed=999),
        link_truth=lambda ep, o, e: (
            o in GenericWorld(spec, seed=42 + ep).entities[e]["links"]),
        exams=list(spec.exams),
    )
