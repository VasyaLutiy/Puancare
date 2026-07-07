"""datalite — крошечный Datalog на голом stdlib.

Факты    — кортежи: ("род", "x0", "яд").
Литералы — кортежи с переменными: ("род", "?x", "яд");
           переменная = строка, начинающаяся с "?".
Негация  — ("не", <литерал>): истинна, если литерал (после текущей
           подстановки) не матчится ни одним фактом.
Правило  — (голова, [литералы тела]).

match(body, facts)  — генератор подстановок, при которых конъюнкция
                      тела истинна (рабочая лошадка worldkit:
                      decision list = первое правило с матчем).
derive(rules, facts) — наивный вывод до неподвижной точки.
query(pattern, facts) — все подстановки для одного литерала.
"""


def is_var(t):
    return isinstance(t, str) and t.startswith("?")


def subst_term(t, s):
    return s.get(t, t) if is_var(t) else t


def unify(pattern, fact, s):
    """Расширяет подстановку s так, чтобы pattern совпал с fact, или None."""
    if len(pattern) != len(fact):
        return None
    out = dict(s)
    for p, f in zip(pattern, fact):
        p = subst_term(p, out)
        if is_var(p):
            out[p] = f
        elif p != f:
            return None
    return out


def match(body, facts, s=None):
    """Все подстановки, при которых конъюнкция body истинна в facts."""
    def rec(i, s):
        if i == len(body):
            yield s
            return
        lit = body[i]
        if lit[0] == "не":
            inner = tuple(subst_term(t, s) for t in lit[1])
            if any(unify(inner, f, {}) is not None
                   for f in facts if f and f[0] == inner[0]):
                return
            yield from rec(i + 1, s)
            return
        for f in facts:
            if f[0] != lit[0]:
                continue
            s2 = unify(lit, f, s)
            if s2 is not None:
                yield from rec(i + 1, s2)

    yield from rec(0, dict(s) if s else {})


def holds(body, facts, s=None):
    """Истинна ли конъюнкция хотя бы при одной подстановке."""
    return next(match(body, facts, s), None) is not None


def derive(rules, facts):
    """Наивный вывод до неподвижной точки. rules: [(head, body)]."""
    facts = set(facts)
    changed = True
    while changed:
        changed = False
        for head, body in rules:
            new_facts = []
            for s in match(body, frozenset(facts)):
                new = tuple(subst_term(t, s) for t in head)
                if new not in facts:
                    new_facts.append(new)
            if new_facts:
                facts.update(new_facts)
                changed = True
    return facts


def query(pattern, facts):
    out = []
    for f in facts:
        if f[0] == pattern[0]:
            s = unify(pattern, f, {})
            if s is not None:
                out.append(s)
    return out


if __name__ == "__main__":
    F = {("род", "x0", "яд"), ("род", "x1", "еда"),
         ("связан", "x1", "e0"), ("состояние", "e0", "свежее")}

    assert holds([("род", "?x", "яд")], F)
    assert not holds([("род", "?x", "камень")], F)
    assert [s["?x"] for s in match([("род", "?x", "еда")], F)] == ["x1"]

    # конъюнкция с join: съедобный И связанный с живой сущностью
    got = list(match([("род", "?x", "еда"), ("связан", "?x", "?e"),
                      ("состояние", "?e", "свежее")], F))
    assert len(got) == 1 and got[0]["?e"] == "e0"

    # негация: x0 ни с кем не связан
    assert holds([("род", "?x", "яд"), ("не", ("связан", "?x", "?e"))], F)

    # вывод: опасность выводится из рода
    R = [(("опасен", "?x"), [("род", "?x", "яд")])]
    assert ("опасен", "x0") in derive(R, F)
    assert ("опасен", "x1") not in derive(R, F)

    # частичная подстановка снаружи (привязка цели)
    assert holds([("род", "?x", "яд")], F, {"?x": "x0"})
    assert not holds([("род", "?x", "яд")], F, {"?x": "x1"})

    print("datalite: все самотесты пройдены")
