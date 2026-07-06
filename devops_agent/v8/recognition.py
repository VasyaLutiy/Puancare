"""
recognition.py — v8 + РАСПОЗНАВАНИЕ ТИПА (убираем чит «тип дан ярлыком»).

Что меняется относительно type_graph.py:
  • Агент НЕ читает тип инстанса. Он наблюдает лишь дешёвую ПОВЕРХНОСТЬ (теги: образ,
    порты, ключи конфига) и сам решает, на какой ИЗВЕСТНЫЙ тип это похоже.
  • Распознавание = дешёвый приор по СХОДСТВУ (Jaccard поверхностей), не точное совпадение.
    Это лишь ГИПОТЕЗА — может ошибиться.
  • Гипотезу проверяет ПРОБА (мир-судья): применяю таблицу угаданного типа одной пробой.
    running → распознал (тёпло). Сломалось → РЕФЬЮТ: угадал неверно → учу как новый тип.
    Это E3 (рефьют), поднятый с уровня фичи на уровень ТИПА.
  • Имена типов агент ПРИДУМЫВАЕТ сам (T0, T1, …). Ярлыки мира (redis/pg) в агент не текут.

Истинный тип инстанса знает ТОЛЬКО мир-судья (для вынесения вердикта). Агент — никогда.
"""

from dataclasses import dataclass, field

MEM_STEPS = [64, 128, 256, 512, 1024]
CONFIG_VALUES = ["default", "tuned"]
LEVERS = {
    "mem":    {"kind": "ordered_monotone", "default": MEM_STEPS[0]},
    "config": {"kind": "categorical",      "default": CONFIG_VALUES[0]},
}
SYMPTOM_LEVER = {"oom": "mem", "unhealthy": "config"}
TAU = 0.5   # порог сходства поверхностей для гипотезы распознавания

# --- Скрытая истина мира: поведение + НАБЛЮДАЕМАЯ поверхность. Агенту недоступна. ---
#     bigcache намеренно похож на redis поверхностью ({kv}), но тяжелее → провоцирует рефьют.
TYPE_TRUTH = {
    "redis":    {"mem_threshold": 128,  "config_sensitive": False, "surface": {"kv", "p6379"}},
    "pg":       {"mem_threshold": 512,  "config_sensitive": False, "surface": {"sql", "p5432", "durable"}},
    "api":      {"mem_threshold": 256,  "config_sensitive": True,  "surface": {"http", "p8080"}},
    "worker":   {"mem_threshold": 128,  "config_sensitive": False, "surface": {"qcons"}},
    "bigcache": {"mem_threshold": 1024, "config_sensitive": False, "surface": {"kv"}},
}


@dataclass
class Instance:
    id: str
    true_type: str               # ТОЛЬКО для мира-судьи; агент это поле НЕ читает
    deps: list = field(default_factory=list)

    @property
    def surface(self) -> set:    # то, что агент реально наблюдает
        return set(TYPE_TRUTH[self.true_type]["surface"])


@dataclass
class Obs:
    phase: str


class World:
    def __init__(self, deployment: dict):
        self.dep = deployment
        self.running: set = set()
        self.n_provisions: int = 0

    def provision(self, inst_id: str, values: dict) -> Obs:
        self.n_provisions += 1
        inst = self.dep[inst_id]
        truth = TYPE_TRUTH[inst.true_type]
        for d in inst.deps:
            if d not in self.running:
                return Obs("dep_down")
        if values["mem"] < truth["mem_threshold"]:
            return Obs("oom")
        if truth["config_sensitive"] and values.get("config") != "tuned":
            return Obs("unhealthy")
        self.running.add(inst_id)
        return Obs("running")


class Oracle:
    def __init__(self):
        self.n_calls = 0

    def propose(self, symptom: str) -> dict:
        self.n_calls += 1
        lever = SYMPTOM_LEVER[symptom]
        return {"lever": lever, "kind": LEVERS[lever]["kind"]}


@dataclass
class TypeNode:
    name: str                                      # ПРИДУМАН агентом (T0, T1, …)
    surface: set = field(default_factory=set)      # наблюдаемая сигнатура (для распознавания)
    rules: dict = field(default_factory=dict)
    covered: dict = field(default_factory=dict)
    deps: set = field(default_factory=set)

    def prescribe(self, lever: str):
        r = self.rules.get(lever)
        if r is None:
            return None
        return r["threshold"] if r["kind"] == "ordered_monotone" else r["ok"]

    def write(self, lever: str, value) -> None:
        kind = LEVERS[lever]["kind"]
        self.rules[lever] = ({"kind": kind, "threshold": value} if kind == "ordered_monotone"
                             else {"kind": kind, "ok": value})

    def render_table(self) -> str:
        if not self.rules:
            return "        (пусто)"
        rows = []
        for lev, r in sorted(self.rules.items()):
            cond = (f"{lev} >= {r['threshold']}" if r["kind"] == "ordered_monotone"
                    else f"{lev} == {r['ok']!r}")
            rows.append(f"        {cond} → ok")
        return "\n".join(rows)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


class KB:
    def __init__(self):
        self.types: dict = {}

    def recognize(self, surface: set):
        """Дешёвый приор: ближайший известный тип по сходству поверхностей. → (имя|None, score)."""
        best, best_s = None, 0.0
        for name, node in self.types.items():
            s = jaccard(surface, node.surface)
            if s > best_s:
                best, best_s = name, s
        return (best, best_s) if best_s >= TAU else (None, best_s)

    def new_type(self, surface: set) -> TypeNode:
        name = f"T{len(self.types)}"
        node = TypeNode(name, surface=set(surface))
        self.types[name] = node
        return node

    def render(self) -> str:
        out = []
        for name, node in self.types.items():
            deps = f"  зависит от {sorted(node.deps)}" if node.deps else ""
            out.append(f"  {name}  поверхность={sorted(node.surface)}{deps}")
            out.append(node.render_table())
        return "\n".join(out)


@dataclass
class InstRec:
    inst: str
    success: bool
    oracle: int
    probes: int
    path: str
    trace: list = field(default_factory=list)


class Agent:
    def __init__(self, kb: KB, oracle: Oracle):
        self.kb = kb
        self.oracle = oracle

    def solve_task(self, target: str, world: World) -> list:
        recs = []
        for inst_id in self._closure(target, world.dep):
            if inst_id in world.running:
                continue
            recs.append(self._solve_instance(inst_id, world))
        return recs

    @staticmethod
    def _closure(target: str, dep: dict) -> list:
        order, seen = [], set()

        def visit(x):
            if x in seen:
                return
            seen.add(x)
            for d in dep[x].deps:
                visit(d)
            order.append(x)

        visit(target)
        return order

    def _solve_instance(self, inst_id: str, world: World) -> InstRec:
        inst = world.dep[inst_id]
        surface = inst.surface                          # агент наблюдает ТОЛЬКО это
        o0, p0 = self.oracle.n_calls, world.n_provisions
        trace = [f"наблюдаю поверхность {sorted(surface)}"]

        guess, score = self.kb.recognize(surface)
        if guess is not None:
            node = self.kb.types[guess]
            trace.append(f"  гипотеза: похоже на {guess} (сходство {score:.2f}) → ПРОБА-проверка")
            vals = self._prescribed(node)
            obs = world.provision(inst_id, vals)
            trace.append(f"  проба mem={vals['mem']} config={vals['config']!r} → {obs.phase}")
            if obs.phase == "running":
                self._absorb_dep(node, inst, world)
                return InstRec(inst_id, True, self.oracle.n_calls - o0, world.n_provisions - p0,
                               f"РАСПОЗНАН {guess} (сходство {score:.2f})", trace)
            trace.append(f"  ✗ РЕФЬЮТ: {obs.phase} — поверхность похожа, но ведёт себя иначе → учу НОВЫЙ тип")

        node = self.kb.new_type(surface)
        self._absorb_dep(node, inst, world)
        ok = self._learn(node, inst_id, world, trace)
        refuted = guess is not None
        path = (f"{'РЕФЬЮТ→' if refuted else ''}НОВЫЙ ТИП {node.name}"
                + (f" (приор ошибся на {guess})" if refuted else " (приор: ничего похожего)"))
        return InstRec(inst_id, ok, self.oracle.n_calls - o0, world.n_provisions - p0, path, trace)

    def _absorb_dep(self, node: TypeNode, inst: Instance, world: World) -> None:
        for d in inst.deps:                              # обобщить проводку в тип-ребро
            dep_node = self._node_of_running(d, world)
            if dep_node:
                node.deps.add(dep_node)

    def _node_of_running(self, inst_id, world):
        """Какому ТИПУ агент уже сопоставил поднятый инстанс-зависимость (по сходству)."""
        g, s = self.kb.recognize(world.dep[inst_id].surface)
        return g

    def _prescribed(self, node: TypeNode) -> dict:
        return {lev: (node.prescribe(lev) if node.prescribe(lev) is not None
                      else LEVERS[lev]["default"]) for lev in LEVERS}

    def _learn(self, node: TypeNode, inst_id: str, world: World, trace: list) -> bool:
        values = {lev: LEVERS[lev]["default"] for lev in LEVERS}
        tried = {lev: {values[lev]} for lev in LEVERS}
        while True:
            obs = world.provision(inst_id, values)
            trace.append(f"  учу: mem={values['mem']} config={values['config']!r} → {obs.phase}")
            if obs.phase == "running":
                for sym, lev in node.covered.items():
                    node.write(lev, values[lev])
                return True
            lever = node.covered.get(obs.phase)
            if lever is None:
                prop = self.oracle.propose(obs.phase)
                lever = prop["lever"]
                node.covered[obs.phase] = lever
                trace.append(f"    разрыв {obs.phase!r} → ОРАКУЛ: рычаг={lever}")
            nxt = self._advance(lever, values[lever], tried[lever])
            if nxt is None:
                return False
            values[lever] = nxt
            tried[lever].add(nxt)

    @staticmethod
    def _advance(lever, cur, tried):
        if LEVERS[lever]["kind"] == "ordered_monotone":
            i = MEM_STEPS.index(cur)
            return MEM_STEPS[i + 1] if i + 1 < len(MEM_STEPS) else None
        for v in CONFIG_VALUES:
            if v not in tried:
                return v
        return None


# ============================ ДЕМО ===============================================

def main() -> None:
    print("=== v8 + распознавание: агент сам выводит тип из поверхности + пробы ===")
    kb, oracle = KB(), Oracle()

    dep = {i.id: i for i in [
        Instance("r1", "redis"),
        Instance("d1", "pg", ["r1"]),
        Instance("a1", "api", ["d1"]),
        Instance("r2", "redis"),
        Instance("b1", "bigcache"),     # поверхность {kv} ⇒ приор скажет «redis», проба опровергнет
        Instance("b2", "bigcache"),     # теперь похож на ВЫУЧЕННЫЙ агентом тип
        Instance("w1", "worker", ["r1"]),
    ]}
    world = World(dep)
    agent = Agent(kb, oracle)

    for target in ["a1", "r2", "b1", "b2", "w1"]:
        print(f"\n— задача: поднять {target} —")
        for rec in agent.solve_task(target, world):
            print(f"  · {rec.inst:4} {rec.path}   oracle={rec.oracle} проб={rec.probes}")
            for line in rec.trace:
                print(f"        {line}")

    print(f"\n{'='*64}\nТАКСОНОМИЯ, КОТОРУЮ АГЕНТ ПОСТРОИЛ САМ (имена — его собственные):\n{'='*64}")
    print(kb.render())
    print(f"\n  типов открыто: {len(kb.types)} | обращений к оракулу: {oracle.n_calls} | "
          f"проб: {world.n_provisions}")
    print("  Ни одного ярлыка типа агент не получал. redis/pg/bigcache знает только мир-судья.")
    print("  b1: приор сказал «как redis», проба опровергла → агент открыл отдельный тип.")
    print("  b2: тот же профиль распознан как УЖЕ ВЫУЧЕННЫЙ агентом тип (0 оракула).")


if __name__ == "__main__":
    main()
