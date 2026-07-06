"""
type_graph.py — v8 переписанный: KB = ГРАФ ТИПОВ; инстансы = ПОТОК ЗАДАЧ.

Урок спарринга, доведённый до конца:
  • Граф знаний (KB) — над ТИПАМИ, не над инстансами. Узел = тип; в нём таблица-закон
    (фича→статус) и тип-уровневые зависимости (api→pg→redis).
  • KB растёт ТОЛЬКО при появлении НОВОГО ТИПА. Размер KB = O(типов), не O(инстансов).
  • Инстансы (db1…dbN) — это ЗАДАЧИ. Их топология одноразова: приходит на вход, решается
    по KB, выбрасывается. В KB инстанс НИЧЕГО не добавляет (кроме регистрации нового типа).
  • DAG зависимостей живёт на уровне ИНСТАНСОВ (какой именно db к какому cache) — это данные
    задачи; обобщается в ТИП-ребро (pg зависит от redis) как часть знания.

Мир/оракул — детерминированные заглушки (без docker/LLM); сигнатуры боевые.
Предыдущий шаг (плоский 10-узловой граф) — в typed_tables.py; этот файл его исправляет.
"""

from dataclasses import dataclass, field

# --- Поверхность рычагов (дана; оракул лишь выбирает, КАКОЙ рычаг под симптом) ---
MEM_STEPS = [64, 128, 256, 512, 1024]
CONFIG_VALUES = ["default", "tuned"]
LEVERS = {
    "mem":    {"kind": "ordered_monotone", "default": MEM_STEPS[0]},
    "config": {"kind": "categorical",      "default": CONFIG_VALUES[0]},
}
SYMPTOM_LEVER = {"oom": "mem", "unhealthy": "config"}

# --- Скрытая истина мира, ключуется ТИПОМ (агенту недоступна) --------------------
TYPE_TRUTH = {
    "redis":  {"mem_threshold": 128, "config_sensitive": False},
    "worker": {"mem_threshold": 128, "config_sensitive": False},
    "pg":     {"mem_threshold": 512, "config_sensitive": False},
    "api":    {"mem_threshold": 256, "config_sensitive": True},    # требует config == "tuned"
    "queue":  {"mem_threshold": 256, "config_sensitive": False},   # новый тип во 2-й волне
}


# ============================ ЗАДАЧА (эфемерна) ==================================

@dataclass
class Instance:
    """Инстанс из задачи: имя + тип + проводка зависимостей. НЕ знание — данные деплоя."""
    id: str
    type: str
    deps: list = field(default_factory=list)   # id других инстансов этой задачи


@dataclass
class Obs:
    phase: str   # running | oom | unhealthy | dep_down


class World:
    """Мир-судья над ОДНИМ деплоем (одноразовая топология). Считает дорогие пробы."""

    def __init__(self, deployment: dict):
        self.dep = deployment            # id -> Instance (эфемерно, выбрасывается с задачей)
        self.running: set = set()
        self.n_provisions: int = 0

    def provision(self, inst_id: str, values: dict) -> Obs:
        self.n_provisions += 1
        inst = self.dep[inst_id]
        truth = TYPE_TRUTH[inst.type]
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
    """Заглушка LLM-источника: симптом → рычаг-кандидат (в бою — Azure, может врать)."""

    def __init__(self):
        self.n_calls = 0

    def propose(self, symptom: str) -> dict:
        self.n_calls += 1
        lever = SYMPTOM_LEVER[symptom]
        return {"lever": lever, "kind": LEVERS[lever]["kind"]}


# ============================ ЗНАНИЕ (вечно) =====================================

@dataclass
class TypeNode:
    """Узел графа знаний = ТИП. Несёт таблицу-закон и тип-уровневые зависимости."""
    name: str
    rules: dict = field(default_factory=dict)     # рычаг -> условие (фича→статус)
    covered: dict = field(default_factory=dict)   # симптом -> рычаг
    deps: set = field(default_factory=set)        # тип-уровневые зависимости (другие типы)

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
        out = []
        for lev, r in sorted(self.rules.items()):
            cond = (f"{lev} >= {r['threshold']}" if r["kind"] == "ordered_monotone"
                    else f"{lev} == {r['ok']!r}")
            out.append(f"        {cond} → ok   [{r['kind']}]")
        return "\n".join(out)


class KB:
    """Граф знаний: узлы = типы. Растёт ТОЛЬКО на новом типе."""

    def __init__(self):
        self.types: dict = {}     # name -> TypeNode

    def ensure(self, type_name: str):
        """→ (узел, был_ли_новым). Единственное место, где KB прибавляет узел."""
        is_new = type_name not in self.types
        node = self.types.setdefault(type_name, TypeNode(type_name))
        return node, is_new

    def render(self) -> str:
        out = []
        for name, node in self.types.items():
            deps = f" → зависит от {sorted(node.deps)}" if node.deps else ""
            out.append(f"  ТИП '{name}'{deps}")
            out.append(node.render_table())
        return "\n".join(out)


@dataclass
class InstRec:
    inst: str
    typ: str
    new_type: bool
    success: bool
    oracle: int
    probes: int
    path: str
    trace: list = field(default_factory=list)


class Agent:
    """Решает задачу-инстанс: распознаёт ТИП → применяет таблицу типа | намывает её (новый тип)."""

    def __init__(self, kb: KB, oracle: Oracle):
        self.kb = kb
        self.oracle = oracle

    def solve_task(self, target: str, world: World) -> list:
        """Задача = поднять инстанс target. Резолвит проводку зависимостей (данные деплоя)."""
        recs = []
        for inst_id in self._closure(target, world.dep):
            if inst_id in world.running:
                continue                                  # зависимость уже поднята ранее
            recs.append(self._solve_instance(inst_id, world))
        return recs

    @staticmethod
    def _closure(target: str, dep: dict) -> list:
        """Инстанс-уровневый топосорт замыкания зависимостей (единственная работа DAG)."""
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
        node, new_type = self.kb.ensure(inst.type)
        for d in inst.deps:                                # обобщить проводку в тип-ребро
            node.deps.add(world.dep[d].type)
        o0, p0 = self.oracle.n_calls, world.n_provisions
        trace = []

        values, tried = {}, {}
        for lev in LEVERS:
            v = node.prescribe(lev)
            values[lev] = v if v is not None else LEVERS[lev]["default"]
            tried[lev] = {values[lev]}

        while True:
            obs = world.provision(inst_id, values)
            trace.append(f"проба mem={values['mem']} config={values['config']!r} → {obs.phase}")
            if obs.phase == "running":
                for sym, lev in node.covered.items():
                    node.write(lev, values[lev])
                path = (f"COLD — новый тип '{inst.type}': таблица намыта" if new_type
                        else f"WARM — тип '{inst.type}' узнан: таблица применена")
                return InstRec(inst_id, inst.type, new_type, True,
                               self.oracle.n_calls - o0, world.n_provisions - p0, path, trace)
            lever = node.covered.get(obs.phase)
            if lever is None:
                prop = self.oracle.propose(obs.phase)
                lever = prop["lever"]
                node.covered[obs.phase] = lever
                trace.append(f"  разрыв {obs.phase!r} → ОРАКУЛ: рычаг={lever} [{prop['kind']}]")
            nxt = self._advance(lever, values[lever], tried[lever])
            if nxt is None:
                return InstRec(inst_id, inst.type, new_type, False,
                               self.oracle.n_calls - o0, world.n_provisions - p0, "исчерпание", trace)
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


# ============================ ДЕМО: две волны задач ==============================

def _deployment(instances: list) -> dict:
    return {i.id: i for i in instances}


def _run_wave(title: str, kb: KB, oracle: Oracle, deployment: dict, tasks: list):
    print(f"\n{'='*64}\n{title}\n{'='*64}")
    world = World(deployment)
    agent = Agent(kb, oracle)
    o_before, types_before = oracle.n_calls, len(kb.types)
    for t in tasks:
        print(f"\n— задача: поднять {t} (инстансы деплоя одноразовы) —")
        for rec in agent.solve_task(t, world):
            tag = "  [+тип в KB]" if rec.new_type else ""
            print(f"  · {rec.inst:8} тип={rec.typ:6} {rec.path}{tag}  oracle={rec.oracle} проб={rec.probes}")
            for line in rec.trace:
                print(f"        {line}")
    print(f"\n  итог волны: oracle +{oracle.n_calls - o_before},  "
          f"типов в KB: {types_before} → {len(kb.types)},  инстансов обработано: {len(deployment)}")


def main() -> None:
    print("=== v8 переписанный: KB = граф типов; инстансы = поток задач ===")
    kb, oracle = KB(), Oracle()

    # --- Волна 1: незнакомый парк, 10 инстансов / 4 типа ---
    dep1 = _deployment([
        Instance("cache1", "redis"),
        Instance("cache2", "redis"),
        Instance("db1", "pg", ["cache1"]),
        Instance("db2", "pg", ["cache1"]),
        Instance("db3", "pg", ["cache2"]),
        Instance("worker1", "worker", ["cache1"]),
        Instance("worker2", "worker", ["cache2"]),
        Instance("api1", "api", ["db1"]),
        Instance("api2", "api", ["db2"]),
        Instance("api3", "api", ["db3"]),
    ])
    _run_wave("ВОЛНА 1 — пустая KB, незнакомый парк (10 инстансов)",
              kb, oracle, dep1, ["api1", "worker1", "api2", "worker2", "api3"])

    # --- Волна 2: ДРУГОЙ деплой, новые имена; типы знакомы, кроме одного нового ---
    dep2 = _deployment([
        Instance("cache9", "redis"),
        Instance("db9", "pg", ["cache9"]),
        Instance("api9", "api", ["db9"]),
        Instance("queue1", "queue"),          # НОВЫЙ тип
    ])
    _run_wave("ВОЛНА 2 — другой деплой, новые инстансы; знакомые типы + 1 новый",
              kb, oracle, dep2, ["api9", "queue1"])

    # --- Итог ---
    print(f"\n{'='*64}\nГРАФ ЗНАНИЙ (KB) — всё, что осталось после двух волн:\n{'='*64}")
    print(kb.render())
    total_instances = len(dep1) + len(dep2)
    print(f"\n  KB = {len(kb.types)} узла-ТИПА.  Прошло задач-инстансов: {total_instances}.")
    print(f"  Всего обращений к оракулу: {oracle.n_calls}.")
    print("  Волна 2: 3 инстанса знакомых типов добавили в KB 0 узлов и стоили 0 оракула;")
    print("  KB выросла ровно на 1 — на новый тип 'queue'. Инстансы граф знаний НЕ пухнут.")


if __name__ == "__main__":
    main()
