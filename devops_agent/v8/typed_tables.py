"""
typed_tables.py — отправная точка скорректированной модели знания (после спарринга).

Зафиксированные положения:
  • Знание агента — НЕ граф. DAG нужен ТОЛЬКО для зависимостей между сущностями;
    его единственный потребитель — топосорт порядка провижининга.
  • Содержательное знание живёт в ЛОКАЛЬНОЙ ТАБЛИЦЕ УЗЛА (action-model типа):
    отображение `условие на фичах → статус` (напр. mem >= 512 → ok).
  • Таблица ключуется ТИПОМ (ось обобщения), не инстансом. N инстансов одного типа
    делят ОДНУ таблицу → знание переносится. Это и отличает модель от мемоизации v7.
  • Таблица растёт из РАЗРЫВОВ: оракул (LLM) предлагает фичу-кандидата, мир-судья
    подтверждает её пробой и выдаёт порог; каждая строка = выжившая гипотеза.

Мир и оракул здесь — детерминированные заглушки (без docker/LLM) ради воспроизводимости;
сигнатуры совпадают с боевыми (v4.Sandbox.provision / Azure-оракул symptom→рычаг).
"""

from dataclasses import dataclass, field

# --- Поверхность рычагов (ДАНА; оракул лишь выбирает, КАКОЙ рычаг под симптом) ---
MEM_STEPS = [64, 128, 256, 512, 1024]
CONFIG_VALUES = ["default", "tuned"]
LEVERS = {
    "mem":    {"kind": "ordered_monotone", "default": MEM_STEPS[0]},
    "config": {"kind": "categorical",      "default": CONFIG_VALUES[0]},
}
SYMPTOM_LEVER = {"oom": "mem", "unhealthy": "config"}

# --- Скрытая истина мира (в бою — поведение docker; агенту НЕдоступна) -----------
TYPE_TRUTH = {
    "redis":  {"mem_threshold": 128, "config_sensitive": False},
    "worker": {"mem_threshold": 128, "config_sensitive": False},
    "pg":     {"mem_threshold": 512, "config_sensitive": False},
    "api":    {"mem_threshold": 256, "config_sensitive": True},   # требует config == "tuned"
}

# --- Граф: 10 инстансов / 4 типа; рёбра = ТОЛЬКО зависимости -------------------
GRAPH = {
    "cache1":  {"type": "redis",  "deps": []},
    "cache2":  {"type": "redis",  "deps": []},
    "db1":     {"type": "pg",     "deps": ["cache1"]},
    "db2":     {"type": "pg",     "deps": ["cache1"]},
    "db3":     {"type": "pg",     "deps": ["cache2"]},
    "worker1": {"type": "worker", "deps": ["cache1"]},
    "worker2": {"type": "worker", "deps": ["cache2"]},
    "api1":    {"type": "api",    "deps": ["db1"]},
    "api2":    {"type": "api",    "deps": ["db2"]},
    "api3":    {"type": "api",    "deps": ["db3"]},
}


@dataclass
class Obs:
    phase: str   # running | oom | unhealthy | dep_down


class World:
    """Мир-судья. provision(node, values) → Obs. Считает пробы (дорогая операция)."""

    def __init__(self):
        self.running: set = set()
        self.n_provisions: int = 0

    def provision(self, node: str, values: dict) -> Obs:
        self.n_provisions += 1
        spec = GRAPH[node]
        truth = TYPE_TRUTH[spec["type"]]
        for d in spec["deps"]:
            if d not in self.running:
                return Obs("dep_down")
        if values["mem"] < truth["mem_threshold"]:
            return Obs("oom")
        if truth["config_sensitive"] and values.get("config") != "tuned":
            return Obs("unhealthy")
        self.running.add(node)
        return Obs("running")


class Oracle:
    """Заглушка LLM-источника: симптом → рычаг-кандидат. В бою — Azure (может врать, мир рассудит)."""

    def __init__(self):
        self.n_calls: int = 0

    def propose(self, symptom: str) -> dict:
        self.n_calls += 1
        lever = SYMPTOM_LEVER[symptom]
        return {"lever": lever, "kind": LEVERS[lever]["kind"]}


@dataclass
class TypeModel:
    """Таблица-закон одного ТИПА: рычаг → выученное условие; симптом → покрывающий рычаг."""
    type_name: str
    rules: dict = field(default_factory=dict)     # lever -> условие
    covered: dict = field(default_factory=dict)   # symptom -> lever (что уже умеем диагностировать)

    def prescribe(self, lever: str):
        """Предписанное значение рычага по выученному правилу (или None — правила нет)."""
        r = self.rules.get(lever)
        if r is None:
            return None
        return r["threshold"] if r["kind"] == "ordered_monotone" else r["ok"]

    def write(self, lever: str, value) -> None:
        kind = LEVERS[lever]["kind"]
        if kind == "ordered_monotone":
            self.rules[lever] = {"kind": kind, "threshold": value}
        else:
            self.rules[lever] = {"kind": kind, "ok": value}

    def render(self) -> str:
        if not self.rules:
            return "      (пусто)"
        out = []
        for lev, r in sorted(self.rules.items()):
            if r["kind"] == "ordered_monotone":
                out.append(f"      {lev:7}: {lev} >= {r['threshold']:<5} → ok   [ordered_monotone]")
            else:
                out.append(f"      {lev:7}: {lev} == {r['ok']!r:9} → ok   [categorical]")
        return "\n".join(out)


@dataclass
class NodeRec:
    node: str
    typ: str
    cold: bool
    success: bool
    oracle: int
    probes: int
    path: str
    trace: list = field(default_factory=list)


class Agent:
    """Решает узел: распознаёт ТИП → применяет таблицу типа | намывает её из разрывов."""

    def __init__(self, world: World, oracle: Oracle, kb: dict):
        self.world = world
        self.oracle = oracle
        self.kb = kb            # type_name -> TypeModel

    def solve_node(self, node: str) -> NodeRec:
        typ = GRAPH[node]["type"]
        cold = typ not in self.kb
        model = self.kb.setdefault(typ, TypeModel(typ))
        o0, p0 = self.oracle.n_calls, self.world.n_provisions
        trace: list = []

        # стартовые значения: предписанные таблицей или дефолтные
        values, tried = {}, {}
        for lev in LEVERS:
            v = model.prescribe(lev)
            values[lev] = v if v is not None else LEVERS[lev]["default"]
            tried[lev] = {values[lev]}

        while True:
            obs = self.world.provision(node, values)
            trace.append(f"проба: mem={values['mem']} config={values['config']!r} → {obs.phase}")

            if obs.phase == "running":
                for sym, lev in model.covered.items():     # зафиксировать выученные условия
                    model.write(lev, values[lev])
                path = ("COLD (тип нов → таблица намыта)" if cold
                        else "WARM (тип узнан → таблица применена)")
                return NodeRec(node, typ, cold, True,
                               self.oracle.n_calls - o0, self.world.n_provisions - p0, path, trace)

            lever = model.covered.get(obs.phase)
            if lever is None:                               # РАЗРЫВ: таблица не знает симптом
                prop = self.oracle.propose(obs.phase)
                lever = prop["lever"]
                model.covered[obs.phase] = lever
                trace.append(f"  разрыв {obs.phase!r}: таблицы нет → ОРАКУЛ: рычаг={lever} [{prop['kind']}]")

            nxt = self._advance(lever, values[lever], tried[lever])
            if nxt is None:
                return NodeRec(node, typ, cold, False,
                               self.oracle.n_calls - o0, self.world.n_provisions - p0,
                               "исчерпание рычага", trace)
            trace.append(f"  стратегия: {lever} {values[lever]} → {nxt}")
            values[lever] = nxt
            tried[lever].add(nxt)

    @staticmethod
    def _advance(lever: str, cur, tried: set):
        if LEVERS[lever]["kind"] == "ordered_monotone":
            i = MEM_STEPS.index(cur)
            return MEM_STEPS[i + 1] if i + 1 < len(MEM_STEPS) else None
        for v in CONFIG_VALUES:
            if v not in tried:
                return v
        return None


def toposort() -> list:
    """Единственная работа DAG: порядок, в котором зависимости поднимаются раньше зависимых."""
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        for d in GRAPH[n]["deps"]:
            visit(d)
        seen.add(n)
        order.append(n)

    for n in GRAPH:
        visit(n)
    return order


def main() -> None:
    print("=== v8: знание = таблицы по типам; DAG = только зависимости ===\n")
    print("Граф (10 узлов / 4 типа):")
    for n, s in GRAPH.items():
        print(f"  {n:7} тип={s['type']:6} deps={s['deps']}")

    order = toposort()
    print(f"\nТопосорт (порядок провижининга — единственная работа DAG):\n  {order}\n")
    print("=" * 64)

    world, oracle, kb = World(), Oracle(), {}
    agent = Agent(world, oracle, kb)
    curve, probes_curve = [], []

    for node in order:
        rec = agent.solve_node(node)
        print(f"\n[{node:7} | тип {rec.typ:6}] {rec.path}")
        for t in rec.trace:
            print("    " + t)
        print(f"    → success={rec.success}  oracle={rec.oracle}  проб={rec.probes}")
        if rec.cold:
            print(f"    ТАБЛИЦА типа '{rec.typ}' создана/дополнена:")
            print(kb[rec.typ].render())
        curve.append(rec.oracle)
        probes_curve.append(rec.probes)

    print("\n" + "=" * 64)
    print("ИТОГ — всё знание агента (4 таблицы на 10 узлов):\n")
    for typ, m in kb.items():
        print(f"  тип '{typ}':")
        print(m.render())
    print(f"\n  LLM-кривая (oracle на узел): {curve}  Σ={sum(curve)}")
    print(f"  проб-кривая:                 {probes_curve}  Σ={sum(probes_curve)}")

    # Контрольный замер БЕЗ переноса: у каждого узла своя пустая память (как per-instance v7).
    w2, o2 = World(), Oracle()
    for node in order:
        Agent(w2, o2, {}).solve_node(node)
    print("\n" + "-" * 64)
    print(f"  без переноса (per-instance, как v7): oracle={o2.n_calls}, проб={w2.n_provisions}")
    print(f"  с таблицей-по-типу (эта модель):     oracle={sum(curve)}, проб={sum(probes_curve)}")
    print("  → знание перенеслось по ТИПУ: повтор типа стоит oracle=0. Это обобщение, не мемоизация.")


if __name__ == "__main__":
    main()
