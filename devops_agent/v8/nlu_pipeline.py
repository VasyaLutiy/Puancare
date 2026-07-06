"""
nlu_pipeline.py — сквозной прогон: человеческая фраза → AGENTJSON → агент грунтует.

Цель: подтвердить работоспособность КОМПИЛЯТОРА (NLU = внешняя LLM) и показать границу,
за которой агент LLM НЕ верит.

  [человек, русский текст]
        │  внешняя LLM (Azure), SYSTEM PROMPT задаёт целевую ISA → строгий AGENTJSON
        ▼
  AGENTJSON = {goal, entities:[{id,label,levers⊂{mem,pool,config}, deps[]}]}   ← ГИПОТЕЗА
        │  агент. LLM мог переутверждать рычаги/связи — реальность их СРЕЖЕТ
        ▼
  заземление пробами в мир-судью: подтвердить|спрунить рычаги и связи, найти пороги,
  распознать тип по поведенческой сигнатуре, довести до goal.

LLM = парсер интента (грамматику ей даём — это не чит). Истину о поведении даёт мир.
Мир здесь синтетический (стенд вместо docker), судит детерминированно.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

KNOBS = ["mem", "pool", "config"]                       # фиксированная поверхность рычагов (ISA)
BASELINE = {"mem": 4096, "pool": 150, "config": "tuned"}
STRESS = {"mem": 32, "pool": 1, "config": "default"}
MEM_STEPS = [64, 128, 256, 512, 1024, 2048]

TASK_SCHEMA = {
    "goal": "целевой статус, которого хочет человек (обычно 'running')",
    "entities": ("JSON-список объектов, каждый {id, label, levers, deps}. "
                 "levers — подмножество ['mem','pool','config'] (какие крутилки у вещи). "
                 "deps — список id ДРУГИХ сущностей из этого же списка, от которых она зависит."),
}

SYSTEM = (
    "You are the NLU COMPILER (front-end) of an ops agent. The agent does NOT understand human "
    "language — it only has a fixed machine vocabulary. Translate the human request into the "
    "agent's program. Vocabulary: each thing is an ENTITY with platform KNOBS chosen ONLY from "
    "['mem','pool','config'], and DEPENDENCIES on other entities by id. This is a HYPOTHESIS — "
    "the agent will verify every knob and dependency against reality, so propose what is plausible. "
    "Include every entity (including dependencies) in the list with consistent ids. JSON only."
)


# ============================ КОМПИЛЯТОР (живая LLM) =============================

def compile_intent(phrase: str):
    """Человеческая фраза → AGENTJSON. Возвращает (нормализованный agentjson, сырой ответ)."""
    from utils_azure import AzureJSON
    az = AzureJSON()
    raw = az.ask(system=SYSTEM, user=phrase, schema=TASK_SCHEMA)

    ents = {}
    for e in raw.get("entities", []):
        eid = str(e.get("id") or e.get("label") or f"e{len(ents)}")
        ents[eid] = {
            "id": eid,
            "label": str(e.get("label", eid)),
            "levers": [l for l in e.get("levers", []) if l in KNOBS],
            "deps": [str(d) for d in e.get("deps", [])],
        }
    for e in ents.values():                              # отбросить ссылки на несуществующие id
        e["deps"] = [d for d in e["deps"] if d in ents]
    return {"goal": raw.get("goal", "running"), "entities": ents}, raw


# ============================ МИР-СУДЬЯ (синтетический) ==========================

def _truth(label: str) -> dict:
    """Скрытая истина по ярлыку (стенд вместо docker). Агенту НЕдоступна."""
    l = label.lower()
    if any(k in l for k in ("data", "db", "sql", "postgre", "база", "баз", "mysql", "mongo")):
        return {"mem": 512, "needs_config": True, "dep_kw": ("cache", "redis", "кэш", "kv", "memcach")}
    if any(k in l for k in ("cache", "redis", "кэш", "kv", "memcach")):
        return {"mem": 128, "needs_config": False, "dep_kw": ()}
    if any(k in l for k in ("gateway", "api", "proxy", "шлюз", "ingress", "balanc", "балан", "nginx")):
        return {"mem": 256, "needs_config": False, "dep_kw": ("data", "db", "база", "sql", "postgre")}
    return {"mem": 256, "needs_config": False, "dep_kw": ()}


class World:
    def __init__(self, entities: dict):
        self.ent = entities
        self.truth = {eid: _truth(e["label"]) for eid, e in entities.items()}
        # реальные зависимости = заявленные, чей ярлык попадает в dep_kw (остальное — галлюцинация)
        self.real_deps = {
            eid: [d for d in e["deps"]
                  if any(k in entities[d]["label"].lower() for k in self.truth[eid]["dep_kw"])]
            for eid, e in entities.items()
        }
        self.running: set = set()
        self.n_probes = 0

    def observe(self, eid: str, values: dict) -> str:
        """Чистая проба-вердикт (не мутирует running)."""
        self.n_probes += 1
        for d in self.real_deps[eid]:
            if d not in self.running:
                return "dep_down"
        if values["mem"] < self.truth[eid]["mem"]:
            return "oom"
        if self.truth[eid]["needs_config"] and values["config"] != "tuned":
            return "unhealthy"
        return "running"

    def up(self, eid):
        self.running.add(eid)

    def down(self, eid):
        self.running.discard(eid)


# ============================ АГЕНТ (грунтует AGENTJSON) =========================

class Agent:
    def __init__(self, world: World):
        self.w = world
        self.types: dict = {}          # сигнатура -> имя типа (агент придумывает сам)

    def run_task(self, agentjson: dict) -> None:
        ents = agentjson["entities"]
        for eid in self._toposort(ents):
            self._ground_entity(eid, ents)

    @staticmethod
    def _toposort(ents: dict) -> list:
        order, seen = [], set()

        def visit(x):
            if x in seen:
                return
            seen.add(x)
            for d in ents[x]["deps"]:
                visit(d)
            order.append(x)

        for x in ents:
            visit(x)
        return order

    def _ground_entity(self, eid: str, ents: dict) -> None:
        e = ents[eid]
        claimed_lev, claimed_dep = e["levers"], e["deps"]
        print(f"\n  ▼ заземляю '{eid}' (label={e['label']!r})")
        print(f"      LLM заявила: levers={claimed_lev}  deps={claimed_dep}")

        if self.w.observe(eid, BASELINE) != "running":   # без deps baseline должен работать
            print("      [!] baseline не встал — зависимости не подняты, пропускаю")
            return

        # 1) Заземлить РЫЧАГИ: стрессуем один, остальные на baseline; деградация → рычаг реален
        real_lev = []
        for L in claimed_lev:
            v = dict(BASELINE); v[L] = STRESS[L]
            verdict = self.w.observe(eid, v)
            real = verdict != "running"
            print(f"      рычаг {L:6}: стресс → {verdict:9} ⇒ {'РЕАЛЕН' if real else 'СРЕЗАН (галлюцинация)'}")
            if real:
                real_lev.append(L)

        # 2) Заземлить СВЯЗИ: роняем одну (остальные подняты); деградация → связь реальна
        real_dep = []
        for D in claimed_dep:
            self.w.down(D)
            verdict = self.w.observe(eid, BASELINE)
            self.w.up(D)
            real = verdict != "running"
            print(f"      связь {D:6}: уронил → {verdict:9} ⇒ {'РЕАЛЬНА' if real else 'СРЕЗАНА (галлюцинация)'}")
            if real:
                real_dep.append(D)

        # 3) Пороги для реальных ordered-рычагов (mem/pool) — удвоением
        good_config = "tuned" if "config" in real_lev else "default"
        thresholds = {}
        for L in [x for x in real_lev if x in ("mem", "pool")]:
            for step in MEM_STEPS:
                v = dict(BASELINE); v["config"] = good_config; v[L] = step
                if self.w.observe(eid, v) == "running":
                    thresholds[L] = step
                    break

        # 4) Поведенческая сигнатура → распознать тип (агент именует сам)
        sig = (tuple(sorted(real_lev)), tuple(sorted(thresholds.items())), good_config == "tuned")
        if sig in self.types:
            tname, fresh = self.types[sig], False
        else:
            tname = f"T{len(self.types)}"
            self.types[sig] = tname
            fresh = True

        # 5) Довести до goal
        final = dict(BASELINE); final["config"] = good_config
        for L, t in thresholds.items():
            final[L] = t
        phase = self.w.observe(eid, final)
        if phase == "running":
            self.w.up(eid)

        print(f"      ⇒ ЗАЗЕМЛЕНО: levers={real_lev} deps={real_dep} пороги={thresholds}")
        print(f"      ⇒ тип {tname} ({'НОВЫЙ — выучен' if fresh else 'УЗНАН — переиспользован'}), "
              f"цель: {phase}")


# ============================ ДЕМО ===============================================

def _run(phrase: str) -> None:
    print("=" * 72)
    print(f"ЧЕЛОВЕК: {phrase!r}")
    print("=" * 72)
    agentjson, raw = compile_intent(phrase)
    print("\n— ВЫХОД КОМПИЛЯТОРА (сырой ответ LLM) —")
    print(f"  {raw}")
    print("\n— AGENTJSON (нормализованный, язык агента) —")
    print(f"  goal = {agentjson['goal']!r}")
    for eid, e in agentjson["entities"].items():
        print(f"  entity {eid:8} label={e['label']!r:18} levers={e['levers']} deps={e['deps']}")

    print("\n— АГЕНТ ГРУНТУЕТ (под капотом) —")
    world = World(agentjson["entities"])
    Agent(world).run_task(agentjson)
    print(f"\n  поднято: {sorted(world.running)} | проб в мир: {world.n_probes} | LLM-вызовов: 1 (компиляция)")


def main() -> None:
    print("=== NLU+LLM компилятор → AGENTJSON → агент грунтует (живой Azure) ===\n")
    for phrase in [
        "Подними новую базу данных PostgreSQL с кэшем Redis и убедись, что она здорова.",
        "Разверни API-шлюз перед этой базой данных.",
    ]:
        _run(phrase)
        print()


if __name__ == "__main__":
    main()
