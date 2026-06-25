"""
loop.py — ПЕТЛЯ агента на новом графе (A-2). Доменно-СЛЕПАЯ по построению.

plan(граф→PDDL→FD) → act(sandbox) → симптом → (кэш | LLM-гипотеза) → mint в граф →
generic-стратегия по KIND → docker судит → амортизация. Рефьют: если симптом приписан
рычагу, чья стратегия его НЕ consumes → kind неверен → переспрос LLM с уликой.

ГРАНИЦА НЕ-ХАРДКОДА: тело run_task() не содержит ни одного доменного литерала (нет mem/pool/
config/oom/...). Что значит симптом — решает LLM; как искать — generic-стратегия по kind;
прав ли LLM — судит docker. Поверхность рычагов ('API') передаётся снаружи, не вшита.
"""

import sys
from dataclasses import dataclass, field

from devops_agent.v4.graph import KnowledgeGraph, MetaType, Rel
from devops_agent.v4.projection import kg_to_pddl, solve
from devops_agent.v4.strategies import make_strategy


@dataclass
class TaskResult:
    entity: str
    success: bool
    trials: int = 0
    llm_calls: int = 0
    refutes: int = 0
    error: str = ""
    values: dict = field(default_factory=dict)   # выученные значения рычагов на момент успеха


def _seed_graph(g: KnowledgeGraph, entity: str, goal_status: str) -> str:
    """Минимальный seed: сущность + целевой статус + терминальная интервенция (без требований)."""
    g.add_node(entity, MetaType.ENTITY)
    g.add_node(goal_status, MetaType.STATUS)
    g.add_edge(entity, Rel.HAS_GOAL, goal_status)
    prov = f"provision_{entity}"
    g.add_node(prov, MetaType.INTERVENTION)
    g.add_edge(prov, Rel.ESTABLISHES, goal_status)
    return prov


def _mint_lever(g: KnowledgeGraph, entity: str, prov: str, lever: str, ntype: MetaType) -> None:
    """Вшить рычаг в граф: resource/setting + HAS + set_<lever> establishes + provision REQUIRES."""
    g.add_node(lever, ntype)
    g.add_edge(entity, Rel.HAS, lever)
    g.add_node(f"set_{lever}", MetaType.INTERVENTION)
    g.add_edge(f"set_{lever}", Rel.ESTABLISHES, lever)
    g.add_edge(prov, Rel.REQUIRES, lever)


def _cfg_for(prop: dict, symptom: str, levers: dict) -> dict:
    """Конфиг generic-стратегии из предложения LLM (доменно-слепо)."""
    kind = prop["kind"]
    if kind == "ordered_monotone":
        return {"trigger": symptom}
    if kind == "categorical":
        return {"trigger": symptom, "options": levers[prop["lever"]].get("options", [])}
    if kind == "bounded":
        return {"low_symptom": prop["low_symptom"], "high_symptom": prop["high_symptom"]}
    raise ValueError(kind)


def run_task(g, sandbox, proposer, entity, levers, defaults, goal_status="running",
             max_trials=40, verbose=True, learned=None, log=print):
    """
    g: KnowledgeGraph (растёт). levers: {name:{type, options?}} — поверхность ('API').
    defaults: {lever: стартовое значение}. learned: кросс-задачный кэш symptom->prop (амортизация).
    Тело — доменно-слепое: домен только в данных (levers/defaults/sandbox), не в логике.
    """
    learned = learned if learned is not None else {}
    prov = _seed_graph(g, entity, goal_status)
    values = dict(defaults)
    history = {l: [] for l in levers}
    strat = {}            # lever -> Strategy
    covered = {}          # symptom -> lever (внутри задачи)
    res = TaskResult(entity=entity, success=True)

    # восстановить стратегии из выученного (амортизация: 0 LLM на знакомый симптом)
    def _install(prop, symptom):
        lever = prop["lever"]
        ntype = MetaType.SETTING if levers[lever]["type"] == "categorical" else MetaType.RESOURCE
        if lever not in {e.dst for e in g.out(entity, Rel.HAS)}:
            _mint_lever(g, entity, prov, lever, ntype)
        strat[lever] = make_strategy(prop["kind"], **_cfg_for(prop, symptom, levers))
        covered[symptom] = lever
        return lever

    for sym, prop in learned.items():        # перенос знаний (без LLM)
        _install(prop, sym)

    while True:
        # PLAN: проекция растущего графа → PDDL → FD (доказывает разрешимость; шлюз достижимости)
        dpath, ppath = kg_to_pddl(g, entity)
        if solve(dpath, ppath) is None:
            res.success = False; res.error = "no_plan"; return res

        # ACT
        obs = sandbox.provision(entity, knobs={lv: values[lv] for lv in levers})
        res.trials += 1
        sym = obs.phase
        if verbose:
            log(f"  try {dict((k, values[k]) for k in levers)} → {sym}")
        if sym == goal_status:
            res.values = dict(values)
            return res
        if res.trials > max_trials:
            res.success = False; res.error = "max_trials"; return res

        # ATTRIBUTE: знакомый симптом → из кэша; иначе спросить LLM (гипотеза)
        if sym in covered and strat[covered[sym]].consumes(sym):
            lever = covered[sym]
        else:
            prop = proposer.propose(sym, levers, {l: strat[l].__class__.__name__ for l in strat})
            res.llm_calls += 1
            lever = prop["lever"]
            if lever in strat and not strat[lever].consumes(sym):
                # РЕФЬЮТ: рычаг известен, но его kind не закрывает этот симптом → docker опроверг гипотезу
                res.refutes += 1
                evidence = (f"lever {lever!r}: applying kind {strat[lever].__class__.__name__} "
                            f"did not resolve it — new symptom {sym!r} appeared on the SAME lever "
                            f"(known symptoms there: {[s for s, l in covered.items() if l == lever]}). "
                            f"It is likely NOT monotone — reconsider the kind (maybe a safe window).")
                prop = proposer.propose(sym, levers, {l: strat[l].__class__.__name__ for l in strat},
                                        refute=evidence)
                res.llm_calls += 1
                if verbose:
                    log(f"  ⟂ REFUTE: {lever} → пересмотр kind → {prop['kind']}  (docker переспорил LLM)")
            lever = _install(prop, sym)
            learned[sym] = prop
            if verbose and res.refutes == 0:
                log(f"  ? cold: {sym!r} → LLM: lever={lever!r} kind={prop['kind']!r}")

        # STEP: продвинуть значение рычага его generic-стратегией
        history[lever].append((values[lever], sym))
        nv = strat[lever].next_value(history[lever])
        if nv is None:
            res.success = False; res.error = f"exhausted:{lever}"; return res
        values[lever] = nv


# ---------------------------------------------------------------------------
# Живой прогон: ПУСТОЙ граф → петля растит его сама, docker судит, есть рефьют.
# ---------------------------------------------------------------------------

def _run_live() -> None:
    from devops_agent.v4.proposer import Proposer
    from devops_agent.v4.sandbox import Sandbox

    # Поверхность рычагов ('API' — дана, без семантики) и truth среды (скрыт от агента).
    LEVERS = {
        "mem": {"type": "numeric"},
        "pool": {"type": "numeric"},
        "config": {"type": "categorical", "options": ["good", "bad"]},
    }
    DEFAULTS = {"mem": 64, "pool": 64, "config": "bad"}
    # truth: footprint(mem), good_config, окно pool [80,100] — нарочно так, что удвоение
    # 64→128 ПЕРЕЛЕТАЕТ окно → форсит рефьют (это дизайн СРЕДЫ, не агента).
    TRUTH = {"app": {"footprint": 350, "good_config": "good", "pool_lo": 80, "pool_hi": 100}}

    sb = Sandbox(TRUTH, network="v4loop")
    sb.prefetch(); sb.reset()
    g = KnowledgeGraph()

    print("=== A-2 ЖИВАЯ ПЕТЛЯ на новом графе (LLM + docker, граф стартует ПУСТОЙ) ===\n")
    print(f"seed-граф: nodes={len(g.nodes)} edges={len(g.edges)}  (пусто — ни одного измерения)")
    print(f"поверхность рычагов (дана как 'API', без смысла): {list(LEVERS)}\n")
    try:
        proposer = Proposer()
        learned = {}
        r = run_task(g, sb, proposer, "app", LEVERS, DEFAULTS, learned=learned)
    finally:
        sb.teardown()

    print("\n--- выученный граф (вырос из провалов) ---")
    print(g.to_json())
    print(f"\nИТОГ: success={r.success} trials={r.trials} LLM-вызовов={r.llm_calls} "
          f"рефьютов={r.refutes} (токенов={proposer.total_tokens})")
    print(f"выучено symptom→{{lever,kind}}: " +
          str({s: (p['lever'], p['kind']) for s, p in learned.items()}))

    # анти-хардкод гвард: в логике стратегий и теле петли нет доменных литералов
    import os
    src_strat = open(os.path.join(os.path.dirname(__file__), "strategies.py")).read().lower()
    body = open(os.path.join(os.path.dirname(__file__), "loop.py")).read()
    run_task_body = body[body.index("def run_task"):body.index("def _run_live")].lower()
    leaks = [t for t in ("mem", "pool", "config", "oom", "footprint") if t in src_strat or t in run_task_body]
    print(f"\nанти-хардкод гвард: доменные литералы в strategies.py/run_task() → {leaks or 'НЕТ ✓'}")

    ok = (r.success and r.refutes >= 1 and not leaks and len(g.nodes) > 1)
    print("\n" + "=" * 64)
    print("ВЕРДИКТ: " + ("НЕ ХАРДКОД ✓ — пустой граф вырос сам через LLM+docker, "
                          "docker опроверг LLM (рефьют), логика доменно-слепа."
                          if ok else "НЕ ДОКАЗАНО ✗ — см. выше (рефьют/успех/гвард)."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--live" in sys.argv:
        _run_live()
    else:
        print("Usage: python -m devops_agent.v4.loop --live   (нужен docker + Azure .env)")
