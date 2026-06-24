"""
domain_gen.py — MetaDomain → текст domain.pddl + problem.pddl; решение через FD.

Порождает форму рукописного v2-домена (types/predicates/actions), но ИЗ РЕЕСТРА.
Граница #8: чистый STRIPS — без :functions, :metric, чисел. Значение рычага (память N,
config X) даёт агент-политика, НЕ план; планировщик лишь секвенирует set_*→deploy и
доказывает разрешимость сгенерённого домена.

Унифицированное правило предусловий (выводит обе v2-формы из (requires, establishes)):
  precond = (not <establishes_i>)  ∧  <requires_j>
    set_mem (establ=[mem_ok], req=[])            → (not (mem_ok ?s))
    deploy  (establ=[running],  req=deploy_reqs) → (and (not (running ?s)) (mem_ok ?s) (config_ok ?s))
"""

import os

from devops_agent.v3.meta_schema import MetaDomain

_OUT_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def _precondition(establishes: list, requires: list) -> str:
    terms = [f"(not ({e} ?s))" for e in establishes] + [f"({p} ?s)" for p in requires]
    if len(terms) == 1:
        return terms[0]
    return "(and " + " ".join(terms) + ")"


def generate_domain(domain: MetaDomain, out_dir: str = _OUT_DIR) -> str:
    """Сериализовать domain.pddl (детерминированный порядок) и вернуть путь."""
    os.makedirs(out_dir, exist_ok=True)
    preds = sorted(domain.predicates.keys())
    ivs = sorted(domain.interventions.keys())
    req = sorted(domain.deploy_requires)

    lines = [
        "(define (domain devops)",
        "  (:requirements :strips :typing :negative-preconditions)",
        "  (:types service - object)",
        "",
        "  (:predicates",
    ]
    pred_atoms = ["(running ?s - service)"] + [f"({p} ?s - service)" for p in preds]
    for i, atom in enumerate(pred_atoms):
        lines.append(f"    {atom}{')' if i == len(pred_atoms) - 1 else ''}")

    # Минтнутые set_* действия (establishes ровно один предикат)
    for name in ivs:
        iv = domain.interventions[name]
        lines += [
            "",
            f"  (:action {name}",
            "    :parameters (?s - service)",
            f"    :precondition {_precondition(iv.establishes, iv.requires)}",
            f"    :effect ({iv.establishes[0]} ?s))",
        ]

    # Терминальный deploy: precond = (not running) ∧ deploy_requires; effect = running
    lines += [
        "",
        "  (:action deploy",
        "    :parameters (?s - service)",
        f"    :precondition {_precondition(['running'], req)}",
        "    :effect (running ?s))",
        ")",
        "",
    ]

    path = os.path.join(out_dir, "domain.pddl")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def generate_problem(goal_service: str, out_dir: str = _OUT_DIR) -> str:
    """Тривиальная проблема: (:objects <goal> - service)(:init)(:goal (running <goal>))."""
    os.makedirs(out_dir, exist_ok=True)
    content = "\n".join([
        "(define (problem devops-p)",
        "  (:domain devops)",
        "",
        f"  (:objects {goal_service} - service)",
        "",
        "  (:init)",
        "",
        f"  (:goal (running {goal_service}))",
        ")",
        "",
    ])
    path = os.path.join(out_dir, "problem.pddl")
    with open(path, "w") as f:
        f.write(content)
    return path


def solve(domain_path: str, problem_path: str) -> list | None:
    """
    Решить сгенерённый домен. Движок — как в devops_agent/bios.py::plan().
    Импорты unified_planning внутри функции → модуль импортируется без зависимостей (для py_compile).
    """
    from unified_planning.engines import PlanGenerationResultStatus
    from unified_planning.io import PDDLReader
    from unified_planning.shortcuts import OneshotPlanner, get_environment

    get_environment().credits_stream = None
    p = PDDLReader().parse_problem(domain_path, problem_path)
    with OneshotPlanner(name="fast-downward-opt") as planner:
        result = planner.solve(p)

    if result.status in (
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        PlanGenerationResultStatus.SOLVED_SATISFICING,
    ):
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Keystone self-test: домен ГЕНЕРИТСЯ (не пишется) и решается FD.
#   seed (пусто)        → план = [deploy]
#   learned (mem+config) → план = [set_mem, set_config, deploy]
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import sys

    from devops_agent.v3.meta_schema import Intervention, MetaDomain, Predicate

    print("=== domain_gen keystone (домен генерится из реестра + решается FD) ===\n")
    errors = []

    # --- SEED: пустой домен (ни mem_ok, ни config_ok) ---
    seed = MetaDomain()
    d0 = generate_domain(seed, out_dir=os.path.join(_OUT_DIR, "seed"))
    p0 = generate_problem("svc_x", out_dir=os.path.join(_OUT_DIR, "seed"))
    print("--- SEED domain.pddl ---")
    print(open(d0).read())
    plan0 = solve(d0, p0)
    print(f"SEED plan: {plan0}")
    if not plan0 or len(plan0) != 1 or "deploy" not in plan0[0]:
        errors.append(f"SEED: ожидали [deploy], got {plan0}")

    # --- LEARNED: mem + config выучены (реконструкция рукописного v2-домена) ---
    learned = MetaDomain(
        predicates={
            "mem_ok": Predicate("mem_ok", "memory"),
            "config_ok": Predicate("config_ok", "config"),
        },
        interventions={
            "set_mem": Intervention("set_mem", establishes=["mem_ok"], binds_actuator="memory"),
            "set_config": Intervention("set_config", establishes=["config_ok"], binds_actuator="config"),
        },
        deploy_requires=["mem_ok", "config_ok"],
    )
    d1 = generate_domain(learned, out_dir=os.path.join(_OUT_DIR, "learned"))
    p1 = generate_problem("svc_x", out_dir=os.path.join(_OUT_DIR, "learned"))
    print("\n--- LEARNED domain.pddl ---")
    print(open(d1).read())
    plan1 = solve(d1, p1)
    print(f"LEARNED plan: {plan1}")
    if not plan1 or len(plan1) != 3:
        errors.append(f"LEARNED: ожидали 3 шага, got {plan1}")
    else:
        joined = " ".join(plan1)
        for need in ("set_mem", "set_config", "deploy"):
            if need not in joined:
                errors.append(f"LEARNED: нет {need} в плане {plan1}")

    # Гвард #8: в сгенерённом домене нет чисел/метрики
    if ":metric" in open(d1).read() or ":functions" in open(d1).read():
        errors.append("ГВАРД #8: в домене появились :metric/:functions")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("domain_gen keystone OK — домен сгенерён и решён FD (seed + learned).")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v3.domain_gen --selftest")
