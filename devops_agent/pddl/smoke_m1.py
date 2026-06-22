"""
M1 DoD smoke test.

Проверяет:
  1. ProblemBuilder генерирует валидный problem.pddl из BiosState.
  2. PDDLReader парсит domain.pddl + problem.pddl без ошибок.
  3. Повторный парсинг того же файла даёт ту же проблему (round-trip).
  4. План известного порога = 3 шага:
       set_config(svc_e, good) → set_mem(svc_e) → deploy(svc_e, good)
     статус SOLVED_OPTIMALLY.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment
from unified_planning.engines import PlanGenerationResultStatus

from devops_agent.bios import BiosState
from devops_agent.pddl.problem_builder import ProblemBuilder

get_environment().credits_stream = None

PDDL_DIR = os.path.dirname(os.path.abspath(__file__))
DOMAIN = os.path.join(PDDL_DIR, "domain.pddl")

print("=== M1 DoD Smoke Test ===\n")
errors = []

# BiosState с известным порогом: svc_e, small buckets unsafe → safe=[512], bad_config={bad}
bios = BiosState.initial(["svc_e"])
for bad_bucket in [64, 128, 256]:
    bios.mark_unsafe("standard", bad_bucket)
bios.mark_bad_config("svc_e", "bad")

print("[1] ProblemBuilder → problem.pddl ...")
pb = ProblemBuilder(bios, PDDL_DIR)
problem_path = pb.build("svc_e")

with open(problem_path) as f:
    content = f.read()
print(f"    Wrote: {problem_path}")
print(f"    Content:\n{content}")

# Ожидаем: mem_cost svc_e = 512, config_ok svc_e good (не bad)
if "(= (mem_cost svc_e) 512)" not in content:
    errors.append("problem.pddl: ожидали (= (mem_cost svc_e) 512)")
if "(config_ok svc_e good)" not in content:
    errors.append("problem.pddl: ожидали (config_ok svc_e good)")
if "(config_ok svc_e bad)" in content:
    errors.append("problem.pddl: bad НЕ должен быть в config_ok")

print("[2] PDDLReader парсит domain + problem ...")
p = PDDLReader().parse_problem(DOMAIN, problem_path)
print(f"    Kind features: {sorted(str(p.kind).split(chr(10)))}")

print("\n[3] Round-trip: повторный парсинг того же файла ...")
p2 = PDDLReader().parse_problem(DOMAIN, problem_path)
if str(p) != str(p2):
    errors.append("Round-trip: повторный парсинг дал другую проблему")
else:
    print("    → идентично ✓")

print("\n[4] FD-opt: SOLVED_OPTIMALLY + 3 шага ...")
with OneshotPlanner(name="fast-downward-opt") as planner:
    result = planner.solve(p)

steps = [str(a) for a in result.plan.actions] if result.plan else []
print(f"    Status : {result.status}")
print(f"    Plan   : {steps}")

if result.status != PlanGenerationResultStatus.SOLVED_OPTIMALLY:
    errors.append(f"Status: ожидали SOLVED_OPTIMALLY, got {result.status}")
if len(steps) != 3:
    errors.append(f"Plan length: ожидали 3, got {len(steps)}: {steps}")
else:
    # set_config и set_mem независимы → порядок между ними не фиксирован доменом.
    # Проверяем наличие обоих + deploy последним.
    has_set_config = any(s.startswith("set_config") and "good" in s for s in steps)
    has_set_mem = any(s.startswith("set_mem") for s in steps)
    last_is_deploy = steps[-1].startswith("deploy") and "good" in steps[-1]
    if not has_set_config:
        errors.append(f"Plan: нет set_config(... good ...): {steps}")
    if not has_set_mem:
        errors.append(f"Plan: нет set_mem: {steps}")
    if not last_is_deploy:
        errors.append(f"Plan: последний шаг не deploy(...good...): {steps[-1]!r}")

print()
if errors:
    print("=== DoD: FAIL ===")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    order = " → ".join(steps)
    print(f"=== DoD: PASS — {order} ===")
    sys.exit(0)
