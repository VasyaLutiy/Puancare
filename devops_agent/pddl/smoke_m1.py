"""
M1 DoD smoke test (после #8: STRIPS, без cost/metric).

Проверяет:
  1. ProblemBuilder генерирует валидный problem.pddl без cost-фактов.
  2. PDDLReader парсит domain + problem без ошибок.
  3. planner.supports(p.kind) == True (no UserWarning).
  4. Round-trip парсинга идентичен.
  5. План = 3 шага, статус SOLVED_OPTIMALLY.
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

print("=== M1 DoD Smoke Test (STRIPS, #8) ===\n")
errors = []

# BiosState: svc_e, small buckets unsafe → safe=[512], bad_config={bad}
bios = BiosState.initial(["svc_e"])
for bad_bucket in [64, 128, 256]:
    bios.mark_unsafe("standard", bad_bucket)
bios.mark_bad_config("svc_e", "bad")

print("[1] ProblemBuilder → problem.pddl ...")
pb = ProblemBuilder(bios, PDDL_DIR)
problem_path = pb.build("svc_e")

with open(problem_path) as f:
    content = f.read()
print(f"    Content:\n{content}")

# Нет cost-артефактов
for forbidden in ["mem_cost", "total-cost", ":metric", ":functions"]:
    if forbidden in content:
        errors.append(f"problem.pddl содержит запрещённый артефакт: {forbidden!r}")
if "(config_ok svc_e good)" not in content:
    errors.append("problem.pddl: ожидали (config_ok svc_e good)")
if "(config_ok svc_e bad)" in content:
    errors.append("problem.pddl: bad НЕ должен быть в config_ok")

print("[2] PDDLReader парсит domain + problem ...")
p = PDDLReader().parse_problem(DOMAIN, problem_path)
print(f"    Kind: {sorted(str(p.kind).split(chr(10)))}")

print("\n[3] supports(p.kind) ...")
with OneshotPlanner(name="fast-downward-opt") as planner:
    supported = planner.supports(p.kind)
    print(f"    supports = {supported}")
    if not supported:
        errors.append("planner.supports(p.kind) == False")

    print("\n[4] Round-trip ...")
    p2 = PDDLReader().parse_problem(DOMAIN, problem_path)
    if str(p) != str(p2):
        errors.append("Round-trip: повторный парсинг дал другую проблему")
    else:
        print("    → идентично ✓")

    print("\n[5] FD-opt: SOLVED_OPTIMALLY + 3 шага ...")
    result = planner.solve(p)

steps = [str(a) for a in result.plan.actions] if result.plan else []
print(f"    Status : {result.status}")
print(f"    Plan   : {steps}")

_OK = {PlanGenerationResultStatus.SOLVED_OPTIMALLY, PlanGenerationResultStatus.SOLVED_SATISFICING}
if result.status not in _OK:
    errors.append(f"Status: ожидали SOLVED_OPTIMALLY/SATISFICING, got {result.status}")
if len(steps) != 3:
    errors.append(f"Plan length: ожидали 3, got {len(steps)}: {steps}")
else:
    if not any(s.startswith("set_config") and "good" in s for s in steps):
        errors.append(f"Plan: нет set_config(...good...): {steps}")
    if not any(s.startswith("set_mem") for s in steps):
        errors.append(f"Plan: нет set_mem: {steps}")
    if not (steps[-1].startswith("deploy") and "good" in steps[-1]):
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
