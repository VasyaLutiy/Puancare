import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
domain_path = os.path.join(script_dir, "domain.pddl")
problem_path = os.path.join(script_dir, "problem.pddl")

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

print("=== M0 Smoke Test: Numeric PDDL Stack ===\n")

print("[1] Parsing domain + problem...")
reader = PDDLReader()
p = reader.parse_problem(domain_path, problem_path)
print(f"    Problem kind: {p.kind}")
print(f"    Features: {list(p.kind.features)}\n")

print("[2] Solving with optimality guarantee...")
with OneshotPlanner(problem_kind=p.kind, optimality_guarantee="SOLVED_OPTIMALLY") as planner:
    print(f"    Engine selected: {planner.name}")
    r = planner.solve(p)

print(f"\n[3] Result:")
print(f"    Status : {r.status}")
print(f"    Plan   : {r.plan}")
print(f"    Engine : {planner.name}")

if str(r.status) == "PlanGenerationResultStatus.SOLVED_OPTIMALLY":
    print("\n=== DoD: PASS — SOLVED_OPTIMALLY ===")
    sys.exit(0)
else:
    print(f"\n=== DoD: FAIL — got {r.status} ===")
    sys.exit(1)
