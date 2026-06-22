import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
domain_path = os.path.join(script_dir, "domain.pddl")
problem_path = os.path.join(script_dir, "problem.pddl")

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

print("=== M0 Smoke Test (v2): Numeric PDDL — FD-opt ===\n")

print("[1] Parsing domain + problem...")
reader = PDDLReader()
p = reader.parse_problem(domain_path, problem_path)
print(f"    Problem kind features:")
for f in sorted(str(p.kind).split("\n")):
    if f.strip():
        print(f"      {f.strip()}")
print()

print("[2] Solving SOLVED_OPTIMALLY (fast-downward-opt)...")
with OneshotPlanner(name="fast-downward-opt") as planner:
    engine_name = planner.name
    print(f"    Engine selected: {engine_name}")
    r = planner.solve(p)

print(f"\n[3] Result:")
print(f"    Status : {r.status}")
print(f"    Engine : {engine_name}")
print(f"    Plan   : {r.plan}")

if r.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
    print("\n=== DoD: PASS — SOLVED_OPTIMALLY ===")
    sys.exit(0)
else:
    print(f"\n=== DoD: FAIL — got {r.status} ===")
    sys.exit(1)
