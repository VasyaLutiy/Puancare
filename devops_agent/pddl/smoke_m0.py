"""
M0 smoke — проверяет что домен парсится и FD находит план.
После #8 домен чистый STRIPS: никаких numeric-fluents/action-costs.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
domain_path = os.path.join(script_dir, "domain.pddl")
problem_path = os.path.join(script_dir, "problem.pddl")

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment
from unified_planning.engines import PlanGenerationResultStatus

get_environment().credits_stream = None

print("=== M0 Smoke Test: STRIPS domain ===\n")

print("[1] Parsing domain + problem...")
p = PDDLReader().parse_problem(domain_path, problem_path)
print(f"    Kind features: {sorted(str(p.kind).split(chr(10)))}\n")

print("[2] fast-downward-opt supports() check...")
with OneshotPlanner(name="fast-downward-opt") as planner:
    supported = planner.supports(p.kind)
    print(f"    supports(p.kind) = {supported}")
    if not supported:
        print("=== DoD: FAIL — supports()==False ===")
        sys.exit(1)
    r = planner.solve(p)

print(f"\n[3] Result:")
print(f"    Status : {r.status}")
print(f"    Plan   : {r.plan}")

_OK = {PlanGenerationResultStatus.SOLVED_OPTIMALLY, PlanGenerationResultStatus.SOLVED_SATISFICING}
if r.status in _OK:
    print(f"\n=== DoD: PASS — {r.status.name}, supports()==True ===")
    sys.exit(0)
else:
    print(f"\n=== DoD: FAIL — got {r.status} ===")
    sys.exit(1)
