"""
BIOS-state: читаемая/редактируемая модель мира.

Поля:
  services         — наблюдаемые фичи сервисов (agent_view)
  buckets          — доступные бакеты памяти [MiB], по возрастанию
  unsafe_mem       — класс-правила: {(workload_class, bucket_mib)}
  unsafe_svc       — per-service исключения: {(svc_name, bucket_mib)}
  known_safe_class — (wc, bucket) подтверждены реальным running
  bad_config       — {(svc_name, config_opt)} плохих конфигов
  reverse_index    — симптом → [причина, ...] (амортизация LLM-оракула)

plan(bios, goal_service) → list[str] | None
  Строит problem.pddl из BIOS, решает через FD-opt.
  Статус SOLVED_OPTIMALLY.
"""

import os
import sys
from dataclasses import dataclass, field

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment

from devops_agent.services import MEM_BUCKETS, agent_view, all_service_names

get_environment().credits_stream = None

_PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


# ---------------------------------------------------------------------------
# BIOS State
# ---------------------------------------------------------------------------

@dataclass
class BiosState:
    """
    Один экземпляр живёт на весь сеанс — знания накапливаются между эпизодами.
    """
    services: dict[str, dict]                         # {svc_name: agent_view(svc_name)}
    buckets: list[int]                                # MEM_BUCKETS по возрастанию
    unsafe_mem: set[tuple[str, int]]                 # {(wc, bucket)} — класс-правила
    unsafe_svc: set[tuple[str, int]]                 # {(svc_name, bucket)} — per-service
    known_safe_class: set[tuple[str, int]]           # {(wc, bucket)} confirmed running
    bad_config: set[tuple[str, str]]                 # {(svc_name, config_opt)}
    incumbent_config: dict[str, str]                 # {svc_name: current applied config}
    reverse_index: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        svcs = {n: agent_view(n) for n in svc_names}
        return cls(
            services=svcs,
            buckets=sorted(MEM_BUCKETS),
            unsafe_mem=set(),
            unsafe_svc=set(),
            known_safe_class=set(),
            bad_config=set(),
            incumbent_config={
                n: svcs[n]["current_config"]
                for n in svc_names
                if "current_config" in svcs[n]
            },
        )

    def mark_unsafe(self, workload_class: str, bucket_mib: int) -> None:
        self.unsafe_mem.add((workload_class, bucket_mib))

    def mark_unsafe_svc(self, svc_name: str, bucket_mib: int) -> None:
        self.unsafe_svc.add((svc_name, bucket_mib))

    def mark_bad_config(self, svc_name: str, config_opt: str) -> None:
        self.bad_config.add((svc_name, config_opt))

    def confirm_safe(self, workload_class: str, bucket_mib: int) -> None:
        self.known_safe_class.add((workload_class, bucket_mib))

    def is_class_confirmed_safe(self, workload_class: str, bucket_mib: int) -> bool:
        return (workload_class, bucket_mib) in self.known_safe_class

    def is_unsafe_for_deploy(self, svc_name: str, bucket_mib: int) -> bool:
        wc = self.services[svc_name]["workload_class"]
        return (
            (wc, bucket_mib) in self.unsafe_mem
            or (svc_name, bucket_mib) in self.unsafe_svc
        )

    def is_bad_config(self, svc_name: str, config_opt: str) -> bool:
        return (svc_name, config_opt) in self.bad_config

    def safe_buckets_for(self, svc_name: str) -> list[int]:
        return [b for b in self.buckets if not self.is_unsafe_for_deploy(svc_name, b)]

    def _choose_config(self, svc: str) -> str | None:
        """
        Выбирает конфиг для world.apply (политика агента, не планировщик).
        Приоритет: наследство (incumbent) если не known-bad, иначе
        детерминированная непровальная альтернатива.
        """
        opts = self.services[svc].get("config_options")
        if not opts:
            return "cfg_default"                        # сервис без config → always-on dummy
        inc = self.incumbent_config.get(svc)
        if inc is not None and not self.is_bad_config(svc, inc):
            return inc                                  # действуем по наследству
        for o in sorted(opts):                          # детерминированная альтернатива
            if not self.is_bad_config(svc, o):
                return o
        return None                                     # все known-bad → нет валидного config


# ---------------------------------------------------------------------------
# Planner (BIOS → problem.pddl → FD-opt)
# ---------------------------------------------------------------------------

def plan(bios: BiosState, goal_service: str) -> list[str] | None:
    """
    Возвращает оптимальный план или None.
    Пример с config: ['set_config(svc_e, good)', 'set_mem(svc_e)', 'deploy(svc_e, good)']
    Пример без config: ['set_config(svc_a, cfg_default)', 'set_mem(svc_a)', 'deploy(svc_a, cfg_default)']
    """
    from devops_agent.pddl.problem_builder import ProblemBuilder

    try:
        pb = ProblemBuilder(bios, _PDDL_DIR)
        problem_path = pb.build(goal_service)
    except ValueError:
        return None

    domain_path = os.path.join(_PDDL_DIR, "domain.pddl")
    p = PDDLReader().parse_problem(domain_path, problem_path)
    with OneshotPlanner(name="fast-downward-opt") as planner:
        result = planner.solve(p)

    if result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== BIOS self-test (M1: PDDL domain) ===\n")
    errors = []

    # 1. Без config: план = [set_config(svc_a, cfg_default), set_mem(svc_a), deploy(svc_a, cfg_default)]
    bios1 = BiosState.initial(["svc_a"])
    steps1 = plan(bios1, "svc_a")
    print(f"Test 1 (svc_a, no config): {steps1}")
    if steps1 is None or len(steps1) != 3:
        errors.append(f"Test1: expected 3-step plan, got {steps1}")
    elif not any("set_config" in s for s in steps1):
        errors.append(f"Test1: no set_config in plan: {steps1}")
    elif not any("set_mem" in s for s in steps1):
        errors.append(f"Test1: no set_mem in plan: {steps1}")
    else:
        print("  → OK ✓")

    # 2. С config (naive): план = [set_config(svc_e, ?), set_mem(svc_e), deploy(svc_e, ?)]
    bios2 = BiosState.initial(["svc_e"])
    steps2 = plan(bios2, "svc_e")
    print(f"\nTest 2 (svc_e naive): {steps2}")
    if steps2 is None or len(steps2) != 3:
        errors.append(f"Test2: expected 3-step plan, got {steps2}")
    elif not any("set_config" in s for s in steps2):
        errors.append(f"Test2: no set_config in plan: {steps2}")
    else:
        print("  → OK ✓")

    # 3. Известный порог: bad_config=(svc_e,bad), unsafe=[64,128,256] → good + 512
    bios3 = BiosState.initial(["svc_e"])
    for bad in [64, 128, 256]:
        bios3.mark_unsafe("standard", bad)
    bios3.mark_bad_config("svc_e", "bad")
    steps3 = plan(bios3, "svc_e")
    print(f"\nTest 3 (known threshold, bad_config={{svc_e,bad}}): {steps3}")
    if steps3 is None or len(steps3) != 3:
        errors.append(f"Test3: expected 3-step plan, got {steps3}")
    elif not any("good" in s for s in steps3):
        errors.append(f"Test3: expected 'good' config, got {steps3}")
    else:
        print("  → OK ✓")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All tests passed. BIOS PDDL plan: OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
