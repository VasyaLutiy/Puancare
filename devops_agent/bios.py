"""
BIOS-state: читаемая/редактируемая модель мира.

Поля:
  services         — наблюдаемые фичи сервисов (agent_view)
  mem_threshold    — {workload_class: working_mib} выученные пороги по классу
  mem_threshold_svc — {svc_name: working_mib} per-service исключения (карвинг)
  incumbent_config — {svc_name: current applied config}
  bad_config       — {(svc_name, config_opt)} плохих конфигов
  reverse_index    — симптом → [причина, ...] (амортизация LLM-оракула)

Константы:
  MEM_BASE    — начальный размер удвоения [MiB]
  MEM_CEILING — жёсткий потолок [MiB]; агент возвращает no_plan при превышении

plan(bios, goal_service) → list[str] | None
  Проверяет достижимость по config через FD; память — за агентом.
"""

import os
import sys
from dataclasses import dataclass, field

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment

from devops_agent.services import agent_view, all_service_names

get_environment().credits_stream = None

_PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")

MEM_BASE: int = 64       # MiB, стартовый размер удвоения
MEM_CEILING: int = 16384  # MiB (16 GiB), жёсткий потолок


# ---------------------------------------------------------------------------
# BIOS State
# ---------------------------------------------------------------------------

@dataclass
class BiosState:
    """
    Один экземпляр живёт на весь сеанс — знания накапливаются между эпизодами.
    """
    services: dict[str, dict]                # {svc_name: agent_view(svc_name)}
    mem_threshold: dict[str, int]            # {workload_class: working_mib}
    mem_threshold_svc: dict[str, int]        # {svc_name: working_mib} карвинг-исключения
    incumbent_config: dict[str, str]         # {svc_name: current applied config}
    bad_config: set[tuple[str, str]]         # {(svc_name, config_opt)}
    reverse_index: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        svcs = {n: agent_view(n) for n in svc_names}
        return cls(
            services=svcs,
            mem_threshold={},
            mem_threshold_svc={},
            incumbent_config={
                n: svcs[n]["current_config"]
                for n in svc_names
                if "current_config" in svcs[n]
            },
            bad_config=set(),
        )

    def mark_bad_config(self, svc_name: str, config_opt: str) -> None:
        self.bad_config.add((svc_name, config_opt))

    def is_bad_config(self, svc_name: str, config_opt: str) -> bool:
        return (svc_name, config_opt) in self.bad_config

    def _choose_mem(self, svc: str) -> int | None:
        """
        Возвращает лучший известный порог памяти (per-svc > class) или None.
        None означает: нет знаний, агент начинает с MEM_BASE.
        """
        if svc in self.mem_threshold_svc:
            return self.mem_threshold_svc[svc]
        wc = self.services[svc]["workload_class"]
        if wc in self.mem_threshold:
            return self.mem_threshold[wc]
        return None

    def _choose_config(self, svc: str) -> str | None:
        """
        Выбирает конфиг для world.apply (политика агента, не планировщик).
        Приоритет: наследство (incumbent) если не known-bad, иначе
        детерминированная непровальная альтернатива.
        """
        opts = self.services[svc].get("config_options")
        if not opts:
            return "cfg_default"
        inc = self.incumbent_config.get(svc)
        if inc is not None and not self.is_bad_config(svc, inc):
            return inc
        for o in sorted(opts):
            if not self.is_bad_config(svc, o):
                return o
        return None


# ---------------------------------------------------------------------------
# Planner (config-feasibility check via PDDL)
# ---------------------------------------------------------------------------

def plan(bios: BiosState, goal_service: str) -> list[str] | None:
    """
    Проверяет config-достижимость через FD.
    Возвращает шаги плана или None (нет валидного конфига).
    Память — за агентом (не проверяется здесь).
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

    if result.status in (
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        PlanGenerationResultStatus.SOLVED_SATISFICING,
    ):
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== BIOS self-test (M3: doubling + thresholds) ===\n")
    errors = []

    # 1. Без config: план есть
    bios1 = BiosState.initial(["svc_a"])
    steps1 = plan(bios1, "svc_a")
    print(f"Test 1 (svc_a, no config): {steps1}")
    if steps1 is None or len(steps1) != 3:
        errors.append(f"Test1: expected 3-step plan, got {steps1}")
    else:
        print("  → OK ✓")

    # 2. С config (naive): план есть
    bios2 = BiosState.initial(["svc_e"])
    steps2 = plan(bios2, "svc_e")
    print(f"\nTest 2 (svc_e naive): {steps2}")
    if steps2 is None or len(steps2) != 3:
        errors.append(f"Test2: expected 3-step plan, got {steps2}")
    else:
        print("  → OK ✓")

    # 3. Все конфиги плохи → plan=None
    bios3 = BiosState.initial(["svc_e"])
    bios3.mark_bad_config("svc_e", "bad")
    bios3.mark_bad_config("svc_e", "good")
    steps3 = plan(bios3, "svc_e")
    print(f"\nTest 3 (all configs bad): {steps3}")
    if steps3 is not None:
        errors.append(f"Test3: expected None, got {steps3}")
    else:
        print("  → OK ✓")

    # 4. _choose_mem: no knowledge → None
    bios4 = BiosState.initial(["svc_e"])
    assert bios4._choose_mem("svc_e") is None, "_choose_mem should be None fresh"
    bios4.mem_threshold["standard"] = 512
    assert bios4._choose_mem("svc_e") == 512, "_choose_mem class threshold"
    bios4.mem_threshold_svc["svc_e"] = 1024
    assert bios4._choose_mem("svc_e") == 1024, "_choose_mem svc exception"
    print("\nTest 4 (_choose_mem): OK ✓")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All tests passed. BIOS M3: OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
