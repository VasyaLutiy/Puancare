"""
BIOS-state: читаемая/редактируемая модель мира.

Поля:
  services         — наблюдаемые фичи сервисов (agent_view)
  buckets          — доступные бакеты памяти [MiB], по возрастанию
  unsafe_mem       — класс-правила: {(workload_class, bucket_mib)}
  unsafe_svc       — per-service исключения: {(svc_name, bucket_mib)}
  known_safe_class — (wc, bucket) подтверждены реальным running
  bad_config       — {(svc_name, config_opt)} плохих конфигов (M6)
  reverse_index    — симптом → [причина, ...] (M4 заглушка, M6 оживает)

compile_to_up(bios, goal_service) → unified_planning.model.Problem

  Домен:
  — safe_deploy(svc, bucket): не в unsafe_mem(wc) и не в unsafe_svc(svc)
  — config dimension активна ТОЛЬКО если хотя бы один сервис имеет config_options.
    config_ok(svc, opt): True если (svc, opt) ∉ bad_config (сервисы без config_options — всегда True)
  — cost(set_mem(b)) = b [MiB] → FD-opt выбирает наименьший безопасный бакет
  — cost(set_config(opt)) = 0 → без влияния на минимизацию памяти

plan(bios, goal_service) → list[str] | None
  Статус SOLVED_OPTIMALLY.
"""

import sys
from dataclasses import dataclass, field

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    Int,
    IntType,
    MinimizeActionCosts,
    Object,
    OneshotPlanner,
    Problem,
    UserType,
    get_environment,
)

from devops_agent.services import MEM_BUCKETS, agent_view, all_service_names

get_environment().credits_stream = None


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
    bad_config: set[tuple[str, str]]                 # {(svc_name, config_opt)} — M6
    reverse_index: dict[str, list[str]] = field(default_factory=dict)  # M6 интуиция

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        return cls(
            services={n: agent_view(n) for n in svc_names},
            buckets=sorted(MEM_BUCKETS),
            unsafe_mem=set(),
            unsafe_svc=set(),
            known_safe_class=set(),
            bad_config=set(),
        )

    # --- Класс-правила ---
    def mark_unsafe(self, workload_class: str, bucket_mib: int) -> None:
        self.unsafe_mem.add((workload_class, bucket_mib))

    # --- Per-service исключения (карвинг M5b) ---
    def mark_unsafe_svc(self, svc_name: str, bucket_mib: int) -> None:
        self.unsafe_svc.add((svc_name, bucket_mib))

    # --- Config (M6) ---
    def mark_bad_config(self, svc_name: str, config_opt: str) -> None:
        self.bad_config.add((svc_name, config_opt))

    # --- Подтверждение безопасности ---
    def confirm_safe(self, workload_class: str, bucket_mib: int) -> None:
        self.known_safe_class.add((workload_class, bucket_mib))

    # --- Запросы ---
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


# ---------------------------------------------------------------------------
# Compile BIOS → unified_planning Problem
# ---------------------------------------------------------------------------

def compile_to_up(bios: BiosState, goal_service: str) -> Problem:
    """
    Создаёт UP Problem.

    Config dimension включается только если хотя бы один сервис имеет config_options.
    Это сохраняет обратную совместимость с M3-M5b (plan без set_config).
    """
    if goal_service not in bios.services:
        raise ValueError(f"Unknown goal service: {goal_service!r}")

    TService = UserType("Service")
    TBucket  = UserType("Bucket")

    mem_set     = Fluent("mem_set",     BoolType(), bucket=TBucket)
    running     = Fluent("running",     BoolType(), service=TService)
    safe_deploy = Fluent("safe_deploy", BoolType(), service=TService, bucket=TBucket)
    bucket_mib  = Fluent("bucket_mib",  IntType(),  bucket=TBucket)

    # --- Config dimension ---
    config_opts: list[str] = sorted({
        opt
        for svc in bios.services.values()
        for opt in svc.get("config_options", [])
    })
    has_config = bool(config_opts)

    if has_config:
        TConfigOpt  = UserType("ConfigOpt")
        config_set  = Fluent("config_set",  BoolType(), opt=TConfigOpt)
        config_ok_f = Fluent("config_ok",   BoolType(), service=TService, opt=TConfigOpt)

        set_config_a = InstantaneousAction("set_config", opt=TConfigOpt)
        opt_p = set_config_a.parameter("opt")
        set_config_a.add_effect(config_set(opt_p), True)

        # deploy(service, bucket, config_opt)
        deploy_a = InstantaneousAction("deploy", service=TService, bucket=TBucket, opt=TConfigOpt)
        s, bk, opt_dp = deploy_a.parameters
        deploy_a.add_precondition(mem_set(bk))
        deploy_a.add_precondition(safe_deploy(s, bk))
        deploy_a.add_precondition(config_set(opt_dp))
        deploy_a.add_precondition(config_ok_f(s, opt_dp))
        deploy_a.add_effect(running(s), True)
    else:
        # Без config dimension (M3-M5b совместимость)
        deploy_a = InstantaneousAction("deploy", service=TService, bucket=TBucket)
        s, bk = deploy_a.parameters
        deploy_a.add_precondition(mem_set(bk))
        deploy_a.add_precondition(safe_deploy(s, bk))
        deploy_a.add_effect(running(s), True)

    # set_mem(bucket): cost = bucket_mib(b)
    set_mem_a = InstantaneousAction("set_mem", bucket=TBucket)
    b = set_mem_a.parameter("bucket")
    set_mem_a.add_effect(mem_set(b), True)

    # --- Problem ---
    problem = Problem("devops")
    problem.add_fluent(mem_set,     default_initial_value=False)
    problem.add_fluent(running,     default_initial_value=False)
    problem.add_fluent(safe_deploy, default_initial_value=False)
    problem.add_fluent(bucket_mib,  default_initial_value=0)
    problem.add_action(set_mem_a)
    problem.add_action(deploy_a)
    if has_config:
        problem.add_fluent(config_set,  default_initial_value=False)
        problem.add_fluent(config_ok_f, default_initial_value=False)
        problem.add_action(set_config_a)

    # --- Объекты (детерминированный порядок) ---
    bucket_objs: dict[int, Object] = {}
    for mib in sorted(bios.buckets):
        obj = Object(f"b{mib}", TBucket)
        problem.add_object(obj)
        bucket_objs[mib] = obj

    service_objs: dict[str, Object] = {}
    for name in sorted(bios.services):
        obj = Object(name, TService)
        problem.add_object(obj)
        service_objs[name] = obj

    config_objs: dict[str, Object] = {}
    if has_config:
        for opt in config_opts:
            obj = Object(opt, TConfigOpt)
            problem.add_object(obj)
            config_objs[opt] = obj

    # --- Initial state: bucket_mib ---
    for mib, obj in bucket_objs.items():
        problem.set_initial_value(bucket_mib(obj), mib)

    # --- Initial state: safe_deploy(svc, bucket) ---
    for name in sorted(bios.services):
        for mib in sorted(bios.buckets):
            is_safe = not bios.is_unsafe_for_deploy(name, mib)
            problem.set_initial_value(safe_deploy(service_objs[name], bucket_objs[mib]), is_safe)

    # --- Initial state: config_ok(svc, opt) ---
    if has_config:
        for name in sorted(bios.services):
            svc_has_config = bool(bios.services[name].get("config_options"))
            for opt in config_opts:
                if svc_has_config:
                    is_ok = not bios.is_bad_config(name, opt)
                else:
                    is_ok = True  # сервисы без config_options принимают любой config
                problem.set_initial_value(config_ok_f(service_objs[name], config_objs[opt]), is_ok)

    # --- Quality metric ---
    if has_config:
        problem.add_quality_metric(
            MinimizeActionCosts(
                {set_mem_a: bucket_mib(b), set_config_a: Int(0)},
                default=Int(0),
            )
        )
    else:
        problem.add_quality_metric(
            MinimizeActionCosts({set_mem_a: bucket_mib(b)}, default=Int(0))
        )

    # --- Goal ---
    problem.add_goal(running(service_objs[goal_service]))
    return problem


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan(bios: BiosState, goal_service: str) -> list[str] | None:
    """
    Возвращает оптимальный план или None.
    С config: ['set_config(good)', 'set_mem(b512)', 'deploy(svc_e, b512, good)']
    Без config: ['set_mem(b512)', 'deploy(svc_a, b512)']
    """
    problem = compile_to_up(bios, goal_service)
    with OneshotPlanner(name="fast-downward-opt") as planner:
        result = planner.solve(problem)
    if result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== BIOS self-test (M6: config dimension) ===\n")
    errors = []

    # 1. Без config: план = [set_mem, deploy] как в M5b
    bios1 = BiosState.initial(["svc_a"])
    steps1 = plan(bios1, "svc_a")
    print(f"Test 1 (no config): {steps1}")
    if steps1 is None or len(steps1) != 2 or not any("b64" in s for s in steps1):
        errors.append(f"Test1: expected 2-step plan with b64, got {steps1}")
    else:
        print("  → OK ✓")

    # 2. С config: план = [set_config, set_mem, deploy]
    bios2 = BiosState.initial(["svc_e"])
    steps2 = plan(bios2, "svc_e")
    print(f"\nTest 2 (svc_e naive): {steps2}")
    if steps2 is None or len(steps2) != 3:
        errors.append(f"Test2: expected 3-step plan, got {steps2}")
    elif not any("set_config" in s for s in steps2):
        errors.append(f"Test2: no set_config in plan: {steps2}")
    else:
        print("  → OK ✓")

    # 3. bad_config=(svc_e,bad) → план должен использовать good
    bios3 = BiosState.initial(["svc_e"])
    for bad in [64, 128, 256]:
        bios3.mark_unsafe("standard", bad)
    bios3.mark_bad_config("svc_e", "bad")
    steps3 = plan(bios3, "svc_e")
    print(f"\nTest 3 (bad_config={{svc_e,bad}}): {steps3}")
    if steps3 is None or not any("good" in s for s in steps3):
        errors.append(f"Test3: expected 'good' config, got {steps3}")
    elif not any("b512" in s for s in steps3):
        errors.append(f"Test3: expected b512, got {steps3}")
    else:
        print("  → OK ✓")

    # 4. Смешанный bios: svc_a (no config) + svc_e (with config)
    bios4 = BiosState.initial(["svc_a", "svc_e"])
    steps4a = plan(bios4, "svc_a")
    steps4e = plan(bios4, "svc_e")
    print(f"\nTest 4a (svc_a mixed bios): {steps4a}")
    print(f"Test 4e (svc_e mixed bios): {steps4e}")
    if steps4a is None or not any("b64" in s for s in steps4a):
        errors.append(f"Test4a: expected b64 for svc_a, got {steps4a}")
    if steps4e is None or not any("set_config" in s for s in steps4e):
        errors.append(f"Test4e: expected set_config for svc_e, got {steps4e}")
    if not errors:
        print("  → OK ✓")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All tests passed. BIOS config dimension: OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
