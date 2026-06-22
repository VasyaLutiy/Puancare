"""
BIOS-state: читаемая/редактируемая модель мира.

Содержит:
  services   — наблюдаемые фичи сервисов (agent_view)
  buckets    — доступные бакеты памяти [MiB]
  unsafe_mem — выученные правила: set{(workload_class, bucket_mib)}

compile_to_up(bios, goal_service) → unified_planning.model.Problem
  Преобразует BIOS-state в UP Problem для pyperplan.

  Encoding (pyperplan поддерживает только позитивные preconditions):
  — safe_mem(wc, bucket): True = пара считается безопасной.
    Изначально True для всех. При обучении конкретная пара → False.
  — deploy precondition: mem_set(b) ∧ has_wc(s,wc) ∧ safe_mem(wc,b)
    Без Not() — pyperplan принимает только AND/FLUENT в preconditions.
"""

import sys
from dataclasses import dataclass, field

import unified_planning as up
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    Not,
    Object,
    OneshotPlanner,
    Problem,
    UserType,
)

from devops_agent.services import MEM_BUCKETS, agent_view, all_service_names

up.shortcuts.get_environment().credits_stream = None  # убираем баннер UP


# ---------------------------------------------------------------------------
# BIOS State
# ---------------------------------------------------------------------------

@dataclass
class BiosState:
    """
    Читаемое/редактируемое состояние агента.
    Агент меняет unsafe_mem → compile_to_up → plan → replan.
    """
    services: dict[str, dict]        # {svc_name: agent_view(svc_name)}
    buckets: list[int]               # MEM_BUCKETS
    unsafe_mem: set[tuple[str, int]] # {(workload_class, bucket_mib)} — выученные правила

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        """Чистый BIOS: сервисы известны, правил нет."""
        return cls(
            services={n: agent_view(n) for n in svc_names},
            buckets=list(MEM_BUCKETS),
            unsafe_mem=set(),
        )

    def mark_unsafe(self, workload_class: str, bucket_mib: int) -> None:
        """Записать выученное правило: wc OOM на bucket_mib."""
        self.unsafe_mem.add((workload_class, bucket_mib))

    def is_unsafe(self, workload_class: str, bucket_mib: int) -> bool:
        return (workload_class, bucket_mib) in self.unsafe_mem

    def safe_buckets_for(self, workload_class: str) -> list[int]:
        """Бакеты, ещё не помеченные как unsafe для данного класса."""
        return [b for b in self.buckets if not self.is_unsafe(workload_class, b)]


# ---------------------------------------------------------------------------
# Compile BIOS → unified_planning Problem
# ---------------------------------------------------------------------------

def compile_to_up(bios: BiosState, goal_service: str) -> Problem:
    """
    Создаёт UP Problem из текущего состояния BIOS.

    Типы:    Service, Bucket, WorkloadClass
    Флюенты:
      mem_set(bucket)           — установлен лимит памяти
      running(service)          — сервис запущен (цель)
      has_wc(service, wc)       — статик: наблюдаемый признак сервиса
      safe_mem(wc, bucket)      — пара (wc, bucket) считается безопасной

    Действия:
      set_mem(bucket)            — установить лимит памяти
      deploy(service, bucket, wc)— деплоить при mem_set ∧ has_wc ∧ safe_mem
    """
    if goal_service not in bios.services:
        raise ValueError(f"Unknown goal service: {goal_service!r}")

    # --- Types ---
    TService = UserType("Service")
    TBucket  = UserType("Bucket")
    TWC      = UserType("WorkloadClass")

    # --- Fluents ---
    mem_set  = Fluent("mem_set",  BoolType(), bucket=TBucket)
    running  = Fluent("running",  BoolType(), service=TService)
    has_wc   = Fluent("has_wc",  BoolType(), service=TService, wc=TWC)
    safe_mem = Fluent("safe_mem", BoolType(), wc=TWC, bucket=TBucket)

    # --- Action: set_mem(bucket) ---
    set_mem_a = InstantaneousAction("set_mem", bucket=TBucket)
    b = set_mem_a.parameter("bucket")
    set_mem_a.add_effect(mem_set(b), True)

    # --- Action: deploy(service, bucket, wc) ---
    # pyperplan принимает только позитивные preconditions (AND/FLUENT).
    # Поэтому unsafe_mem кодируем как safe_mem = True (безопасно).
    deploy_a = InstantaneousAction("deploy", service=TService, bucket=TBucket, wc=TWC)
    s, bk, wc = deploy_a.parameters
    deploy_a.add_precondition(mem_set(bk))
    deploy_a.add_precondition(has_wc(s, wc))
    deploy_a.add_precondition(safe_mem(wc, bk))
    deploy_a.add_effect(running(s), True)

    # --- Problem ---
    problem = Problem("devops")
    problem.add_fluent(mem_set,  default_initial_value=False)
    problem.add_fluent(running,  default_initial_value=False)
    problem.add_fluent(has_wc,   default_initial_value=False)
    problem.add_fluent(safe_mem, default_initial_value=False)
    problem.add_action(set_mem_a)
    problem.add_action(deploy_a)

    # --- Objects ---
    bucket_objs: dict[int, Object] = {}
    for mib in bios.buckets:
        obj = Object(f"b{mib}", TBucket)
        problem.add_object(obj)
        bucket_objs[mib] = obj

    service_objs: dict[str, Object] = {}
    for name in bios.services:
        obj = Object(name, TService)
        problem.add_object(obj)
        service_objs[name] = obj

    wc_objs: dict[str, Object] = {}
    for svc in bios.services.values():
        wc = svc["workload_class"]
        if wc not in wc_objs:
            obj = Object(wc, TWC)
            problem.add_object(obj)
            wc_objs[wc] = obj

    # --- Initial state: static facts ---
    for name, svc in bios.services.items():
        wc = svc["workload_class"]
        problem.set_initial_value(has_wc(service_objs[name], wc_objs[wc]), True)

    # --- Initial state: safe_mem ---
    # True для каждой (wc, bucket) пары, кроме выученных unsafe.
    for wc_str, wc_obj in wc_objs.items():
        for mib, b_obj in bucket_objs.items():
            is_safe = not bios.is_unsafe(wc_str, mib)
            problem.set_initial_value(safe_mem(wc_obj, b_obj), is_safe)

    # --- Goal ---
    problem.add_goal(running(service_objs[goal_service]))

    return problem


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan(bios: BiosState, goal_service: str) -> list[str] | None:
    """
    Возвращает план как список строк ['set_mem(b512)', 'deploy(svc_a, b512, heavy)'],
    или None если план не найден.
    """
    from unified_planning.engines import PlanGenerationResultStatus

    problem = compile_to_up(bios, goal_service)
    with OneshotPlanner(name="pyperplan") as planner:
        result = planner.solve(problem)

    if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Self-test (M1)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from unified_planning.engines import PlanGenerationResultStatus

    print("=== BIOS self-test (M1) ===\n")
    errors = []

    # Наивный домен: правил нет, цель — svc_a running
    bios = BiosState.initial(["svc_a"])
    steps = plan(bios, "svc_a")

    print(f"Plan for svc_a (naive domain, no rules):")
    if steps is None:
        print("  [FAIL] No plan found")
        errors.append("Naive domain: planner returned None")
    else:
        for i, step in enumerate(steps):
            print(f"  {i+1}. {step}")

        # Проверяем структуру плана
        if len(steps) != 2:
            errors.append(f"Expected 2 steps, got {len(steps)}: {steps}")
        if not any("set_mem" in s for s in steps):
            errors.append(f"No set_mem step in plan: {steps}")
        if not any("deploy" in s and "svc_a" in s for s in steps):
            errors.append(f"No deploy(svc_a) step in plan: {steps}")

    print()

    # Тест с одним unsafe правилом — убеждаемся что планировщик обходит его
    bios2 = BiosState.initial(["svc_a"])
    # Все бакеты кроме 512 и 1024 помечаем unsafe для heavy
    for bad_bucket in [64, 128, 256]:
        bios2.mark_unsafe("heavy", bad_bucket)

    steps2 = plan(bios2, "svc_a")
    print(f"Plan for svc_a (unsafe: heavy@[64,128,256]):")
    if steps2 is None:
        print("  [FAIL] No plan found")
        errors.append("Restricted domain: planner returned None")
    else:
        for i, step in enumerate(steps2):
            print(f"  {i+1}. {step}")

        # deploy должен использовать b512 или b1024
        safe_used = any("b512" in s or "b1024" in s for s in steps2)
        if not safe_used:
            errors.append(f"Planner chose unsafe bucket: {steps2}")
        else:
            print("  → safe bucket chosen ✓")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All checks passed. BIOS + pyperplan: OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
