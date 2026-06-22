"""
BIOS-state: читаемая/редактируемая модель мира.

Содержит:
  services   — наблюдаемые фичи сервисов (agent_view)
  buckets    — доступные бакеты памяти [MiB], отсортированы по возрастанию
  unsafe_mem — выученные правила: set{(workload_class, bucket_mib)}

compile_to_up(bios, goal_service) → unified_planning.model.Problem
  Преобразует BIOS-state в UP Problem для fast-downward-opt.

  Encoding:
  — safe_mem(wc, bucket): True = пара безопасна. Default False.
    Изначально True для всех. При обучении → False.
    (pyperplan не поддерживал Not в preconditions; FD поддерживает,
    но encoding safe_mem оставлен как есть — чище и совместимо.)
  — bucket_mib(bucket): статик-числа MiB. Цена set_mem(b) = bucket_mib(b).
  — MinimizeActionCosts → планировщик выбирает наименьший безопасный бакет.

plan(bios, goal_service) → list[str] | None
  Статус SOLVED_OPTIMALLY (не SATISFICING).
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

get_environment().credits_stream = None  # убираем баннер UP


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
    buckets: list[int]               # MEM_BUCKETS, отсортированы по возрастанию
    unsafe_mem: set[tuple[str, int]] # {(workload_class, bucket_mib)} — выученные правила

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        """Чистый BIOS: сервисы известны, правил нет."""
        return cls(
            services={n: agent_view(n) for n in svc_names},
            buckets=sorted(MEM_BUCKETS),
            unsafe_mem=set(),
        )

    def mark_unsafe(self, workload_class: str, bucket_mib: int) -> None:
        """Записать выученное правило: wc OOM на bucket_mib."""
        self.unsafe_mem.add((workload_class, bucket_mib))

    def is_unsafe(self, workload_class: str, bucket_mib: int) -> bool:
        return (workload_class, bucket_mib) in self.unsafe_mem

    def safe_buckets_for(self, workload_class: str) -> list[int]:
        """Бакеты, ещё не помеченные как unsafe для данного класса (по возрастанию)."""
        return [b for b in self.buckets if not self.is_unsafe(workload_class, b)]


# ---------------------------------------------------------------------------
# Compile BIOS → unified_planning Problem
# ---------------------------------------------------------------------------

def compile_to_up(bios: BiosState, goal_service: str) -> Problem:
    """
    Создаёт UP Problem из текущего состояния BIOS.

    Типы:    Service, Bucket, WorkloadClass
    Флюенты:
      mem_set(bucket)            — установлен лимит памяти
      running(service)           — сервис запущен (цель)
      has_wc(service, wc)        — статик: наблюдаемый признак сервиса
      safe_mem(wc, bucket)       — пара безопасна (True по умолчанию для всех)
      bucket_mib(bucket)         — числовой размер бакета [MiB] (статик)

    Действия:
      set_mem(bucket)            — установить лимит памяти; cost = bucket_mib(b)
      deploy(service, bucket, wc)— деплоить при mem_set ∧ has_wc ∧ safe_mem; cost = 0

    Качество:
      MinimizeActionCosts → планировщик выбирает наименьший безопасный бакет.
    """
    if goal_service not in bios.services:
        raise ValueError(f"Unknown goal service: {goal_service!r}")

    # --- Types ---
    TService = UserType("Service")
    TBucket  = UserType("Bucket")
    TWC      = UserType("WorkloadClass")

    # --- Fluents ---
    mem_set    = Fluent("mem_set",    BoolType(), bucket=TBucket)
    running    = Fluent("running",    BoolType(), service=TService)
    has_wc     = Fluent("has_wc",    BoolType(), service=TService, wc=TWC)
    safe_mem   = Fluent("safe_mem",  BoolType(), wc=TWC, bucket=TBucket)
    bucket_mib = Fluent("bucket_mib", IntType(),  bucket=TBucket)

    # --- Action: set_mem(bucket) ---
    # cost = bucket_mib(b) → FD-opt минимизирует → выбирает наименьший безопасный бакет
    set_mem_a = InstantaneousAction("set_mem", bucket=TBucket)
    b = set_mem_a.parameter("bucket")
    set_mem_a.add_effect(mem_set(b), True)

    # --- Action: deploy(service, bucket, wc) ---
    deploy_a = InstantaneousAction("deploy", service=TService, bucket=TBucket, wc=TWC)
    s, bk, wc = deploy_a.parameters
    deploy_a.add_precondition(mem_set(bk))
    deploy_a.add_precondition(has_wc(s, wc))
    deploy_a.add_precondition(safe_mem(wc, bk))
    deploy_a.add_effect(running(s), True)

    # --- Problem ---
    problem = Problem("devops")
    problem.add_fluent(mem_set,    default_initial_value=False)
    problem.add_fluent(running,    default_initial_value=False)
    problem.add_fluent(has_wc,     default_initial_value=False)
    problem.add_fluent(safe_mem,   default_initial_value=False)
    problem.add_fluent(bucket_mib, default_initial_value=0)
    problem.add_action(set_mem_a)
    problem.add_action(deploy_a)

    # --- Objects (детерминированный порядок для воспроизводимости плана) ---
    bucket_objs: dict[int, Object] = {}
    for mib in sorted(bios.buckets):             # по возрастанию
        obj = Object(f"b{mib}", TBucket)
        problem.add_object(obj)
        bucket_objs[mib] = obj

    service_objs: dict[str, Object] = {}
    for name in sorted(bios.services):           # лексикографически
        obj = Object(name, TService)
        problem.add_object(obj)
        service_objs[name] = obj

    wc_objs: dict[str, Object] = {}
    for name in sorted(bios.services):
        wc = bios.services[name]["workload_class"]
        if wc not in wc_objs:
            obj = Object(wc, TWC)
            problem.add_object(obj)
            wc_objs[wc] = obj

    # --- Initial state: числа бакетов ---
    for mib, obj in bucket_objs.items():
        problem.set_initial_value(bucket_mib(obj), mib)

    # --- Initial state: статик-признаки сервисов ---
    for name, svc in sorted(bios.services.items()):
        wc = svc["workload_class"]
        problem.set_initial_value(has_wc(service_objs[name], wc_objs[wc]), True)

    # --- Initial state: safe_mem ---
    # True для каждой (wc, bucket) пары, кроме выученных unsafe.
    # Сортируем для детерминизма (bios.unsafe_mem — set, порядок не гарантирован).
    unsafe_set = bios.unsafe_mem  # {(wc_str, mib)}
    for wc_str in sorted(wc_objs):
        for mib in sorted(bucket_objs):
            is_safe = (wc_str, mib) not in unsafe_set
            problem.set_initial_value(safe_mem(wc_objs[wc_str], bucket_objs[mib]), is_safe)

    # --- Quality metric: минимизировать суммарный cost ---
    # cost(set_mem(b)) = bucket_mib(b); cost(deploy(...)) = 0
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
    Возвращает оптимальный план как список строк, например:
      ['set_mem(b512)', 'deploy(svc_a, b512, heavy)']
    или None если план не найден (все бакеты unsafe / задача неразрешима).

    Использует fast-downward-opt; требует статус SOLVED_OPTIMALLY.
    """
    problem = compile_to_up(bios, goal_service)
    with OneshotPlanner(name="fast-downward-opt") as planner:
        result = planner.solve(problem)

    if result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
        return [str(a) for a in result.plan.actions]
    return None


# ---------------------------------------------------------------------------
# Self-test (M1 → M2 переход)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== BIOS self-test (fast-downward-opt) ===\n")
    errors = []

    # 1. Наивный домен: правил нет → должны выбрать наименьший бакет
    bios = BiosState.initial(["svc_a"])
    steps = plan(bios, "svc_a")
    print("Test 1: naive domain (no rules) → expect smallest bucket (b64)")
    if steps is None:
        errors.append("Test1: no plan found")
    else:
        for i, s in enumerate(steps): print(f"  {i+1}. {s}")
        if not any("b64" in s for s in steps):
            errors.append(f"Test1: expected b64, got {steps}")
        else:
            print("  → b64 chosen ✓")

    print()

    # 2. unsafe heavy@[64,128] → должны выбрать b256 (минимальный безопасный)
    bios2 = BiosState.initial(["svc_a"])
    bios2.mark_unsafe("heavy", 64)
    bios2.mark_unsafe("heavy", 128)
    steps2 = plan(bios2, "svc_a")
    print("Test 2: unsafe heavy@[64,128] → expect b256")
    if steps2 is None:
        errors.append("Test2: no plan found")
    else:
        for i, s in enumerate(steps2): print(f"  {i+1}. {s}")
        if not any("b256" in s for s in steps2):
            errors.append(f"Test2: expected b256, got {steps2}")
        else:
            print("  → b256 chosen ✓")

    print()

    # 3. unsafe heavy@[64,128,256] → должны выбрать b512
    bios3 = BiosState.initial(["svc_a"])
    for bad in [64, 128, 256]:
        bios3.mark_unsafe("heavy", bad)
    steps3 = plan(bios3, "svc_a")
    print("Test 3: unsafe heavy@[64,128,256] → expect b512")
    if steps3 is None:
        errors.append("Test3: no plan found")
    else:
        for i, s in enumerate(steps3): print(f"  {i+1}. {s}")
        if not any("b512" in s for s in steps3):
            errors.append(f"Test3: expected b512, got {steps3}")
        else:
            print("  → b512 chosen ✓")

    print()

    # 4. Детерминизм: один и тот же problem → один и тот же план (в текущем процессе)
    bios4 = BiosState.initial(["svc_a"])
    bios4.mark_unsafe("heavy", 64)
    plans_same = [plan(bios4, "svc_a") for _ in range(3)]
    print("Test 4: determinism — 3 calls same bios → same plan")
    if len(set(tuple(p) for p in plans_same)) != 1:
        errors.append(f"Test4: non-deterministic plans: {plans_same}")
    else:
        print(f"  → all 3 identical: {plans_same[0]} ✓")

    print()

    if errors:
        print("FAILED:")
        for e in errors: print(f"  {e}")
        sys.exit(1)
    else:
        print("All tests passed. fast-downward-opt: OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
