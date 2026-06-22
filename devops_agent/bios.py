"""
BIOS-state: читаемая/редактируемая модель мира.

Поля:
  services         — наблюдаемые фичи сервисов (agent_view)
  buckets          — доступные бакеты [MiB], отсортированы по возрастанию
  unsafe_mem       — класс-правила: {(workload_class, bucket_mib)}
  unsafe_svc       — per-service исключения: {(svc_name, bucket_mib)}
  known_safe_class — (wc, bucket) подтверждены безопасными реальным запуском
  reverse_index    — заглушка M6: symptom → [(wc, bucket), ...]

compile_to_up(bios, goal_service) → unified_planning.model.Problem

  Домен (M5b, упрощённый):
  — Убраны has_wc и wc-параметр из deploy: вся информация о классе
    и исключениях упакована в safe_deploy(service, bucket).
  — safe_deploy(svc, b) = True, если
      (wc_of_svc, b) ∉ unsafe_mem  AND  (svc, b) ∉ unsafe_svc
  — deploy(service, bucket): PRE: mem_set(b) ∧ safe_deploy(svc, b)
  — cost(set_mem(b)) = b [MiB] → FD-opt выбирает наименьший безопасный бакет.

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
    services: dict[str, dict]                       # {svc_name: agent_view(svc_name)}
    buckets: list[int]                              # MEM_BUCKETS по возрастанию
    unsafe_mem: set[tuple[str, int]]               # {(workload_class, bucket)} — класс-правила
    unsafe_svc: set[tuple[str, int]]               # {(svc_name, bucket)} — per-service исключения
    known_safe_class: set[tuple[str, int]]         # {(wc, bucket)} подтверждены реальным running
    reverse_index: dict[str, list[tuple[str, int]]] = field(default_factory=dict)  # заглушка M6

    @classmethod
    def initial(cls, svc_names: list[str]) -> "BiosState":
        """Чистый BIOS: сервисы известны, правил нет."""
        return cls(
            services={n: agent_view(n) for n in svc_names},
            buckets=sorted(MEM_BUCKETS),
            unsafe_mem=set(),
            unsafe_svc=set(),
            known_safe_class=set(),
        )

    # --- Класс-правила ---

    def mark_unsafe(self, workload_class: str, bucket_mib: int) -> None:
        """Пометить (wc, bucket) как небезопасный для всего класса."""
        self.unsafe_mem.add((workload_class, bucket_mib))

    # --- Per-service исключения (карвинг) ---

    def mark_unsafe_svc(self, svc_name: str, bucket_mib: int) -> None:
        """Пометить (svc, bucket) как небезопасный конкретно для этого сервиса."""
        self.unsafe_svc.add((svc_name, bucket_mib))

    # --- Подтверждение безопасности ---

    def confirm_safe(self, workload_class: str, bucket_mib: int) -> None:
        """Записать, что (wc, bucket) реально безопасен — видели running."""
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

    def safe_buckets_for(self, svc_name: str) -> list[int]:
        return [b for b in self.buckets if not self.is_unsafe_for_deploy(svc_name, b)]


# ---------------------------------------------------------------------------
# Compile BIOS → unified_planning Problem
# ---------------------------------------------------------------------------

def compile_to_up(bios: BiosState, goal_service: str) -> Problem:
    """
    Создаёт UP Problem из текущего состояния BIOS.

    Типы:    Service, Bucket
    Флюенты:
      mem_set(bucket)             — установлен лимит памяти
      running(service)            — сервис запущен (цель)
      safe_deploy(service,bucket) — деплой разрешён (класс + svc правила)
      bucket_mib(bucket)          — числовой размер бакета [MiB]

    Действия:
      set_mem(bucket)             — установить лимит; cost = bucket_mib(b)
      deploy(service,bucket)      — деплоить при mem_set ∧ safe_deploy
    """
    if goal_service not in bios.services:
        raise ValueError(f"Unknown goal service: {goal_service!r}")

    TService = UserType("Service")
    TBucket  = UserType("Bucket")

    mem_set      = Fluent("mem_set",      BoolType(), bucket=TBucket)
    running      = Fluent("running",      BoolType(), service=TService)
    safe_deploy  = Fluent("safe_deploy",  BoolType(), service=TService, bucket=TBucket)
    bucket_mib   = Fluent("bucket_mib",   IntType(),  bucket=TBucket)

    # set_mem(bucket): cost = bucket_mib(b)
    set_mem_a = InstantaneousAction("set_mem", bucket=TBucket)
    b = set_mem_a.parameter("bucket")
    set_mem_a.add_effect(mem_set(b), True)

    # deploy(service, bucket): PRE: mem_set ∧ safe_deploy
    deploy_a = InstantaneousAction("deploy", service=TService, bucket=TBucket)
    s, bk = deploy_a.parameters
    deploy_a.add_precondition(mem_set(bk))
    deploy_a.add_precondition(safe_deploy(s, bk))
    deploy_a.add_effect(running(s), True)

    problem = Problem("devops")
    problem.add_fluent(mem_set,     default_initial_value=False)
    problem.add_fluent(running,     default_initial_value=False)
    problem.add_fluent(safe_deploy, default_initial_value=False)
    problem.add_fluent(bucket_mib,  default_initial_value=0)
    problem.add_action(set_mem_a)
    problem.add_action(deploy_a)

    # Объекты (детерминированный порядок)
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

    # Начальное состояние: числа бакетов
    for mib, obj in bucket_objs.items():
        problem.set_initial_value(bucket_mib(obj), mib)

    # Начальное состояние: safe_deploy(svc, bucket)
    # Комбинирует класс-правила и per-service исключения в один флюент.
    for name in sorted(bios.services):
        for mib in sorted(bios.buckets):
            is_safe = not bios.is_unsafe_for_deploy(name, mib)
            problem.set_initial_value(safe_deploy(service_objs[name], bucket_objs[mib]), is_safe)

    # Качество: минимизировать суммарный cost
    problem.add_quality_metric(
        MinimizeActionCosts({set_mem_a: bucket_mib(b)}, default=Int(0))
    )

    # Цель
    problem.add_goal(running(service_objs[goal_service]))

    return problem


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan(bios: BiosState, goal_service: str) -> list[str] | None:
    """
    Возвращает оптимальный план ['set_mem(b512)', 'deploy(svc_a, b512)']
    или None если плана нет (все бакеты заблокированы).
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
    print("=== BIOS self-test (fast-downward-opt + safe_deploy) ===\n")
    errors = []

    # 1. Наивный домен
    bios = BiosState.initial(["svc_a"])
    steps = plan(bios, "svc_a")
    print("Test 1: naive → b64")
    if steps is None or not any("b64" in s for s in steps):
        errors.append(f"Test1: expected b64, got {steps}")
    else:
        print(f"  {steps} ✓")

    # 2. Класс-правило: unsafe heavy@[64,128] → b256
    bios2 = BiosState.initial(["svc_a"])
    bios2.mark_unsafe("heavy", 64)
    bios2.mark_unsafe("heavy", 128)
    steps2 = plan(bios2, "svc_a")
    print("Test 2: unsafe heavy@[64,128] → b256")
    if steps2 is None or not any("b256" in s for s in steps2):
        errors.append(f"Test2: expected b256, got {steps2}")
    else:
        print(f"  {steps2} ✓")

    # 3. Per-service исключение: svc_a safe@512, svc_c исключение@512 → b1024
    bios3 = BiosState.initial(["svc_a", "svc_c"])
    for bad in [64, 128, 256]:
        bios3.mark_unsafe("heavy", bad)
    bios3.mark_unsafe_svc("svc_c", 512)  # карвинг-исключение
    steps3 = plan(bios3, "svc_c")
    steps3a = plan(bios3, "svc_a")
    print("Test 3: svc_c исключение@512 → b1024; svc_a по-прежнему b512")
    ok3 = (steps3 and any("b1024" in s for s in steps3) and
           steps3a and any("b512" in s for s in steps3a))
    if not ok3:
        errors.append(f"Test3: svc_c={steps3}, svc_a={steps3a}")
    else:
        print(f"  svc_c: {steps3} ✓")
        print(f"  svc_a: {steps3a} ✓")

    print()
    if errors:
        print("FAILED:")
        for e in errors: print(f"  {e}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.bios --selftest")
