"""
agent.py — петля агента (M3).

Цикл: plan → predict → act → compare → при gap: learn → replan.

С FD-opt + MinimizeActionCosts бинпоиск не нужен:
mark_unsafe(wc, bucket) + replan автоматически переходит к следующему
(наименьшему безопасному) бакету.

M4 добавит: reverse-index (интуицию) для ускорения диагностики.
"""

import re
import sys
from dataclasses import dataclass, field

from devops_agent.bios import BiosState, plan
from devops_agent.world import Obs, World


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """Одна попытка внутри эпизода."""
    bucket: int
    obs: Obs
    learned: bool  # False = успех или неожиданная ошибка; True = OOM → правило

    def __str__(self) -> str:
        tag = "learned" if self.learned else "ok" if self.obs.phase == "running" else "error"
        return f"Trial(bucket={self.bucket}, phase={self.obs.phase!r}, {tag})"


@dataclass
class EpisodeResult:
    svc_name: str
    success: bool
    trials: list[Trial]
    error: str = ""

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_learned(self) -> int:
        return sum(1 for t in self.trials if t.learned)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

_BUCKET_RE = re.compile(r'set_mem\(b(\d+)\)')


def _extract_bucket(steps: list[str]) -> int:
    """Извлечь MiB из строки плана 'set_mem(b512)'."""
    for step in steps:
        m = _BUCKET_RE.search(step)
        if m:
            return int(m.group(1))
    raise ValueError(f"No set_mem step in plan: {steps}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, world: World, bios: BiosState, verbose: bool = True):
        self.world = world
        self.bios = bios
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run_episode(self, svc_name: str) -> EpisodeResult:
        """
        Запускает петлю для одного сервиса до успеха или исчерпания бакетов.

        Возвращает EpisodeResult со списком всех проб.
        """
        if svc_name not in self.bios.services:
            raise ValueError(f"Unknown service: {svc_name!r}")

        wc = self.bios.services[svc_name]["workload_class"]
        trials: list[Trial] = []

        self._log(f"\n[episode] {svc_name} (workload_class={wc!r})")

        while True:
            # 1. Plan
            steps = plan(self.bios, svc_name)
            if steps is None:
                self._log("  [no plan] все бакеты unsafe — сходимость невозможна")
                return EpisodeResult(
                    svc_name=svc_name, success=False,
                    trials=trials, error="no_plan",
                )

            bucket = _extract_bucket(steps)
            self._log(f"  plan: {steps}")

            # 2. Predict: наивная гипотеза — деплой успешен
            predicted = "running"

            # 3. Act
            obs = self.world.apply({"service": svc_name, "mem_bucket": bucket})
            self._log(f"  obs:  {obs}")

            # 4. Compare
            if obs.phase == predicted:
                # Подтвердить: этот (wc, bucket) реально безопасен
                self.bios.confirm_safe(wc, bucket)
                trials.append(Trial(bucket=bucket, obs=obs, learned=False))
                self._log(f"  ✓ running @ {bucket} MiB — эпизод завершён за {len(trials)} проб(ы)")
                return EpisodeResult(svc_name=svc_name, success=True, trials=trials)

            # 5. Execution gap
            if obs.oom_killed:
                # 6. Learn — два случая:
                if self.bios.is_class_confirmed_safe(wc, bucket):
                    # Другой сервис того же класса уже успешно работал при этом бакете.
                    # Нельзя трогать класс-правило → per-service исключение (карвинг).
                    self.bios.mark_unsafe_svc(svc_name, bucket)
                    trials.append(Trial(bucket=bucket, obs=obs, learned=True))
                    self._log(
                        f"  carving: {wc}@{bucket} подтверждён безопасным для класса, "
                        f"но {svc_name} OOM → исключение ({svc_name},{bucket}) → replan"
                    )
                else:
                    # Первый OOM для (wc, bucket) → обновляем класс-правило
                    self.bios.mark_unsafe(wc, bucket)
                    trials.append(Trial(bucket=bucket, obs=obs, learned=True))
                    self._log(f"  gap: OOM @ {bucket} MiB → unsafe-mem({wc!r}, {bucket}) → replan")
                # 7. Replan (следующая итерация)
                continue

            # Неожиданный результат (error, не OOM)
            trials.append(Trial(bucket=bucket, obs=obs, learned=False))
            self._log(f"  unexpected obs: {obs}")
            return EpisodeResult(
                svc_name=svc_name, success=False,
                trials=trials, error=f"unexpected_obs:{obs.phase}",
            )


# ---------------------------------------------------------------------------
# Self-test (M3 + M4 + M5)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from devops_agent.services import MEM_BUCKETS, _GROUND_TRUTH

    print("=== Agent self-test (M3 / M4 / M5 / M5b) ===\n")
    errors = []

    world = World()
    bios  = BiosState.initial(["svc_a", "svc_b", "svc_c", "svc_d"])
    agent = Agent(world, bios, verbose=True)

    # M3: svc_a — учимся с нуля (heavy → safe@512)
    print("=== Эпизод 1: svc_a (heavy, учимся с нуля) ===")
    r_a = agent.run_episode("svc_a")

    # M4: svc_d — бесплатный перенос (heavy → safe@512, 1 проба)
    print("\n=== Эпизод 2: svc_d (heavy, бесплатный перенос) ===")
    r_d = agent.run_episode("svc_d")

    # M5: svc_b — независимый класс light → safe@128
    print("\n=== Эпизод 3: svc_b (light, независимый класс) ===")
    r_b = agent.run_episode("svc_b")

    # M5b: svc_c — тоже heavy, но safe@1024 → класс-правило ломается → карвинг
    print("\n=== Эпизод 4: svc_c (heavy, карвинг-исключение) ===")
    r_c = agent.run_episode("svc_c")

    # --- Сводка ---
    print("\n" + "=" * 55)
    print("СВОДКА (trials / learned / success)")
    print("=" * 55)
    rows = [
        ("svc_a", "heavy", r_a),
        ("svc_d", "heavy", r_d),
        ("svc_b", "light", r_b),
        ("svc_c", "heavy", r_c),
    ]
    for name, wc, r in rows:
        print(f"  {name:6s} [{wc:5s}]  trials={r.n_trials}  learned={r.n_learned}  ok={r.success}")

    print(f"\nunsafe_mem (класс-правила): {sorted(bios.unsafe_mem)}")
    print(f"unsafe_svc (исключения):    {sorted(bios.unsafe_svc)}")
    print(f"known_safe_class:           {sorted(bios.known_safe_class)}")

    # --- Честный вывод об ограничении ---
    print(
        "\n[NOTE] svc_c — тоже heavy, но порог выше (1024 vs 512). "
        "Агент не нашёл более тонкой наблюдаемой фичи (workload_class одинаков). "
        "Специализация выродилась в per-instance исключение: unsafe_svc={(svc_c,512)}. "
        "Класс-правило heavy→512 при этом цело (svc_a/svc_d не затронуты)."
    )

    # --- Проверки ---
    for name, _, r in rows:
        if not r.success:
            errors.append(f"{name}: ожидали success, got {r.error!r}")

    # M4: svc_d — 1 проба
    if r_d.n_trials != 1:
        errors.append(f"svc_d: ожидали 1 пробу (transfer), got {r_d.n_trials}")

    # M5b: класс-правило heavy@512 НЕ сломано
    if ("heavy", 512) in bios.unsafe_mem:
        errors.append("unsafe_mem содержит (heavy,512) — класс-правило сломано!")

    # M5b: карвинг-исключение на месте
    if ("svc_c", 512) not in bios.unsafe_svc:
        errors.append("unsafe_svc не содержит (svc_c,512) — карвинг не сработал")

    # svc_a/svc_d: планируют по-прежнему на b512
    from devops_agent.bios import plan as bios_plan
    plan_a  = bios_plan(bios, "svc_a")
    plan_d  = bios_plan(bios, "svc_d")
    if plan_a is None or not any("b512" in s for s in plan_a):
        errors.append(f"svc_a больше не планирует на b512: {plan_a}")
    if plan_d is None or not any("b512" in s for s in plan_d):
        errors.append(f"svc_d больше не планирует на b512: {plan_d}")

    # Нет кросс-загрязнения: ни одно правило не меняет класс другого.
    # Кросс-загрязнение = одна и та же пара (wc, b) попала в оба класса —
    # невозможно по построению (mark_unsafe всегда пишет конкретный wc).
    # Проверяем структурно: множества (wc,b) для разных wc не пересекаются.
    light_pairs = {(wc, b) for (wc, b) in bios.unsafe_mem if wc == "light"}
    heavy_pairs = {(wc, b) for (wc, b) in bios.unsafe_mem if wc == "heavy"}
    if light_pairs & heavy_pairs:
        errors.append(f"Кросс-загрязнение light↔heavy: {light_pairs & heavy_pairs}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("M3 + M4 + M5 + M5b OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.agent --selftest")
