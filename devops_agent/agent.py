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
                trials.append(Trial(bucket=bucket, obs=obs, learned=False))
                self._log(f"  ✓ running @ {bucket} MiB — эпизод завершён за {len(trials)} проб(ы)")
                return EpisodeResult(svc_name=svc_name, success=True, trials=trials)

            # 5. Execution gap
            if obs.oom_killed:
                # 6. Learn: пометить бакет как unsafe для данного workload_class
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
# Self-test (M3: рождение правила)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from devops_agent.services import _GROUND_TRUTH, all_service_names

    print("=== Agent self-test (M3: рождение правила) ===\n")
    errors = []

    world = World()
    bios  = BiosState.initial(["svc_a"])
    agent = Agent(world, bios, verbose=True)

    result = agent.run_episode("svc_a")

    print(f"\n--- итог ---")
    print(f"success:  {result.success}")
    print(f"n_trials: {result.n_trials}")
    print(f"n_learned:{result.n_learned}")
    print(f"unsafe_mem после эпизода: {sorted(bios.unsafe_mem)}")

    # Ожидания из ground truth
    min_safe = _GROUND_TRUTH["svc_a"]["min_safe_bucket"]
    from devops_agent.services import MEM_BUCKETS
    unsafe_expected = {("heavy", b) for b in MEM_BUCKETS if b < min_safe}

    if not result.success:
        errors.append(f"Ожидали success=True, got error={result.error!r}")
    if result.n_trials != result.n_learned + 1:
        errors.append(f"Ожидали n_trials = n_learned + 1, got {result.n_trials}/{result.n_learned}")
    if bios.unsafe_mem != unsafe_expected:
        errors.append(f"unsafe_mem: ожидали {sorted(unsafe_expected)}, got {sorted(bios.unsafe_mem)}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("Рождение правила подтверждено. M3 OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.agent --selftest")
