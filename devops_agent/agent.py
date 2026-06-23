"""
agent.py — петля агента (M3–M6).

Цикл: plan → predict → act → compare → diagnose → learn → replan.

M6 добавляет:
  - второй симптом: "unhealthy" (плохой config)
  - reverse_index: symptom → [cause, ...] — строится опытом, не засевается
  - oracle: вызывается ТОЛЬКО когда reverse_index пуст (cost-gated)
  - _bump_reverse: каждый успешный фикс поднимает причину на первое место

Default causes (без oracle, для очевидных симптомов):
  oom_killed → memory (очевидно, не требует LLM)
  unhealthy  → config (используется как fallback если oracle=None)
"""

import sys
from dataclasses import dataclass, field

from devops_agent.bios import BiosState, plan
from devops_agent.model.ontology import WorldModel as _WorldModel
from devops_agent.world import Obs, World

_wm = _WorldModel.load()


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """Одна попытка внутри эпизода."""
    bucket: int
    config: str
    obs: Obs
    learned: bool
    oracle_used: bool = False

    def __str__(self) -> str:
        tag = "learned" if self.learned else ("ok" if self.obs.phase == "running" else "error")
        o = " [oracle]" if self.oracle_used else ""
        return f"Trial(bucket={self.bucket}, config={self.config!r}, phase={self.obs.phase!r}, {tag}{o})"


@dataclass
class EpisodeResult:
    svc_name: str
    success: bool
    trials: list[Trial]
    oracle_calls: int = 0
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



def _bump_reverse(reverse_index: dict[str, list[str]], symptom: str, cause: str) -> None:
    """Поднять cause на первое место в reverse_index[symptom]."""
    causes = reverse_index.setdefault(symptom, [])
    if cause in causes:
        causes.remove(cause)
    causes.insert(0, cause)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, world: World, bios: BiosState, oracle=None, verbose: bool = True):
        """
        oracle: devops_agent.oracle.Oracle или None.
          None → используются fallback-причины из онтологии без LLM-вызовов.
        """
        self.world = world
        self.bios = bios
        self.oracle = oracle
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run_episode(self, svc_name: str) -> EpisodeResult:
        """
        Запускает петлю для одного сервиса до успеха или исчерпания бакетов.
        """
        if svc_name not in self.bios.services:
            raise ValueError(f"Unknown service: {svc_name!r}")

        wc = self.bios.services[svc_name]["workload_class"]
        trials: list[Trial] = []
        episode_oracle_calls = 0
        last_symptom: str | None = None
        last_cause: str | None = None

        self._log(f"\n[episode] {svc_name} (workload_class={wc!r})")

        while True:
            # 1. Plan
            steps = plan(self.bios, svc_name)
            if steps is None:
                self._log("  [no plan] все бакеты/конфиги заблокированы")
                return EpisodeResult(
                    svc_name=svc_name, success=False,
                    trials=trials, oracle_calls=episode_oracle_calls,
                    error="no_plan",
                )

            bucket = self.bios.safe_buckets_for(svc_name)[0]
            config = self.bios._choose_config(svc_name)
            self._log(f"  plan: {steps}")

            # 2. Act
            obs = self.world.apply({"service": svc_name, "mem_bucket": bucket, "config": config})
            self._log(f"  obs:  {obs}")

            # 3. Compare: predict=running
            if obs.phase == "running":
                # Успех: подтвердить class-safe, зафиксировать incumbent, поднять причину
                self.bios.confirm_safe(wc, bucket)
                self.bios.incumbent_config[svc_name] = config
                if last_symptom and last_cause:
                    _bump_reverse(self.bios.reverse_index, last_symptom, last_cause)
                trials.append(Trial(bucket=bucket, config=config, obs=obs, learned=False))
                self._log(
                    f"  ✓ running @ {bucket} MiB config={config!r} "
                    f"— эпизод завершён за {len(trials)} проб(ы)"
                )
                return EpisodeResult(
                    svc_name=svc_name, success=True,
                    trials=trials, oracle_calls=episode_oracle_calls,
                )

            # 4. Execution gap — классифицируем симптом
            symptom = _wm.classify(obs)

            # 5. Diagnostic routing
            oracle_used = False
            suspects = self.bios.reverse_index.get(symptom, [])
            if suspects:
                cause = suspects[0]
            elif _wm.obvious_cause(symptom) is not None:
                # Очевидная причина — оракул не нужен даже если доступен
                cause = _wm.obvious_cause(symptom)
            elif self.oracle is not None:
                # Cost-gated: LLM только на плоском приоре неочевидного симптома
                ctx = {"service": svc_name, "workload_class": wc,
                       "mem_bucket": bucket, "config": config}
                causes = self.oracle.suggest_causes(symptom, ctx)
                cause = causes[0]
                self.bios.reverse_index[symptom] = [cause]
                episode_oracle_calls += 1
                oracle_used = True
                self._log(
                    f"  oracle: {symptom!r} → {causes} "
                    f"(total_calls={self.oracle.n_calls}, "
                    f"tokens={self.oracle.total_tokens})"
                )
            else:
                cause = _wm.fallback_cause(symptom) or "memory"

            last_symptom = symptom
            last_cause = cause
            trials.append(Trial(bucket=bucket, config=config, obs=obs,
                                learned=True, oracle_used=oracle_used))

            # 6. Apply fix (диспетчер по repair_kind из онтологии)
            rk = _wm.repair_kind(cause)
            if rk == "mem_threshold":
                if self.bios.is_class_confirmed_safe(wc, bucket):
                    self.bios.mark_unsafe_svc(svc_name, bucket)
                    self._log(
                        f"  carving: {wc}@{bucket} safe→exception ({svc_name},{bucket})"
                    )
                else:
                    self.bios.mark_unsafe(wc, bucket)
                    self._log(
                        f"  gap: OOM@{bucket}→unsafe-mem({wc},{bucket})"
                    )
            elif rk == "bad_config":
                self.bios.mark_bad_config(svc_name, config)
                self._log(
                    f"  gap: unhealthy@config={config!r}→bad_config({svc_name},{config!r})"
                )
            else:
                self._log(f"  unknown repair kind for cause {cause!r}, skipping fix")

            # 7. Unexpected (не OOM, не unhealthy)
            if obs.phase == "error":
                return EpisodeResult(
                    svc_name=svc_name, success=False,
                    trials=trials, oracle_calls=episode_oracle_calls,
                    error=f"unexpected_obs:{obs.phase}",
                )

            # 8. Replan
            continue


# ---------------------------------------------------------------------------
# Self-test (M3 / M4 / M5 / M5b / M6)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from devops_agent.services import MEM_BUCKETS, _GROUND_TRUTH

    print("=== Agent self-test (M3 / M4 / M5 / M5b / M6) ===\n")
    errors = []

    world = World()
    # M3-M5b: без oracle (дефолтные причины)
    bios_m5 = BiosState.initial(["svc_a", "svc_b", "svc_c", "svc_d"])
    agent_m5 = Agent(world, bios_m5, oracle=None, verbose=True)

    print("=== Эпизод 1: svc_a (heavy, учимся с нуля) ===")
    r_a = agent_m5.run_episode("svc_a")

    print("\n=== Эпизод 2: svc_d (heavy, бесплатный перенос) ===")
    r_d = agent_m5.run_episode("svc_d")

    print("\n=== Эпизод 3: svc_b (light, независимый класс) ===")
    r_b = agent_m5.run_episode("svc_b")

    print("\n=== Эпизод 4: svc_c (heavy, карвинг-исключение) ===")
    r_c = agent_m5.run_episode("svc_c")

    print(f"\n  reverse_index после M3-M5b: {bios_m5.reverse_index}")
    oom_index = bios_m5.reverse_index.get("oom_killed", [])
    if not oom_index or oom_index[0] != "memory":
        errors.append(f"reverse_index['oom_killed'] после M3-M5b: ожидали ['memory'], got {oom_index}")

    # --- M6: svc_e с оракулом ---
    try:
        from devops_agent.oracle import Oracle
        oracle = Oracle()
        oracle_available = True
    except Exception as e:
        print(f"\n  [SKIP M6] Oracle недоступен: {e}")
        oracle_available = False

    if not oracle_available:
        errors.append("M6 пропущен (Oracle недоступен) — skip считается ошибкой")

    if oracle_available:
        # Один BIOS на весь сеанс: переносим накопленные знания из M3-M5b
        bios_m6 = BiosState.initial(["svc_a", "svc_b", "svc_c", "svc_d", "svc_e"])
        # Переносим знания из bios_m5 (unsafe_mem, known_safe_class, reverse_index)
        bios_m6.unsafe_mem = set(bios_m5.unsafe_mem)
        bios_m6.unsafe_svc = set(bios_m5.unsafe_svc)
        bios_m6.known_safe_class = set(bios_m5.known_safe_class)
        bios_m6.reverse_index = dict(bios_m5.reverse_index)
        agent_m6 = Agent(world, bios_m6, oracle=oracle, verbose=True)

        print("\n=== Эпизод A: svc_e (первый раз, oracle ожидается 1 вызов) ===")
        r_ea = agent_m6.run_episode("svc_e")

        tokens_after_a = oracle.total_tokens
        calls_after_a = oracle.n_calls

        print("\n=== Эпизод B: svc_e (повторно, oracle ожидается 0 вызовов) ===")
        r_eb = agent_m6.run_episode("svc_e")

        # --- Сводка M6 ---
        print("\n" + "=" * 60)
        print("СВОДКА M6")
        print("=" * 60)
        print(f"  Эп. A svc_e: success={r_ea.success}, trials={r_ea.n_trials}, oracle_calls={r_ea.oracle_calls}")
        print(f"  Эп. B svc_e: success={r_eb.success}, trials={r_eb.n_trials}, oracle_calls={r_eb.oracle_calls}")
        print(f"  oracle.n_calls={oracle.n_calls}, oracle.total_tokens={oracle.total_tokens}")
        print(f"  reverse_index: {bios_m6.reverse_index}")
        print(f"  bad_config: {sorted(bios_m6.bad_config)}")
        print(
            "\n[NOTE] Закрытый словарь причин {'memory','config'} — ограничение v1. "
            "Новые типы отказов потребуют расширения словаря в v2."
        )

        # Проверки M6
        if not r_ea.success:
            errors.append(f"Эп. A: ожидали success, got error={r_ea.error!r}")
        if r_ea.oracle_calls != 1:
            errors.append(f"Эп. A: ожидали oracle_calls=1, got {r_ea.oracle_calls}")
        if not r_eb.success:
            errors.append(f"Эп. B: ожидали success, got error={r_eb.error!r}")
        if r_eb.oracle_calls != 0:
            errors.append(f"Эп. B: ожидали oracle_calls=0, got {r_eb.oracle_calls}")
        if oracle.total_tokens == 0:
            errors.append("oracle.total_tokens=0 после Эп.A — учёт токенов не работает")
        if oracle.total_tokens != tokens_after_a:
            errors.append(f"Токены выросли после Эп.B: {tokens_after_a} → {oracle.total_tokens}")

    # Проверки M3-M5b (регресс)
    print("\n" + "=" * 60)
    print("СВОДКА M3-M5b (регресс)")
    print("=" * 60)
    rows = [
        ("svc_a", "heavy",   r_a),
        ("svc_d", "heavy",   r_d),
        ("svc_b", "light",   r_b),
        ("svc_c", "heavy",   r_c),
    ]
    for name, wc, r in rows:
        print(f"  {name:6s} [{wc:5s}]  trials={r.n_trials}  learned={r.n_learned}  ok={r.success}")

    for name, _, r in rows:
        if not r.success:
            errors.append(f"{name}: ожидали success, got {r.error!r}")
    if r_d.n_trials != 1:
        errors.append(f"svc_d: ожидали 1 пробу (transfer), got {r_d.n_trials}")
    if ("heavy", 512) in bios_m5.unsafe_mem:
        errors.append("unsafe_mem содержит (heavy,512) — класс-правило сломано!")
    if ("svc_c", 512) not in bios_m5.unsafe_svc:
        errors.append("unsafe_svc не содержит (svc_c,512) — карвинг не сработал")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("M3 + M4 + M5 + M5b + M6 OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.agent --selftest")
