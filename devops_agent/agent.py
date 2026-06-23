"""
agent.py — петля агента (M3–M6+#7).

Цикл: _choose_mem → _choose_config → act → observe → learn → повторить.

Memory: удвоение от MEM_BASE до running (правило, не список).
  oom_killed → current_mem *= 2. Потолок MEM_CEILING → no_plan.
  Запись:
    fresh (нет class threshold) → mem_threshold[wc]
    carving (class threshold не сработал) → mem_threshold_svc[svc]
    transfer (class threshold сразу) → без изменений

Config: наследство через _choose_config → incumbent.
  unhealthy → mark_bad_config → выбрать следующий config, НЕ менять mem.

Oracle: вызывается только при плоском приоре (reverse_index пуст).
"""

import sys
from dataclasses import dataclass, field

from devops_agent.bios import BiosState, MEM_BASE, MEM_CEILING
from devops_agent.model.ontology import WorldModel as _WorldModel
from devops_agent.world import Obs, World

_wm = _WorldModel.load()


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """Одна попытка внутри эпизода."""
    bucket: int       # mem_mib в этой попытке
    config: str
    obs: Obs
    learned: bool
    oracle_used: bool = False

    def __str__(self) -> str:
        tag = "learned" if self.learned else ("ok" if self.obs.phase == "running" else "error")
        o = " [oracle]" if self.oracle_used else ""
        return f"Trial(mem={self.bucket}m, config={self.config!r}, phase={self.obs.phase!r}, {tag}{o})"


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
        Запускает петлю удвоения для одного сервиса до running или потолка.
        """
        if svc_name not in self.bios.services:
            raise ValueError(f"Unknown service: {svc_name!r}")

        wc = self.bios.services[svc_name]["workload_class"]
        trials: list[Trial] = []
        oracle_calls = 0
        last_symptom: str | None = None
        last_cause: str | None = None

        # --- Начальное состояние памяти ---
        has_class_threshold = wc in self.bios.mem_threshold
        has_svc_threshold = svc_name in self.bios.mem_threshold_svc
        current_mem = self.bios._choose_mem(svc_name) or MEM_BASE
        # carving: True если начали с class threshold и уже получили OOM,
        # или если есть svc-исключение (обновляем его при OOM)
        carving = has_svc_threshold

        self._log(f"\n[episode] {svc_name} (wc={wc!r}, mem_start={current_mem}m, "
                  f"carving={carving})")

        while True:
            # --- Потолок ---
            if current_mem > MEM_CEILING:
                self._log(f"  [ceiling] {current_mem}m > {MEM_CEILING}m → no_plan")
                return EpisodeResult(
                    svc_name=svc_name, success=False,
                    trials=trials, oracle_calls=oracle_calls,
                    error="no_plan",
                )

            # --- Config feasibility ---
            config = self.bios._choose_config(svc_name)
            if config is None:
                self._log("  [no config] все конфиги заблокированы → no_plan")
                return EpisodeResult(
                    svc_name=svc_name, success=False,
                    trials=trials, oracle_calls=oracle_calls,
                    error="no_plan",
                )

            self._log(f"  try: mem={current_mem}m config={config!r}")

            # --- Act ---
            obs = self.world.apply({
                "service": svc_name,
                "mem_bucket": current_mem,
                "config": config,
            })
            self._log(f"  obs:  {obs}")

            # --- Success ---
            if obs.phase == "running":
                # Запись порога памяти
                if carving:
                    self.bios.mem_threshold_svc[svc_name] = current_mem
                elif not has_class_threshold:
                    # Свежее обучение → запись class threshold
                    self.bios.mem_threshold[wc] = current_mem
                # else: transfer сработал → без изменений

                self.bios.incumbent_config[svc_name] = config
                if last_symptom and last_cause:
                    _bump_reverse(self.bios.reverse_index, last_symptom, last_cause)
                trials.append(Trial(bucket=current_mem, config=config, obs=obs, learned=False))
                self._log(
                    f"  ✓ running @ {current_mem}m config={config!r} "
                    f"— эпизод завершён за {len(trials)} проб(ы)"
                )
                return EpisodeResult(
                    svc_name=svc_name, success=True,
                    trials=trials, oracle_calls=oracle_calls,
                )

            # --- Execution gap: classify + diagnose ---
            symptom = _wm.classify(obs)
            oracle_used = False
            suspects = self.bios.reverse_index.get(symptom, [])
            if suspects:
                cause = suspects[0]
            elif _wm.obvious_cause(symptom) is not None:
                cause = _wm.obvious_cause(symptom)
            elif self.oracle is not None:
                ctx = {"service": svc_name, "workload_class": wc,
                       "mem_bucket": current_mem, "config": config}
                causes = self.oracle.suggest_causes(symptom, ctx)
                cause = causes[0]
                self.bios.reverse_index[symptom] = [cause]
                oracle_calls += 1
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
            trials.append(Trial(bucket=current_mem, config=config, obs=obs,
                                learned=True, oracle_used=oracle_used))

            # --- Apply fix ---
            rk = _wm.repair_kind(cause)
            if rk == "mem_threshold":
                # Память: удвоить
                if not carving and has_class_threshold:
                    # Первый OOM на class threshold → начинаем carving
                    carving = True
                current_mem *= 2
                self._log(f"  gap: OOM → mem*2={current_mem}m")
            elif rk == "bad_config":
                # Конфиг: заблокировать и повторить с тем же mem
                self.bios.mark_bad_config(svc_name, config)
                self._log(
                    f"  gap: unhealthy@config={config!r} → bad_config({svc_name},{config!r})"
                )
            else:
                self._log(f"  unknown repair kind for cause {cause!r}, skipping fix")
                # Неожиданное состояние — прерываем
                if obs.phase == "error":
                    return EpisodeResult(
                        svc_name=svc_name, success=False,
                        trials=trials, oracle_calls=oracle_calls,
                        error=f"unexpected_obs:{obs.phase}",
                    )


# ---------------------------------------------------------------------------
# Self-test (M3 / M4 / M5 / M5b / M6)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from devops_agent.services import _GROUND_TRUTH

    print("=== Agent self-test (M3 / M4 / M5 / M5b / M6) ===\n")
    errors = []

    world = World()
    bios_m5 = BiosState.initial(["svc_a", "svc_b", "svc_c", "svc_d", "svc_f"])
    agent_m5 = Agent(world, bios_m5, oracle=None, verbose=True)

    print("=== Эп 1: svc_a (heavy, учимся с нуля) ===")
    r_a = agent_m5.run_episode("svc_a")

    print("\n=== Эп 2: svc_d (heavy, бесплатный перенос) ===")
    r_d = agent_m5.run_episode("svc_d")

    print("\n=== Эп 3: svc_b (light, независимый класс) ===")
    r_b = agent_m5.run_episode("svc_b")

    print("\n=== Эп 4: svc_c (heavy, карвинг-исключение) ===")
    r_c = agent_m5.run_episode("svc_c")

    print("\n=== Эп 5: svc_f (xlarge, anti-hardcode: 2048m без правок кода) ===")
    r_f = agent_m5.run_episode("svc_f")

    print(f"\n  reverse_index после M3-M5b: {bios_m5.reverse_index}")
    print(f"  mem_threshold: {bios_m5.mem_threshold}")
    print(f"  mem_threshold_svc: {bios_m5.mem_threshold_svc}")

    oom_index = bios_m5.reverse_index.get("oom_killed", [])
    if not oom_index or oom_index[0] != "memory":
        errors.append(f"reverse_index['oom_killed']: ожидали ['memory'], got {oom_index}")

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
        bios_m6 = BiosState.initial(["svc_a", "svc_b", "svc_c", "svc_d", "svc_e"])
        # Переносим накопленные знания из M3-M5b
        bios_m6.mem_threshold = dict(bios_m5.mem_threshold)
        bios_m6.mem_threshold_svc = dict(bios_m5.mem_threshold_svc)
        bios_m6.reverse_index = dict(bios_m5.reverse_index)
        agent_m6 = Agent(world, bios_m6, oracle=oracle, verbose=True)

        print("\n=== Эп A: svc_e (первый раз, oracle ожидается 1 вызов) ===")
        r_ea = agent_m6.run_episode("svc_e")

        tokens_after_a = oracle.total_tokens
        calls_after_a = oracle.n_calls

        print("\n=== Эп B: svc_e (повторно, oracle ожидается 0 вызовов) ===")
        r_eb = agent_m6.run_episode("svc_e")

        print("\n" + "=" * 60)
        print("СВОДКА M6")
        print("=" * 60)
        print(f"  Эп A svc_e: success={r_ea.success}, trials={r_ea.n_trials}, oracle={r_ea.oracle_calls}")
        print(f"  Эп B svc_e: success={r_eb.success}, trials={r_eb.n_trials}, oracle={r_eb.oracle_calls}")
        print(f"  oracle.n_calls={oracle.n_calls}, total_tokens={oracle.total_tokens}")
        print(f"  reverse_index: {bios_m6.reverse_index}")
        print(f"  bad_config: {sorted(bios_m6.bad_config)}")

        if not r_ea.success:
            errors.append(f"Эп A: ожидали success, got error={r_ea.error!r}")
        if r_ea.oracle_calls != 1:
            errors.append(f"Эп A: ожидали oracle_calls=1, got {r_ea.oracle_calls}")
        if not r_eb.success:
            errors.append(f"Эп B: ожидали success, got error={r_eb.error!r}")
        if r_eb.oracle_calls != 0:
            errors.append(f"Эп B: ожидали oracle_calls=0, got {r_eb.oracle_calls}")
        if oracle.total_tokens == 0:
            errors.append("oracle.total_tokens=0 — учёт не работает")
        if oracle.total_tokens != tokens_after_a:
            errors.append(f"Токены выросли после Эп B: {tokens_after_a} → {oracle.total_tokens}")
        if ("svc_e", "bad") not in bios_m6.bad_config:
            errors.append("bad_config не содержит (svc_e,bad)")

    # --- Проверки M3-M5b ---
    print("\n" + "=" * 60)
    print("СВОДКА M3-M5b")
    print("=" * 60)
    rows = [
        ("svc_a", "heavy",   r_a,  4),
        ("svc_d", "heavy",   r_d,  1),
        ("svc_b", "light",   r_b,  2),
        ("svc_c", "heavy",   r_c,  2),
        ("svc_f", "xlarge",  r_f,  None),  # anti-hardcode: trials = log2(2048/64) + 1 = 6
    ]
    for name, wc, r, expected_trials in rows:
        print(f"  {name:6s} [{wc:6s}]  trials={r.n_trials}  ok={r.success}")

    for name, _, r, expected_trials in rows:
        if not r.success:
            errors.append(f"{name}: ожидали success, got {r.error!r}")
        if expected_trials is not None and r.n_trials != expected_trials:
            errors.append(f"{name}: ожидали {expected_trials} проб, got {r.n_trials}")

    if r_d.n_trials != 1:
        errors.append(f"svc_d: ожидали 1 пробу (transfer), got {r_d.n_trials}")
    if bios_m5.mem_threshold.get("heavy") != 512:
        errors.append(f"mem_threshold[heavy]: ожидали 512, got {bios_m5.mem_threshold.get('heavy')}")
    if bios_m5.mem_threshold_svc.get("svc_c") != 1024:
        errors.append(f"mem_threshold_svc[svc_c]: ожидали 1024, got {bios_m5.mem_threshold_svc.get('svc_c')}")
    if bios_m5.mem_threshold.get("xlarge") != 2048:
        errors.append(f"mem_threshold[xlarge]: ожидали 2048 (anti-hardcode), got {bios_m5.mem_threshold.get('xlarge')}")

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
