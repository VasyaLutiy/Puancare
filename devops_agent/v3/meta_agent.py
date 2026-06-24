"""
meta_agent.py — петля-выращиватель. Обобщение v2 agent.py.

Цикл одного эпизода:
  (пере)solve домена из реестра → выбрать значения рычагов → World.apply →
    running? → record_running + амортизация → успех
    gap (phase!=running): symptom = WorldModel.classify(obs)
      covered_symptoms[symptom]=D (КЭШ) → strategy.on_gap → 0 LLM
      холод                          → proposer.propose → mint_dimension → +1 LLM, домен пересолвить
      затем: advance значения рычага D стратегией; None → исчерпание (ceiling/no_config).

Метрика §8-v3: llm_calls на эпизод (дельта bios.llm_calls). Граница v2: значение рычага
даёт стратегия/политика; план — только логика (что сделать и в каком порядке, не сколько).
Домен пересолвится ТОЛЬКО когда вырос (минтинг) — иначе план переиспользуется.
"""

import sys
from dataclasses import dataclass, field

from devops_agent.model.ontology import WorldModel
from devops_agent.world import Obs, World
from devops_agent.v3 import domain_gen
from devops_agent.v3.meta_bios import MetaBios
from devops_agent.v3.proposer import ReplayProposer, SchemaProposer
from devops_agent.v3.strategies import strategy_for

_wm = WorldModel.load()


@dataclass
class Trial:
    values: dict                 # {actuator: value} в этой пробе
    obs: Obs
    minted: str | None = None    # имя предиката, если на этой пробе чеканили
    llm_used: bool = False

    def __str__(self) -> str:
        m = f" mint={self.minted}" if self.minted else ""
        return f"Trial({self.values}, phase={self.obs.phase!r}{m})"


@dataclass
class EpisodeResult:
    svc_name: str
    success: bool
    trials: list = field(default_factory=list)
    llm_calls: int = 0
    error: str = ""

    @property
    def n_trials(self) -> int:
        return len(self.trials)


class MetaAgent:
    def __init__(self, world: World, bios: MetaBios, proposer: SchemaProposer, verbose: bool = True):
        self.world = world
        self.bios = bios
        self.proposer = proposer
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run_episode(self, svc_name: str) -> EpisodeResult:
        if svc_name not in self.bios.services:
            raise ValueError(f"Unknown service: {svc_name!r}")

        bios = self.bios
        wc = bios.services[svc_name]["workload_class"]
        llm_start = bios.llm_calls
        values = {ak: bios.initial_value(ak, svc_name) for ak in bios.actuators}
        trials: list = []
        plan = None  # пересолвится при первом проходе и после каждого минтинга

        self._log(f"\n[episode] {svc_name} (wc={wc!r}, start={values})")

        while True:
            # --- (пере)solve: домен из реестра, план доказывает достижимость + секвенс ---
            if plan is None:
                dpath = domain_gen.generate_domain(bios.domain)
                ppath = domain_gen.generate_problem(svc_name)
                plan = domain_gen.solve(dpath, ppath)
                self._log(f"  plan: {plan}")
                if plan is None:
                    return EpisodeResult(svc_name, False, trials, bios.llm_calls - llm_start, "no_plan")

            # --- Исчерпание рычага (стратегия вернула None) ---
            if values["memory"] is None:
                return EpisodeResult(svc_name, False, trials, bios.llm_calls - llm_start, "ceiling")
            if values["config"] is None:
                return EpisodeResult(svc_name, False, trials, bios.llm_calls - llm_start, "no_config")

            # --- Act ---
            self._log(f"  try: mem={values['memory']}m config={values['config']!r}")
            obs = self.world.apply({
                "service": svc_name,
                "mem_bucket": values["memory"],
                "config": values["config"],
            })
            self._log(f"  obs:  {obs}")

            # --- Success ---
            if obs.phase == "running":
                bios.record_running(svc_name, values["memory"], values["config"])
                trials.append(Trial(dict(values), obs))
                self._log(f"  ✓ running @ {values} — за {len(trials)} проб(ы)")
                return EpisodeResult(svc_name, True, trials, bios.llm_calls - llm_start)

            # --- Gap: классифицируем + (кэш | холодное предложение) ---
            symptom = _wm.classify(obs)
            minted = None
            llm_used = False
            D = bios.is_covered(symptom)
            if D is None:
                # ХОЛОД: спросить LLM (cost-gated), вшить гипотезу как измерение
                ctx = {"service": svc_name, "workload_class": wc, "values": dict(values)}
                proposal = self.proposer.propose(symptom, ctx, bios.known_actuators())
                D = proposal["dimension"]
                minted = bios.mint_dimension(D, proposal["kind"], symptom)
                bios.llm_calls += 1
                llm_used = True
                plan = None  # домен вырос → пересолвить
                self._log(
                    f"  cold: {symptom!r} → mint dim={D!r} kind={proposal['kind']!r} "
                    f"pred={minted!r} (LLM #{self.proposer.n_calls}; llm_proposed={proposal.get('_llm_name')!r})"
                )
            else:
                self._log(f"  cache: {symptom!r} → dim={D!r} (0 LLM)")

            # --- Advance значения рычага D через его стратегию ---
            strat = strategy_for(bios.strategies[D])
            new_val = strat.on_gap(bios, svc_name, values[D])
            self._log(f"  step: {D} {values[D]} → {new_val}")
            values[D] = new_val
            trials.append(Trial(dict(values), obs, minted=minted, llm_used=llm_used))
            # цикл: исчерпание (None) поймается в начале следующего прохода


# ---------------------------------------------------------------------------
# Self-test — REPLAY-режим (без токенов). Регрессия ПОЛНОЙ ПЕТЛИ на реальном docker.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== MetaAgent self-test (REPLAY — петля разрыв→mint→verify→amortize, 0 токенов) ===\n")
    errors = []

    bios = MetaBios.seed(["svc_a", "svc_d", "svc_b", "svc_e"])

    # Гвард семени: НИ mem_ok, НИ config_ok в стартовом домене
    seed_domain = open(domain_gen.generate_domain(bios.domain)).read()
    if "mem_ok" in seed_domain or "config_ok" in seed_domain:
        errors.append("СЕМЯ содержит mem_ok/config_ok — должны быть порождены, не вписаны!")

    agent = MetaAgent(World(), bios, ReplayProposer(), verbose=True)

    print("=== Эп 1: svc_a (heavy, memory холодное — mint) ===")
    r_a = agent.run_episode("svc_a")
    print("\n=== Эп 2: svc_d (heavy, перенос — 0 LLM, 1 проба) ===")
    r_d = agent.run_episode("svc_d")
    print("\n=== Эп 3: svc_b (light, memory уже covered — 0 LLM) ===")
    r_b = agent.run_episode("svc_b")
    print("\n=== Эп 4: svc_e (нужны mem+config; config холодное — mint) ===")
    r_e = agent.run_episode("svc_e")

    # --- Сводка ---
    print("\n" + "=" * 60)
    print("СВОДКА (replay)")
    print("=" * 60)
    rows = [("svc_a", r_a), ("svc_d", r_d), ("svc_b", r_b), ("svc_e", r_e)]
    for name, r in rows:
        print(f"  {name:6s}  ok={r.success}  trials={r.n_trials}  llm_calls={r.llm_calls}")
    print(f"  mem_threshold:     {bios.mem_threshold}")
    print(f"  mem_threshold_svc: {bios.mem_threshold_svc}")
    print(f"  covered_symptoms:  {bios.covered_symptoms}")
    print(f"  bad_config:        {sorted(bios.bad_config)}")
    llm_curve = [r.llm_calls for _, r in rows]
    print(f"  LLM-кривая на задачу: {llm_curve}  (ожидаем [1,0,0,1] — амортизация)")

    final_domain = open(domain_gen.generate_domain(bios.domain)).read()
    print("\n--- Выученный domain.pddl (порождён из разрывов) ---")
    print(final_domain)

    # --- Проверки ---
    for name, r in rows:
        if not r.success:
            errors.append(f"{name}: ожидали success, got error={r.error!r}")
    if r_a.llm_calls != 1:
        errors.append(f"svc_a: ожидали llm_calls=1 (mint memory), got {r_a.llm_calls}")
    if r_d.llm_calls != 0 or r_d.n_trials != 1:
        errors.append(f"svc_d: ожидали перенос (0 LLM, 1 проба), got llm={r_d.llm_calls} trials={r_d.n_trials}")
    if r_b.llm_calls != 0:
        errors.append(f"svc_b: memory уже covered → ожидали 0 LLM, got {r_b.llm_calls}")
    if r_e.llm_calls != 1:
        errors.append(f"svc_e: ожидали llm_calls=1 (mint config), got {r_e.llm_calls}")
    if llm_curve != [1, 0, 0, 1]:
        errors.append(f"LLM-кривая {llm_curve} != [1,0,0,1] (амортизация сломана)")
    if bios.mem_threshold.get("heavy") != 512:
        errors.append(f"mem_threshold[heavy] != 512: {bios.mem_threshold.get('heavy')}")
    if bios.mem_threshold.get("light") != 128:
        errors.append(f"mem_threshold[light] != 128: {bios.mem_threshold.get('light')}")
    if bios.mem_threshold.get("standard") != 512:
        errors.append(f"mem_threshold[standard] != 512: {bios.mem_threshold.get('standard')}")
    for need in ("mem_ok", "config_ok", "set_mem", "set_config", "deploy"):
        if need not in final_domain:
            errors.append(f"выученный домен без {need}")
    if ("svc_e", "bad") not in bios.bad_config:
        errors.append("bad_config без (svc_e,'bad') — config-обучение не сработало")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("MetaAgent replay-selftest OK — домен вырос из разрывов, LLM амортизировался [1,0,0,1].")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v3.meta_agent --selftest")
