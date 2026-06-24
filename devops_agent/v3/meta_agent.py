"""
meta_agent.py — петля-выращиватель. Обобщение v2 agent.py.

Цикл одного эпизода:
  gen domain+problem → solve(FD) → выбрать значения рычагов → World.apply →
    running? → record_running + амортизация → успех
    gap (phase!=running): symptom = WorldModel.classify(obs)
      covered_symptoms[symptom]=D (КЭШ) → strategy.on_gap → replan         [0 LLM]
      холод                          → proposer.propose → mint_dimension → strategy.initial → replan [+1 LLM]
      verify/refute (грубый MVP): стратегия исчерпана (ceiling/все config bad) → unmint + 1 ретрай.

Метрика для §8-v3: llm_calls на эпизод (дельта bios.llm_calls). Граница v2: значение рычага
даёт стратегия/политика, план — только логика (что сделать, не сколько).
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
    values: dict          # {actuator_key: value} в этой пробе
    obs: Obs
    minted: str | None = None   # имя измерения, если на этой пробе чеканили
    llm_used: bool = False


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
        """
        Главный цикл. TODO(v3): реализовать согласно докстрингу модуля + DevopsPlanV3.md §Петля.
        Снять llm_calls на входе, вернуть дельту в EpisodeResult.llm_calls.
        """
        raise NotImplementedError("TODO(v3): meta-петля run_episode")


# ---------------------------------------------------------------------------
# Self-test — REPLAY-режим (без токенов). Регрессия ПРОВОДКИ, не доказательство.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """
    TODO(v3): replay-прогон на srv (docker):
      bios = MetaBios.seed([...]); agent = MetaAgent(World(), bios, ReplayProposer())
      Эп1 mem-svc (cold, 1 mint) → Эп2 mem-svc (transfer, 0 mint) → Эп3 config-svc (cold, 1 mint).
    Проверки: success; сгенерённый domain.pddl содержит mem_ok+config_ok+3 экшена (≈ эталон v2);
              llm_calls падает к 0 на переносе; g-вард: в семени mem_ok/config_ok НЕ было.
    """
    print("=== MetaAgent self-test (REPLAY — регрессия проводки, не пруф) ===")
    raise NotImplementedError("TODO(v3): replay selftest")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v3.meta_agent --selftest")
