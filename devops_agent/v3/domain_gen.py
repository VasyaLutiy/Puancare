"""
domain_gen.py — MetaDomain → текст domain.pddl + problem.pddl; решение через FD.

Порождает форму рукописного v2-домена (types/predicates/actions), но ИЗ РЕЕСТРА.
Граница #8: чистый STRIPS — без :functions, :metric, чисел. Значение рычага (память N,
config X) даёт агент-политика, НЕ план; планировщик лишь секвенирует set_*→deploy и
доказывает разрешимость сгенерённого домена.

Эталон, который должен получиться после обучения mem+config:
  (:types service - object)
  (:predicates (running ?s)(mem_ok ?s)(config_ok ?s))
  (:action set_mem ... :effect (mem_ok ?s))
  (:action set_config ... :effect (config_ok ?s))
  (:action deploy :precondition (and (not(running ?s))(mem_ok ?s)(config_ok ?s)) :effect (running ?s))
"""

import os

from devops_agent.v3.meta_schema import MetaDomain

_OUT_DIR = os.path.join(os.path.dirname(__file__), "_generated")


def generate_domain(domain: MetaDomain, out_dir: str = _OUT_DIR) -> str:
    """
    Сериализовать domain.pddl и вернуть путь.
    TODO(v3): types (service); predicates (running + каждый из domain.predicates);
              по :action на каждую intervention (precond=requires, effect=establishes);
              deploy: precond = (and (not (running ?s)) + каждый из deploy_requires).
              Детерминированный порядок (sorted), как в problem_builder.py.
    """
    raise NotImplementedError("TODO(v3): сериализация домена из реестра")


def generate_problem(goal_service: str, out_dir: str = _OUT_DIR) -> str:
    """
    Тривиальная проблема: (:objects <goal> - service) (:init) (:goal (running <goal>)).
    TODO(v3): записать problem.pddl, вернуть путь.
    """
    raise NotImplementedError("TODO(v3): сериализация проблемы")


def solve(domain_path: str, problem_path: str) -> list[str] | None:
    """
    Решить сгенерённый домен. КОПИЯ движка из devops_agent/bios.py::plan():
      PDDLReader().parse_problem(domain_path, problem_path)
      OneshotPlanner(name="fast-downward-opt").solve(p)
      SOLVED_OPTIMALLY|SOLVED_SATISFICING → [str(a) for a in plan.actions], иначе None.
    TODO(v3): перенести тело (импорты unified_planning внутри, как в bios.py).
    """
    raise NotImplementedError("TODO(v3): скопировать движок из bios.plan()")
