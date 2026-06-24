"""
harness_v3.py — §8 для v3. Доказывает «самообучение, не LLM-фронтенд».

Стиль run_experiment.py, но метрика — кривая llm_calls НА ЗАДАЧУ (живой LiveProposer).
Поток: [mem/A холод, mem/B перенос, cfg/svc_e холод, cfg-повтор перенос, combined-novel F'].
Ожидаемая кривая: [1, 0, 1, 0, 0].

Вердикт ЖИВ ⇔ все три:
  1. Перенос:     llm_calls(перенос) < llm_calls(холод) того же структ-класса (идеально 0).
  2. Амортизация: после холодного всплеска каждого класса хвост → 0.
  3. Детерминизм: повтор полного прогона → ИДЕНТИЧНЫЙ сгенерённый domain.pddl
                  (канонизация имён гарантирует; сравниваем нормализованный текст).

F' (novel-task фальсификатор, аналог секции F из run_experiment.py):
  сервис, которому нужны И память, И config, НИКОГДА не виденный, инъекция ТОЛЬКО данными
  (_SERVICES[...] = {...} в рантайме + cleanup). Должен сойтись с 0 НОВЫХ LLM-вызовов.
  Если потребовал свежий LLM → печать «не обобщается» — часть вердикта, не маскируем.

Гварды: в сгенерённом domain.pddl нет :metric/чисел (#8 не вернулся);
        в seed() нет mem_ok/config_ok (порождены, не вписаны).
"""

import sys

# TODO(v3): импорты — World, MetaBios, MetaAgent, LiveProposer, services._SERVICES, _alloc_*_cmd


def run_section_amortization():
    """A: прогнать поток задач, собрать кривую llm_calls/задача. TODO(v3)."""
    raise NotImplementedError("TODO(v3): §A амортизация")


def run_section_f_novel_task():
    """F': инъекция combined-novel сервиса (данные), эпизод, ассерт 0 новых LLM. TODO(v3)."""
    raise NotImplementedError("TODO(v3): §F' novel-task")


def print_verdict(curve: list, determinism_ok: bool, f_ok: bool) -> bool:
    """3 критерия → ЖИВ/ХОРОНИМ + honest-notes. TODO(v3): порт print_verdict из run_experiment.py."""
    raise NotImplementedError("TODO(v3): вердикт")


def main() -> None:
    print("=== §8-v3 ХАРНЕСС (живой LLM) ===")
    raise NotImplementedError("TODO(v3): оркестрация A→F'→вердикт + гварды grep")


if __name__ == "__main__":
    main()
