"""
harness_v3.py — §8 для v3. Доказывает «самообучение, не LLM-фронтенд». ЖИВОЙ Azure.

Стиль run_experiment.py, но метрика — кривая llm_calls НА ЗАДАЧУ (LiveProposer).
Поток (run 1): [svc_a, svc_d, svc_b, svc_e, svc_e] на ОДНОМ MetaBios (знания копятся).
Ожидаемая кривая: [1, 0, 0, 1, 0] — 5 задач, всего 2 живых вызова (по одному на класс
симптома oom_killed/unhealthy), дальше 0.

Вердикт ЖИВ ⇔ все:
  1. cold:   каждый новый класс симптома стоит ровно 1 вызов (curve[0]=1, curve[3]=1);
  2. amort:  все повторные/покрытые задачи стоят 0 (curve[1,2,4]=0); total == число классов (2), не задач (5);
  3. F':     novel-task (сервис, нужны И mem, И config; инъекция ДАННЫМИ) сходится с 0 НОВЫХ вызовов;
  4. determ: повтор живого прогона (свежий MetaBios+LLM) → ИДЕНТИЧНЫЙ сгенерённый domain.pddl
             (канонизация имён выше вариативности LLM).

Гварды: в сгенерённом домене нет :metric/:functions (#8); в семени нет mem_ok/config_ok (порождены).
"""

import sys

from devops_agent.services import _IMAGE, _SERVICES, _alloc_and_config_cmd, agent_view
from devops_agent.v3 import domain_gen
from devops_agent.v3.meta_agent import MetaAgent
from devops_agent.v3.meta_bios import MetaBios
from devops_agent.v3.proposer import LiveProposer
from devops_agent.world import World

STREAM = ["svc_a", "svc_d", "svc_b", "svc_e", "svc_e"]

_NOVEL = "svc_novel_combo"   # имя PDDL-объекта ДОЛЖНО начинаться с буквы (не "_")
_NOVEL_CLASS = "premium"
_NOVEL_FOOTPRINT = 350   # MiB, не степень двойки → min safe = 512


# ---------------------------------------------------------------------------
# A. Амортизация — поток задач на живом LLM, кривая llm_calls/задача
# ---------------------------------------------------------------------------

def run_section_amortization(world: World):
    print("=" * 64)
    print("A. АМОРТИЗАЦИЯ — поток задач, живой LLM, кривая llm_calls/задача")
    print("=" * 64)
    bios = MetaBios.seed(["svc_a", "svc_d", "svc_b", "svc_e"])
    seed_domain = open(domain_gen.generate_domain(bios.domain)).read()
    proposer = LiveProposer()
    agent = MetaAgent(world, bios, proposer, verbose=False)

    curve = []
    for svc in STREAM:
        r = agent.run_episode(svc)
        curve.append(r.llm_calls)
        print(f"  → {svc}: ok={r.success} trials={r.n_trials} llm_calls={r.llm_calls}")

    learned_domain = open(domain_gen.generate_domain(bios.domain)).read()
    return {
        "bios": bios, "agent": agent, "proposer": proposer,
        "curve": curve, "seed_domain": seed_domain, "learned_domain": learned_domain,
    }


# ---------------------------------------------------------------------------
# F'. Novel-task фальсификатор — новый класс, нужны mem+config, инъекция ДАННЫМИ
# ---------------------------------------------------------------------------

def run_section_f(world: World, trained_bios: MetaBios, trained_agent: MetaAgent):
    print("\n" + "=" * 64)
    print("F'. NOVEL-TASK ФАЛЬСИФИКАТОР — новый класс premium (mem+config), только данные")
    print("=" * 64)
    print(f"  Инъекция: {_NOVEL} (class={_NOVEL_CLASS}, footprint={_NOVEL_FOOTPRINT}m, config=bad)")
    print("  agent/bios/world/meta_schema НЕ трогаем — только запись в _SERVICES.")

    # Рантайм-инъекция (как секция F в run_experiment.py) — только данные
    _SERVICES[_NOVEL] = {
        "image": _IMAGE,
        "workload_class": _NOVEL_CLASS,
        "config_options": ["good", "bad"],
        "current_config": "bad",
        "cmd": ["python", "-c", _alloc_and_config_cmd(_NOVEL_FOOTPRINT)],
    }
    try:
        # Добавляем сервис в УЖЕ ОБУЧЕННЫЙ bios (знания сохранены → ждём 0 новых LLM)
        trained_bios.services[_NOVEL] = agent_view(_NOVEL)
        trained_bios.incumbent_config[_NOVEL] = trained_bios.services[_NOVEL]["current_config"]
        r = trained_agent.run_episode(_NOVEL)
        print(f"\n  → {_NOVEL}: ok={r.success} trials={r.n_trials} НОВЫХ llm_calls={r.llm_calls}")
        print(f"  mem_threshold[{_NOVEL_CLASS}] = {trained_bios.mem_threshold.get(_NOVEL_CLASS)} (ждём 512)")
        return r
    finally:
        _SERVICES.pop(_NOVEL, None)


# ---------------------------------------------------------------------------
# Детерминизм — повтор живого прогона, сверка сгенерённого домена
# ---------------------------------------------------------------------------

def run_determinism(world: World) -> str:
    print("\n" + "=" * 64)
    print("ДЕТЕРМИНИЗМ — свежий MetaBios + живой LLM, короткий поток [svc_a, svc_e]")
    print("=" * 64)
    bios = MetaBios.seed(["svc_a", "svc_e"])
    agent = MetaAgent(world, bios, LiveProposer(), verbose=False)
    for svc in ["svc_a", "svc_e"]:
        agent.run_episode(svc)
    return open(domain_gen.generate_domain(bios.domain)).read()


# ---------------------------------------------------------------------------
# Вердикт
# ---------------------------------------------------------------------------

def print_verdict(A: dict, f_result, domain_run2: str) -> bool:
    curve = A["curve"]
    seed_domain = A["seed_domain"]
    learned = A["learned_domain"]

    print("\n" + "=" * 64)
    print("ВЕРДИКТ §8-v3")
    print("=" * 64)
    print(f"  Поток:       {STREAM}")
    print(f"  LLM-кривая:  {curve}   (total={sum(curve)}, токенов={A['proposer'].total_tokens})")

    ok_cold = curve[0] == 1 and curve[3] == 1
    ok_amort = curve[1] == 0 and curve[2] == 0 and curve[4] == 0
    ok_total = sum(curve) == 2                       # == число классов симптомов, не задач (5)
    ok_f = bool(f_result.success and f_result.llm_calls == 0)
    ok_det = learned == domain_run2
    ok_guard_metric = ":metric" not in learned and ":functions" not in learned
    ok_guard_seed = "mem_ok" not in seed_domain and "config_ok" not in seed_domain

    print(f"  [1] cold (1 вызов/класс):        {'✓' if ok_cold else '✗'}  (curve[0]={curve[0]}, curve[3]={curve[3]})")
    print(f"  [2] amort (повторы=0, total=2):  {'✓' if ok_amort and ok_total else '✗'}  (5 задач → {sum(curve)} вызова)")
    print(f"  [3] F' (novel, 0 новых LLM):     {'✓' if ok_f else '✗'}  (ok={f_result.success}, llm={f_result.llm_calls})")
    print(f"  [4] детерминизм домена:          {'✓' if ok_det else '✗'}  (run1 == run2)")
    print(f"  [g] гвард #8 (нет :metric):      {'✓' if ok_guard_metric else '✗'}")
    print(f"  [g] гвард семени (нет *_ok):     {'✓' if ok_guard_seed else '✗'}")

    alive = all([ok_cold, ok_amort, ok_total, ok_f, ok_det, ok_guard_metric, ok_guard_seed])
    print()
    if alive:
        print("  ► ВЕРДИКТ: ЖИВ ✓ — схема выращивается из разрывов, LLM амортизируется,")
        print("    novel-task обобщается без новых вызовов, домен детерминирован.")
    else:
        print("  ► ВЕРДИКТ: ХОРОНИМ ✗ — см. провалившиеся критерии выше (не подгоняем).")

    print("\n[HONEST-NOTES v3]")
    print("  • Активны 2 мета-типа: Resource(память)+Setting(config). Dependency/темпоральные — вне MVP.")
    print("  • LLM знает про mem/config — это ОЖИДАЕМО: он генератор гипотез. Доказывает не «LLM глуп»,")
    print("    а кривая llm→0 (амортизация) + F' с 0 новых вызовов + детерминизм артефакта.")
    print("  • Синонимия гасится канонизацией имени из рычага (LLM-имя игнорируется).")
    print("  • Refute грубый (исчерпание стратегии); info-gain выбор пробы отложен. Docker≠k8s.")
    return alive


def main() -> None:
    print("=== §8-v3 ХАРНЕСС (ЖИВОЙ Azure) ===\n")
    world = World()
    A = run_section_amortization(world)
    f_result = run_section_f(world, A["bios"], A["agent"])
    domain_run2 = run_determinism(world)
    alive = print_verdict(A, f_result, domain_run2)
    sys.exit(0 if alive else 1)


if __name__ == "__main__":
    main()
