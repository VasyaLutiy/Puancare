"""
grounding.py — ЗАЗЕМЛЕНИЕ структуры в мир (E3 на уровне сущностей).

LLM воспринимает и ЧАСТО переутверждает (галлюцинирует ресурсы/связи из приоров). Заземление
проверяет КАЖДЫЙ заявленный элемент интервенцией в песочнице: элемент РЕАЛЕН ⟺ воздействие на
него меняет исход сущности. Не подтвердилось → ПРУНИМ. Это и делает базу знаний заземлённой,
а не эхом LLM. Реальность переспоривает LLM поэлементно.

Оракул реальности — v4 Sandbox (скрытый truth + provision-проба). Без PDDL.
"""

from devops_agent.v4.sandbox import Sandbox, _run

# «стресс» и «безопасное» значения рычагов (имена = knob'ы мира/API)
BASELINE = {"mem": 4096, "config": "good", "pool": 150}
STRESS = {"mem": 32, "config": "bad", "pool": 1}


class Grounder:
    def __init__(self, sandbox: Sandbox):
        self.sb = sandbox

    def _up(self, name):
        self.sb.provision(name, knobs=BASELINE)

    def _down(self, name):
        _run(["docker", "rm", "-f", self.sb._c(name)])

    def ground(self, entity: str, levers: list, deps: list):
        """Заявленные levers+deps → (подтверждённая структура, verdicts по каждому элементу)."""
        verdicts = {}
        for d in deps:
            self._up(d)
        base = self.sb.provision(entity, knobs=BASELINE)
        if base.phase != "running":
            return None, {"_baseline": f"не удалось получить рабочий baseline: {base.phase}"}

        # рычаги: стрессим один, остальные на baseline; деградация → рычаг реален
        for L in levers:
            k = dict(BASELINE)
            k[L] = STRESS.get(L, STRESS["mem"])
            obs = self.sb.provision(entity, knobs=k)
            real = obs.phase != "running"
            verdicts[f"lever:{L}"] = ("РЕАЛЕН" if real else "ПРУНЕД", obs.phase)

        # зависимости: роняем одну (остальные подняты); деградация → связь реальна
        for D in deps:
            self._down(D)
            for other in deps:
                if other != D:
                    self._up(other)
            obs = self.sb.provision(entity, knobs=BASELINE)
            real = obs.phase != "running"
            verdicts[f"dep:{D}"] = ("РЕАЛЕН" if real else "ПРУНЕД", obs.phase)
            self._up(D)

        grounded = {
            "levers": [L for L in levers if verdicts[f"lever:{L}"][0] == "РЕАЛЕН"],
            "deps": [D for D in deps if verdicts[f"dep:{D}"][0] == "РЕАЛЕН"],
        }
        return grounded, verdicts


# ---------------------------------------------------------------------------
# Демо: кандидат ПЕРЕУТВЕРЖДАЕТ (config и dep logger галлюцинированы) → реальность срезает.
# ---------------------------------------------------------------------------

_TRUTH = {
    "db":     {"footprint": 350, "deps": ["cache"]},   # РЕАЛЬНО: память важна, зависит от cache
    "cache":  {"footprint": 64},                        # настоящая зависимость (поднимаемая)
    "logger": {"footprint": 64},                        # существует, но db от него НЕ зависит
}


def main() -> None:
    import sys
    print("=== v6 ЗАЗЕМЛЕНИЕ: реальность срезает галлюцинации LLM (E3 на сущностях) ===\n")
    sb = Sandbox(_TRUTH, network="v6ground")
    sb.prefetch(); sb.reset()
    g = Grounder(sb)

    # кандидат-фрагмент (что «увидел» LLM): config и logger — переутверждение
    claimed_levers = ["mem", "config"]
    claimed_deps = ["cache", "logger"]
    print(f"  LLM заявил: levers={claimed_levers}, deps={claimed_deps}")
    print("  (правда среды: важна только mem; зависимость только cache. config и logger — галлюцинация)\n")
    try:
        grounded, verdicts = g.ground("db", claimed_levers, claimed_deps)
    finally:
        sb.teardown()

    if grounded is None:
        print(f"  baseline не получен: {verdicts}")
        sys.exit(1)
    for elem, (verdict, phase) in verdicts.items():
        print(f"    {elem:14} → {verdict}  (проба: {phase})")
    print(f"\n  заземлённая структура: {grounded}")

    claimed = len(claimed_levers) + len(claimed_deps)
    kept = len(grounded["levers"]) + len(grounded["deps"])
    pruned = claimed - kept
    real_kept = "mem" in grounded["levers"] and "cache" in grounded["deps"]
    halluc_pruned = "config" not in grounded["levers"] and "logger" not in grounded["deps"]
    print("=" * 60)
    if pruned > 0 and real_kept and halluc_pruned:
        print(f"ВЕРДИКТ: РЕАЛЬНОСТЬ ПЕРЕСПОРИЛА LLM ✓ — из {claimed} заявленных оставлено {kept} (реальные),")
        print(f"  срезано {pruned} галлюцинаций (config, logger). База знаний ЗАЗЕМЛЕНА, не эхо LLM.")
        print("  Заземление с зубами: проба интервенцией пруит то, что реальность не подтверждает.")
    else:
        print(f"ВЕРДИКТ: ТОЛЕРАНТНО/ХРУПКО ✗ — kept={kept} pruned={pruned} "
              f"real_kept={real_kept} halluc_pruned={halluc_pruned} (грунтер беззуб или сломан)")


if __name__ == "__main__":
    main()
