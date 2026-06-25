"""
task_agent.py — агент, решающий ЗАДАЧУ «подними сущность X здоровой», читая/дополняя базу.

Задача (eid, raw) стучится. Агент:
  знаю raw (есть в KB)  → ДОВЕРЯЮ: применяю (поднять deps + provision) без заземления;
  не знаю               → ЗАЗЕМЛЯЮ (perceive+ground), кладу в KB, потом применяю.
Стоимость = (LLM-вызовы, docker-проб). База ценна, если узнавание ПРОПУСКАЕТ дорогое заземление.
"""

from devops_agent.v4.sandbox import Sandbox
from devops_agent.v6.grounding import BASELINE, Grounder
from devops_agent.v6.nlu import KNOB_KIND, LiveNLU


class CountingSandbox(Sandbox):
    """Песочница, считающая provision-вызовы (для замера стоимости задачи)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_provisions = 0

    def provision(self, *a, **k):
        self.n_provisions += 1
        return super().provision(*a, **k)


def _raw_key(raw: dict) -> str:
    """Стабильный признак сырья для узнавания (напр. образ). Узнавание дёшево; результат проверяет мир."""
    return raw.get("image", "?")


def _apply(sandbox, eid: str, raw_deps: list) -> bool:
    """Поднять зависимости, затем сущность (generous). Возврат: дошёл ли до running."""
    for d in raw_deps:
        if d in sandbox.truth:
            sandbox.provision(d, knobs=BASELINE)
    return sandbox.provision(eid, knobs=BASELINE).phase == "running"


def _learn(eid, raw, raw_deps, kb, key, nlu, grounder):
    """Незнакомое: воспринять (LLM) + ЗАЗЕМЛИТЬ (дорого) → recipe в KB."""
    nlu.perceive_entity(raw, eid)                          # 1 LLM (восприятие)
    known_deps = [d for d in raw_deps if d in grounder.sb.truth]
    grounded, _ = grounder.ground(eid, list(KNOB_KIND), known_deps)
    kb[key] = {"levers": grounded["levers"] if grounded else [],
               "deps": grounded["deps"] if grounded else []}
    return kb[key]


def solve(task, kb, sandbox, nlu, grounder) -> tuple:
    """task=(eid, raw). → (success, {llm, provisions, path})."""
    eid, raw = task
    p0, llm0 = sandbox.n_provisions, nlu.n_calls
    key = _raw_key(raw)
    raw_deps = raw.get("connects_to", [])

    if key in kb:                                          # ТЁПЛО: знаю → доверяю, заземление ПРОПУСКАЮ
        path = "WARM (узнал → доверяю, без заземления)"
        ok = _apply(sandbox, eid, raw_deps)
        if not ok:                                         # доверие не оправдалось → доучить
            path = "WARM→fallback COLD"
            _learn(eid, raw, raw_deps, kb, key, nlu, grounder)
            ok = _apply(sandbox, eid, raw_deps)
    else:                                                  # ХОЛОДНО: незнакомо → заземляю, не действую вслепую
        path = "COLD (perceive + ground + learn)"
        _learn(eid, raw, raw_deps, kb, key, nlu, grounder)
        ok = _apply(sandbox, eid, raw_deps)

    return ok, {"llm": nlu.n_calls - llm0, "provisions": sandbox.n_provisions - p0, "path": path}
