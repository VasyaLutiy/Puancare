"""
meta_bios.py — MetaBios: выращенный домен + выученные значения + амортизация.

Обобщение v2 BiosState: вместо рукописной схемы — растущий MetaDomain. Старт = только
рычаги (из agent_view) + пустой домен с одним deploy (БЕЗ mem_ok/config_ok — их минтит опыт).
covered_symptoms — зеркало v2 reverse_index (кэш symptom→измерение, основа амортизации).
"""

from dataclasses import dataclass, field

from devops_agent.services import agent_view
from devops_agent.v3.meta_schema import (
    Actuator,
    Intervention,
    MetaDomain,
    Predicate,
    canonical_predicate_name,
)

MEM_BASE: int = 64        # порт bios.py — стартовый размер удвоения
MEM_CEILING: int = 16384  # жёсткий потолок

# рычаг → имя PDDL-действия (совпадает с рукописным v2: set_mem / set_config)
_ACTION = {"memory": "set_mem", "config": "set_config"}


@dataclass
class MetaBios:
    """Живёт на весь сеанс — схема и значения копятся между задачами."""
    services: dict                       # {svc: agent_view(svc)}
    actuators: dict                      # {key: Actuator}
    domain: MetaDomain
    strategies: dict                     # dimension → kind
    covered_symptoms: dict               # symptom → dimension (пусто = холод → LLM)
    mem_threshold: dict = field(default_factory=dict)       # wc → mib
    mem_threshold_svc: dict = field(default_factory=dict)   # svc → mib (карвинг)
    bad_config: set = field(default_factory=set)            # {(svc, opt)}
    incumbent_config: dict = field(default_factory=dict)    # svc → applied config
    llm_calls: int = 0                   # счётчик; харнесс снимает дельту per-task

    # --- Семя -------------------------------------------------------------
    @classmethod
    def seed(cls, svc_names: list) -> "MetaBios":
        """
        Только рычаги (memory, config) + пустой домен (deploy без предусловий-предикатов).
        НИ mem_ok, НИ config_ok — доказательство, что они порождены опытом, а не вписаны.
        """
        svcs = {n: agent_view(n) for n in svc_names}
        incumbent = {
            n: svcs[n]["current_config"]
            for n in svc_names
            if "current_config" in svcs[n]
        }
        actuators = {
            "memory": Actuator("memory", "mem_bucket", MEM_BASE),
            "config": Actuator("config", "config", "good"),
        }
        return cls(
            services=svcs,
            actuators=actuators,
            domain=MetaDomain(),          # пусто: predicates={}, interventions={}, deploy_requires=[]
            strategies={},
            covered_symptoms={},
            incumbent_config=incumbent,
        )

    def known_actuators(self) -> list:
        return list(self.actuators.keys())

    # --- Минтинг измерения (cold-ветка петли) -----------------------------
    def mint_dimension(self, dimension: str, kind: str, symptom: str) -> str:
        """
        Вшить новое измерение по гипотезе LLM (имя предиката — каноническое, из рычага):
          predicate {dim}_ok → domain.predicates
          intervention set_{dim} (establishes=[{dim}_ok]) → domain.interventions
          deploy_requires += {dim}_ok ; strategies[dim]=kind ; covered_symptoms[symptom]=dim
        """
        pred = canonical_predicate_name(dimension)
        self.domain.predicates[pred] = Predicate(pred, dimension)
        action = _ACTION.get(dimension, f"set_{dimension}")
        self.domain.interventions[action] = Intervention(
            action, establishes=[pred], binds_actuator=dimension
        )
        if pred not in self.domain.deploy_requires:
            self.domain.deploy_requires.append(pred)
        self.strategies[dimension] = kind
        self.covered_symptoms[symptom] = dimension
        return pred

    def is_covered(self, symptom: str):
        return self.covered_symptoms.get(symptom)

    # --- Выбор значений рычагов (политика агента, не план) ----------------
    def _choose_mem(self, svc: str):
        if svc in self.mem_threshold_svc:
            return self.mem_threshold_svc[svc]
        wc = self.services[svc]["workload_class"]
        return self.mem_threshold.get(wc)

    def choose_config(self, svc: str):
        """Порт v2 _choose_config: incumbent если не bad, иначе sorted non-bad, иначе None."""
        opts = self.services[svc].get("config_options")
        if not opts:
            return "cfg_default"                       # сервис без config → always-on dummy
        inc = self.incumbent_config.get(svc)
        if inc is not None and not self.is_bad_config(svc, inc):
            return inc
        for o in sorted(opts):
            if not self.is_bad_config(svc, o):
                return o
        return None                                    # все known-bad → нет валидного config

    def initial_value(self, actuator_key: str, svc: str):
        """Стартовое значение рычага в эпизоде (до первого разрыва)."""
        if actuator_key == "memory":
            return self._choose_mem(svc) or MEM_BASE   # known threshold → перенос за 1 пробу
        if actuator_key == "config":
            return self.choose_config(svc)
        raise ValueError(f"неизвестный рычаг {actuator_key!r}")

    # --- Запись успеха ----------------------------------------------------
    def record_running(self, svc: str, mem: int, config) -> None:
        """Порт success-ветки agent.py: class threshold | карвинг | перенос; + incumbent."""
        wc = self.services[svc]["workload_class"]
        if wc in self.mem_threshold and mem != self.mem_threshold[wc]:
            self.mem_threshold_svc[svc] = mem          # карвинг (класс-правило не сработало)
        elif wc not in self.mem_threshold:
            self.mem_threshold[wc] = mem               # свежее class-правило
        # else: mem == class threshold → перенос, без изменений
        if "config_options" in self.services[svc]:
            self.incumbent_config[svc] = config

    def mark_bad_config(self, svc: str, opt) -> None:
        self.bad_config.add((svc, opt))

    def is_bad_config(self, svc: str, opt) -> bool:
        return (svc, opt) in self.bad_config
