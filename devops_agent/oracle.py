"""
oracle.py — LLM-оракул для диагностики симптомов.

Вызывается agent.py только когда reverse_index пуст (cost-gated).
Возвращает ранжированный список причин и учитывает токены.

HONEST-NOTE v1: закрытый словарь причин {"memory", "config"}.
  Не пригоден для симптомов за пределами этого набора.
  Расширение словаря — отдельная задача (v2).
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для импорта utils_azure
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils_azure import AzureJSON  # noqa: E402

from devops_agent.model.ontology import WorldModel as _WorldModel

_wm = _WorldModel.load()


class Oracle:
    """
    LLM-оракул: предлагает ранжированный список причин симптома.

    Агент вызывает suggest_causes() ТОЛЬКО когда reverse_index пуст.
    Oracle не решает когда его звать — это cost-gate на стороне агента.

    Атрибуты:
      n_calls      — сколько раз вызывался оракул
      total_tokens — суммарно использовано токенов (prompt + completion)
    """

    def __init__(self, env_path: str | None = None):
        self._az = AzureJSON(env_path=env_path)
        self.n_calls: int = 0
        self.total_tokens: int = 0

    def suggest_causes(self, symptom: str, context: dict) -> list[str]:
        """
        Возвращает ранжированный список из wm.causes, наиболее вероятное — первым.
        Всегда возвращает хотя бы один элемент (fallback = wm.causes).
        """
        _causes = _wm.causes
        system = (
            "You are a DevOps diagnostic assistant. "
            "Given a service symptom and context, rank root causes by likelihood.\n"
            f"Use ONLY causes from this closed set: {_causes}.\n"
            "Most likely cause first. Respond with valid JSON only."
        )
        user = f"symptom: {symptom!r}\ncontext: {context}"

        result = self._az.ask(
            system=system,
            user=user,
            schema={"causes": f"list, most-likely-first, values only from {_causes}"},
        )

        # Учёт токенов из _last_usage (добавлено в utils_azure.ask)
        if getattr(self._az, "_last_usage", None) is not None:
            self.total_tokens += getattr(self._az._last_usage, "total_tokens", 0)

        self.n_calls += 1

        raw = result.get("causes", [])
        if isinstance(raw, str):
            raw = [c.strip() for c in raw.split(",")]
        valid = [c for c in raw if c in _causes]
        return valid if valid else list(_causes)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== Oracle self-test ===\n")
    errors = []

    oracle = Oracle()

    # Тест 1: unhealthy → ожидаем "config" на первом месте
    causes1 = oracle.suggest_causes(
        "unhealthy",
        {"service": "svc_e", "workload_class": "standard", "mem_bucket": 512},
    )
    print(f"unhealthy → {causes1}  (tokens so far: {oracle.total_tokens})")
    if not causes1 or causes1[0] != "config":
        errors.append(f"unhealthy: expected 'config' first, got {causes1}")

    # Тест 2: oom_killed → ожидаем "memory" на первом месте
    causes2 = oracle.suggest_causes(
        "oom_killed",
        {"service": "svc_a", "workload_class": "heavy", "mem_bucket": 64},
    )
    print(f"oom_killed → {causes2}  (tokens so far: {oracle.total_tokens})")
    if not causes2 or causes2[0] != "memory":
        errors.append(f"oom_killed: expected 'memory' first, got {causes2}")

    print(f"\nn_calls={oracle.n_calls}, total_tokens={oracle.total_tokens}")

    if oracle.n_calls != 2:
        errors.append(f"n_calls: expected 2, got {oracle.n_calls}")
    if oracle.total_tokens == 0:
        errors.append("total_tokens == 0 — учёт не работает")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("Oracle self-test OK.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.oracle --selftest")
