"""
Реестр синтетических сервисов.

Агент получает только agent_view(svc_name) → {name, workload_class, image}.
cmd и числа аллокации — внутренняя механика sandbox, агент её не видит.

_GROUND_TRUTH — наше знание (ground truth); агент его не получает никогда.
"""

MEM_BUCKETS = [64, 128, 256, 512, 1024]  # MiB

_IMAGE = "python:3.11-slim"


def _alloc_cmd(mib: int) -> str:
    """
    Python-код для Docker -c.
    bytearray на Linux ленив (lazy mmap) — страницы не выделяются
    физически до первого обращения. Явно трогаем каждую страницу (4 KiB),
    чтобы RSS точно отражал footprint и OOM срабатывал честно.
    """
    n = mib * 1024 * 1024
    return (
        f"import time\n"
        f"n={n}\n"
        f"x=bytearray(n)\n"
        f"for i in range(0,n,4096):x[i]=1\n"
        f"time.sleep(5)"
    )


# Полные записи сервисов — содержат cmd (механику sandbox).
# Агент к этому словарю НЕ обращается напрямую; только через agent_view().
_SERVICES = {
    "svc_a": {
        "image": _IMAGE,
        "workload_class": "heavy",
        # footprint ~350 MiB → OOM при ≤256 MiB, safe при ≥512 MiB
        "cmd": ["python", "-c", _alloc_cmd(350)],
    },
    "svc_b": {
        "image": _IMAGE,
        "workload_class": "light",
        # footprint ~65 MiB → OOM при 64 MiB, safe при ≥128 MiB
        "cmd": ["python", "-c", _alloc_cmd(65)],
    },
    "svc_c": {
        "image": _IMAGE,
        "workload_class": "heavy",
        # footprint ~800 MiB → OOM при ≤512 MiB, safe при ≥1024 MiB
        # M5b: тоже heavy, но другой порог → класс-правило (heavy→512) ломается.
        # Агент не находит более тонкой фичи → специализация выходит на per-instance исключение.
        "cmd": ["python", "-c", _alloc_cmd(800)],
    },
    "svc_d": {
        "image": _IMAGE,
        "workload_class": "heavy",
        # footprint ~380 MiB → OOM при ≤256 MiB, safe при ≥512 MiB
        # M5: чистый zero-shot перенос правила heavy→512 с svc_a
        "cmd": ["python", "-c", _alloc_cmd(380)],
    },
}

# Истина: минимальный безопасный бакет для каждого сервиса.
# Агент должен это ВЫУЧИТЬ через пробы, не получить готовым.
_GROUND_TRUTH = {
    "svc_a": {"footprint_mib": 350, "min_safe_bucket": 512},
    "svc_b": {"footprint_mib": 65,  "min_safe_bucket": 128},
    "svc_c": {"footprint_mib": 800, "min_safe_bucket": 1024},  # heavy, но порог выше
    "svc_d": {"footprint_mib": 380, "min_safe_bucket": 512},
}


def agent_view(svc_name: str) -> dict:
    """
    Единственный способ для агента узнать о сервисе.
    Возвращает только наблюдаемые признаки — без cmd, без чисел аллокации.
    """
    svc = _SERVICES[svc_name]
    return {
        "name": svc_name,
        "workload_class": svc["workload_class"],
        "image": svc["image"],
    }


def all_service_names() -> list[str]:
    return list(_SERVICES.keys())
