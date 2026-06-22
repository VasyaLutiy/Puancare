"""
Реестр синтетических сервисов.

Агент получает только agent_view(svc_name) → {name, workload_class, image, config_options?}.
cmd и числа аллокации — внутренняя механика sandbox, агент её не видит.

_GROUND_TRUTH — наше знание (ground truth); агент его не получает никогда.
"""

MEM_BUCKETS = [64, 128, 256, 512, 1024]  # MiB

_IMAGE = "python:3.11-slim"


def _alloc_cmd(mib: int) -> str:
    """
    bytearray на Linux ленив (lazy mmap) — трогаем каждую страницу (4 KiB)
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


def _alloc_and_config_cmd(mib: int) -> str:
    """
    Как _alloc_cmd, но после аллокации читает env CONFIG.
    Если CONFIG != 'good' → sys.exit(3) (sentinel для unhealthy).
    Детерминировано, без NLP: world.py маппит exit_code==3 → phase='unhealthy'.
    """
    n = mib * 1024 * 1024
    return (
        f"import os,time\n"
        f"n={n}\n"
        f"x=bytearray(n)\n"
        f"for i in range(0,n,4096):x[i]=1\n"
        f"config=os.environ.get('CONFIG','')\n"
        f"if config!='good':import sys;sys.exit(3)\n"
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
        # M5b: тоже heavy, но порог выше → per-instance исключение
        "cmd": ["python", "-c", _alloc_cmd(800)],
    },
    "svc_d": {
        "image": _IMAGE,
        "workload_class": "heavy",
        # footprint ~380 MiB → OOM при ≤256 MiB, safe при ≥512 MiB
        # M4: чистый zero-shot перенос правила heavy→512 с svc_a
        "cmd": ["python", "-c", _alloc_cmd(380)],
    },
    "svc_e": {
        "image": _IMAGE,
        "workload_class": "standard",
        # footprint ~350 MiB → OOM при ≤256 MiB, safe при ≥512 MiB
        # M6: нужна И память (≥512), И config=good
        # exit(3) при config≠good → world маппит на phase='unhealthy'
        "config_options": ["good", "bad"],  # наблюдаемый набор, агент видит через agent_view
        "cmd": ["python", "-c", _alloc_and_config_cmd(350)],
    },
}

# Истина: минимальный безопасный бакет (и config для svc_e).
# Агент должен это ВЫУЧИТЬ через пробы, не получить готовым.
_GROUND_TRUTH = {
    "svc_a": {"footprint_mib": 350, "min_safe_bucket": 512},
    "svc_b": {"footprint_mib": 65,  "min_safe_bucket": 128},
    "svc_c": {"footprint_mib": 800, "min_safe_bucket": 1024},
    "svc_d": {"footprint_mib": 380, "min_safe_bucket": 512},
    "svc_e": {"footprint_mib": 350, "min_safe_bucket": 512, "good_config": "good"},
}


def agent_view(svc_name: str) -> dict:
    """
    Единственный способ для агента узнать о сервисе.
    Возвращает только наблюдаемые признаки — без cmd, без числовых footprint.
    config_options включается только если сервис имеет knob конфига.
    """
    svc = _SERVICES[svc_name]
    view: dict = {
        "name": svc_name,
        "workload_class": svc["workload_class"],
        "image": svc["image"],
    }
    if "config_options" in svc:
        view["config_options"] = list(svc["config_options"])
    return view


def all_service_names() -> list[str]:
    return list(_SERVICES.keys())
