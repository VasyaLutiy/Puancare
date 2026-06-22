"""
Реестр синтетических сервисов.

Агент видит только SERVICES (image, workload_class, cmd).
_GROUND_TRUTH — наше знание; агент его не получает.
"""

MEM_BUCKETS = [64, 128, 256, 512, 1024]  # MiB

# Python overhead в 3.11-slim ~40 MiB; учтено в выборе footprint.
# bytearray(N) нулирует память → все страницы физически выделяются.
def _alloc_cmd(mib: int) -> str:
    """
    Python one-liner для Docker -c.
    bytearray на Linux ленив (lazy mmap/calloc) — страницы не выделяются
    физически до первого обращения. Явно трогаем каждую страницу (4 KiB),
    чтобы RSS отражал реальное потребление и OOM срабатывал честно.
    """
    n = mib * 1024 * 1024
    return (
        f"import time\n"
        f"n={n}\n"
        f"x=bytearray(n)\n"
        f"for i in range(0,n,4096):x[i]=1\n"
        f"time.sleep(5)"
    )


SERVICES = {
    "svc_a": {
        "image": "python:3.11-slim",
        "workload_class": "heavy",
        # физический footprint ~350 MiB → OOM при ≤256 MiB, безопасно при ≥512 MiB
        "cmd": ["python", "-c", _alloc_cmd(350)],
    },
    "svc_b": {
        "image": "python:3.11-slim",
        "workload_class": "light",
        # физический footprint ~65 MiB → OOM при 64 MiB, безопасно при ≥128 MiB
        "cmd": ["python", "-c", _alloc_cmd(65)],
    },
    "svc_c": {
        "image": "python:3.11-slim",
        "workload_class": "heavy",
        # физический footprint ~800 MiB → OOM при ≤512 MiB, безопасно при ≥1024 MiB
        "cmd": ["python", "-c", _alloc_cmd(800)],
    },
}

# Истина: минимальный безопасный бакет для каждого сервиса.
# Агент должен это ВЫУЧИТЬ, не получить готовым.
_GROUND_TRUTH = {
    "svc_a": {"footprint_mib": 350, "min_safe_bucket": 512},
    "svc_b": {"footprint_mib": 65,  "min_safe_bucket": 128},
    "svc_c": {"footprint_mib": 800, "min_safe_bucket": 1024},
}
