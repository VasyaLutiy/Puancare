"""
Sandbox-Gym: World.apply(spec) -> Obs

spec  = {"service": str, "mem_bucket": int}
Obs   = {phase, exit_code, oom_killed}

Одно применение — один детерминированный результат.
Восприятие = парсинг структуры docker inspect, без NLP.
"""

import json
import subprocess
import sys
from dataclasses import dataclass

from devops_agent.services import (
    MEM_BUCKETS,
    _GROUND_TRUTH,
    _SERVICES,
    agent_view,
    all_service_names,
)


@dataclass
class Obs:
    phase: str       # "running" | "oom_killed" | "error"
    exit_code: int
    oom_killed: bool

    def __str__(self) -> str:
        return f"Obs(phase={self.phase!r}, exit_code={self.exit_code}, oom_killed={self.oom_killed})"


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    # ubuntu не в группе docker в текущей сессии → sudo
    # (группа добавлена, но требует нового логина; sudo без пароля настроен)
    if args and args[0] == "docker":
        args = ["sudo"] + args
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _prefetch_image(image: str) -> None:
    """Явный pull образа до первого docker run, чтобы не влиять на timeout."""
    check = _run(["docker", "image", "inspect", image])
    if check.returncode != 0:
        print(f"  [pull] {image} ...", flush=True)
        subprocess.run(["sudo", "docker", "pull", image], check=True, timeout=300)


class World:
    def apply(self, spec: dict) -> Obs:
        """
        Запускает сервис в Docker с заданным лимитом памяти.
        Возвращает Obs из docker inspect — один spec, один Obs.
        Агент передаёт spec; World сам находит cmd в _SERVICES (агент cmd не видит).
        """
        svc_name = spec["service"]
        mem_bucket = spec["mem_bucket"]

        if svc_name not in _SERVICES:
            raise ValueError(f"Unknown service: {svc_name!r}")
        if mem_bucket not in MEM_BUCKETS:
            raise ValueError(f"Bad mem_bucket={mem_bucket}, allowed: {MEM_BUCKETS}")

        svc = _SERVICES[svc_name]
        container = f"devops-{svc_name}-{mem_bucket}"
        mem_flag = f"{mem_bucket}m"

        # Гарантируем чистое состояние
        _run(["docker", "rm", "-f", container])

        try:
            _run(
                [
                    "docker", "run",
                    "--name", container,
                    "--memory", mem_flag,
                    "--memory-swap", mem_flag,  # swap=0 → OOM честный
                    svc["image"],
                ] + svc["cmd"],
                timeout=60,
            )

            result = _run(
                ["docker", "inspect", container, "--format", "{{json .State}}"]
            )

            try:
                state = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                # inspect вернул пусто: контейнер не создался или docker упал
                return Obs(phase="error", exit_code=-1, oom_killed=False)

            oom_killed = bool(state.get("OOMKilled", False))
            exit_code = int(state.get("ExitCode", -1))

            if oom_killed:
                phase = "oom_killed"
            elif exit_code == 0:
                phase = "running"
            else:
                phase = "error"

            return Obs(phase=phase, exit_code=exit_code, oom_killed=oom_killed)

        except subprocess.TimeoutExpired:
            return Obs(phase="error", exit_code=-1, oom_killed=False)

        finally:
            # rm -f останавливает и удаляет контейнер даже если он ещё работает
            # (нужно при TimeoutExpired — docker run убивает клиент, но не контейнер)
            _run(["docker", "rm", "-f", container])


def _build_selftest_cases() -> list[tuple[str, int, bool]]:
    """
    Строит тест-кейсы из _GROUND_TRUTH (DRY): для каждого сервиса
    берём бакет ниже порога (ожидаем OOM) и сам порог (ожидаем running).
    """
    cases = []
    for svc_name, truth in _GROUND_TRUTH.items():
        min_safe = truth["min_safe_bucket"]
        idx = MEM_BUCKETS.index(min_safe)
        if idx > 0:
            cases.append((svc_name, MEM_BUCKETS[idx - 1], True))   # ниже порога → OOM
        cases.append((svc_name, min_safe, False))                   # на пороге → running
    return cases


def _selftest() -> None:
    world = World()
    errors = []

    # Пре-пул всех образов — чтобы pull не считался в timeout docker run
    images = {_SERVICES[s]["image"] for s in all_service_names()}
    print("=== World self-test ===\n")
    for img in images:
        _prefetch_image(img)

    cases = _build_selftest_cases()
    print()
    for svc, bucket, expect_oom in cases:
        obs = world.apply({"service": svc, "mem_bucket": bucket})
        ok = obs.oom_killed == expect_oom
        status = "OK" if ok else "FAIL"
        label = f"{svc} @ {bucket} MiB"
        print(f"  [{status}] {label:20s}  {obs}")
        if not ok:
            errors.append(f"{label}: expected oom_killed={expect_oom}, got {obs}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"All {len(cases)} cases passed. Ground-truth holds.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.world --selftest")
