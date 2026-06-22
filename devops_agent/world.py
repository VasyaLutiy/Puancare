"""
Sandbox-Gym: World.apply(spec) -> Obs

spec  = {"service": str, "mem_bucket": int, "config": str}
         config — опционален, default "good"
Obs   = {phase, exit_code, oom_killed}
         phase: "running" | "oom_killed" | "unhealthy" | "error"

Маппинг exit_code (детерминировано, без NLP):
  OOMKilled=True           → oom_killed
  exit_code=0              → running
  exit_code=3, не OOM      → unhealthy  (sentinel svc_e)
  иначе                    → error
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
    phase: str       # "running" | "oom_killed" | "unhealthy" | "error"
    exit_code: int
    oom_killed: bool

    def __str__(self) -> str:
        return f"Obs(phase={self.phase!r}, exit_code={self.exit_code}, oom_killed={self.oom_killed})"


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    # ubuntu не в группе docker в текущей сессии → sudo
    if args and args[0] == "docker":
        args = ["sudo"] + args
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _prefetch_image(image: str) -> None:
    check = _run(["docker", "image", "inspect", image])
    if check.returncode != 0:
        print(f"  [pull] {image} ...", flush=True)
        subprocess.run(["sudo", "docker", "pull", image], check=True, timeout=300)


class World:
    def apply(self, spec: dict) -> Obs:
        """
        Запускает сервис в Docker с заданным лимитом памяти и config.
        Возвращает Obs из docker inspect — один spec, один Obs.
        Агент передаёт spec; World сам находит cmd в _SERVICES (агент cmd не видит).
        """
        svc_name  = spec["service"]
        mem_bucket = spec["mem_bucket"]
        config    = spec.get("config", "good")  # knob конфига, default "good"

        if svc_name not in _SERVICES:
            raise ValueError(f"Unknown service: {svc_name!r}")
        if mem_bucket not in MEM_BUCKETS:
            raise ValueError(f"Bad mem_bucket={mem_bucket}, allowed: {MEM_BUCKETS}")

        svc = _SERVICES[svc_name]
        container = f"devops-{svc_name}-{mem_bucket}"
        mem_flag  = f"{mem_bucket}m"

        _run(["docker", "rm", "-f", container])

        try:
            _run(
                [
                    "docker", "run",
                    "--name", container,
                    "--memory", mem_flag,
                    "--memory-swap", mem_flag,   # swap=0 → OOM честный
                    "--env", f"CONFIG={config}", # knob для svc_e и других с config
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
                return Obs(phase="error", exit_code=-1, oom_killed=False)

            oom_killed = bool(state.get("OOMKilled", False))
            exit_code  = int(state.get("ExitCode", -1))

            if oom_killed:
                phase = "oom_killed"
            elif exit_code == 0:
                phase = "running"
            elif exit_code == 3 and not oom_killed:
                phase = "unhealthy"   # sentinel: плохой config (svc_e exit(3))
            else:
                phase = "error"

            return Obs(phase=phase, exit_code=exit_code, oom_killed=oom_killed)

        except subprocess.TimeoutExpired:
            return Obs(phase="error", exit_code=-1, oom_killed=False)

        finally:
            # rm -f останавливает и удаляет даже работающий контейнер
            _run(["docker", "rm", "-f", container])


def _build_selftest_cases() -> list[tuple]:
    """
    Стандартные кейсы из _GROUND_TRUTH (memory dimension).
    Для svc_e: используем good config — проверяем только memory boundary.
    Дополнительные config-кейсы добавлены вручную ниже.
    """
    cases = []
    for svc_name, truth in _GROUND_TRUTH.items():
        min_safe = truth["min_safe_bucket"]
        idx = MEM_BUCKETS.index(min_safe)
        config = truth.get("good_config", "good")
        if idx > 0:
            cases.append((svc_name, MEM_BUCKETS[idx - 1], config, "oom_killed"))
        cases.append((svc_name, min_safe, config, "running"))
    return cases


def _selftest() -> None:
    world = World()
    errors = []

    images = {_SERVICES[s]["image"] for s in all_service_names()}
    print("=== World self-test ===\n")
    for img in images:
        _prefetch_image(img)

    # Стандартные memory-кейсы
    cases = _build_selftest_cases()

    # Дополнительные кейсы для svc_e (config dimension)
    extra = [
        ("svc_e", 512, "bad",  "unhealthy"),  # memory ok, bad config → unhealthy
        ("svc_e", 512, "good", "running"),     # memory ok, good config → running
        ("svc_e", 64,  "good", "oom_killed"),  # memory bad → oom (config не важен)
    ]

    print()
    all_cases = cases + extra
    for row in all_cases:
        svc, bucket, config, expected_phase = row
        spec = {"service": svc, "mem_bucket": bucket, "config": config}
        obs = world.apply(spec)
        ok = obs.phase == expected_phase
        status = "OK" if ok else "FAIL"
        label = f"{svc} @ {bucket} MiB config={config!r}"
        print(f"  [{status}] {label:35s}  {obs}")
        if not ok:
            errors.append(f"{label}: expected phase={expected_phase!r}, got {obs}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"All {len(all_cases)} cases passed. Ground-truth holds.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.world --selftest")
