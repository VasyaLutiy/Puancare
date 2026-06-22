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

from devops_agent.services import MEM_BUCKETS, SERVICES, _GROUND_TRUTH


@dataclass
class Obs:
    phase: str       # "running" | "oom_killed" | "error"
    exit_code: int
    oom_killed: bool

    def __str__(self) -> str:
        return f"Obs(phase={self.phase!r}, exit_code={self.exit_code}, oom_killed={self.oom_killed})"


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    # ubuntu не в группе docker в текущей сессии — используем sudo
    if args and args[0] == "docker":
        args = ["sudo"] + args
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


class World:
    def apply(self, spec: dict) -> Obs:
        """
        Запускает сервис в Docker с заданным лимитом памяти.
        Возвращает Obs из docker inspect — один spec, один Obs.
        """
        svc_name = spec["service"]
        mem_bucket = spec["mem_bucket"]

        if svc_name not in SERVICES:
            raise ValueError(f"Unknown service: {svc_name!r}")
        if mem_bucket not in MEM_BUCKETS:
            raise ValueError(f"Bad mem_bucket={mem_bucket}, allowed: {MEM_BUCKETS}")

        svc = SERVICES[svc_name]
        container = f"devops-{svc_name}-{mem_bucket}"
        mem_flag = f"{mem_bucket}m"

        # Гарантируем чистое состояние перед запуском
        _run(["docker", "rm", "-f", container])

        try:
            _run(
                [
                    "docker", "run",
                    "--name", container,
                    "--memory", mem_flag,
                    "--memory-swap", mem_flag,  # swap = 0, чтобы OOM был честным
                    svc["image"],
                ] + svc["cmd"],
                timeout=30,
            )

            result = _run(
                ["docker", "inspect", container, "--format", "{{json .State}}"]
            )
            state = json.loads(result.stdout.strip())

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
            _run(["docker", "rm", "-f", container])


def _selftest():
    world = World()
    errors = []

    cases = [
        # (service, bucket, expect_oom)
        ("svc_a", 256,  True),
        ("svc_a", 512,  False),
        ("svc_b", 64,   True),
        ("svc_b", 128,  False),
        ("svc_c", 512,  True),
        ("svc_c", 1024, False),
    ]

    print("=== World self-test ===\n")
    for svc, bucket, expect_oom in cases:
        spec = {"service": svc, "mem_bucket": bucket}
        obs = world.apply(spec)
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
        print("All cases passed. Ground-truth holds.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.world --selftest")
