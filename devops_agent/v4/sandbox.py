"""
sandbox.py — богатая stateful многоконтейнерная песочница (A-0).

Расширяет одноразовый мир v3 до 4 верифицируемых измерений на РЕАЛЬНОМ docker:
  • mem        — OOM при --memory < footprint            (ordered_monotone)
  • config     — exit 3 при CONFIG != good_config         (categorical)
  • pool       — exit 4 (<lo) / exit 5 (>hi)              (non-monotone bounded → REFUTATION)
  • dependency — здоровая сущность СЛУШАЕТ порт; зависимая КОННЕКТИТСЯ к ней по имени
                 в user-network; не дозвонилась → exit 6 (порядок-в-РЕАЛЬНОСТИ)

Stateful: здоровая сущность остаётся поднята (sleep) и слушает порт — поэтому зависимые могут
реально к ней подключиться, и ПОРЯДОК провижинга влияет на исход. truth (footprint/lo/hi/good/deps)
скрыт от агента — он узнаёт измерения только по СИМПТОМАМ (как в v3).
"""

import subprocess
import sys
import time
from dataclasses import dataclass

_IMAGE = "python:3.11-slim"
_PORT = 8000

# Универсальное поведение сущности (внутри контейнера). Все измерения в одном cmd,
# активны только те, чьи env заданы (footprint/good/hi/deps).
_CMD = (
    "import os,sys,socket,time\n"
    "fp=int(os.environ.get('FOOTPRINT','0'))\n"
    "if fp:\n"
    " x=bytearray(fp*1024*1024)\n"
    " for i in range(0,len(x),4096):x[i]=1\n"
    "cfg=os.environ.get('CONFIG','');good=os.environ.get('GOOD_CONFIG','')\n"
    "if good and cfg!=good:sys.exit(3)\n"
    "hi=int(os.environ.get('POOL_HI','0'))\n"
    "if hi:\n"
    " p=int(os.environ.get('POOL','0'));lo=int(os.environ.get('POOL_LO','0'))\n"
    " if p<lo:sys.exit(4)\n"
    " if p>hi:sys.exit(5)\n"
    "deps=[d for d in os.environ.get('DEPS','').split(',') if d]\n"
    "dp=int(os.environ.get('DEP_PORT','8000'))\n"
    "for d in deps:\n"
    " try:\n"
    "  s=socket.create_connection((d,dp),timeout=3);s.close()\n"
    " except OSError:\n"
    "  sys.exit(6)\n"
    "port=int(os.environ.get('PORT','8000'))\n"
    "srv=socket.socket();srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
    "srv.bind(('0.0.0.0',port));srv.listen(8)\n"
    "time.sleep(int(os.environ.get('TTL','30')))\n"
)


@dataclass
class Obs:
    phase: str       # running | oom_killed | bad_config | pool_under | pool_over | dep_unmet | error
    exit_code: int
    oom_killed: bool

    def __str__(self) -> str:
        return f"Obs(phase={self.phase!r}, exit_code={self.exit_code}, oom={self.oom_killed})"


def _run(args, timeout=30):
    if args and args[0] == "docker":
        args = ["sudo"] + args
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def classify(exit_code: int, oom: bool) -> str:
    if oom:
        return "oom_killed"
    return {3: "bad_config", 4: "pool_under", 5: "pool_over", 6: "dep_unmet"}.get(exit_code, "error")


class Sandbox:
    """truth: {name: {footprint?, good_config?, pool_lo?, pool_hi?, deps?:[...]}}. Агент truth не видит."""

    def __init__(self, truth: dict, network: str = "v4net", image: str = _IMAGE):
        self.truth = truth
        self.net = network
        self.image = image

    def _c(self, name: str) -> str:
        return f"v4-{name}"

    def reset(self) -> None:
        for name in self.truth:
            _run(["docker", "rm", "-f", self._c(name)])
        _run(["docker", "network", "rm", self.net])
        _run(["docker", "network", "create", self.net])

    def prefetch(self) -> None:
        if _run(["docker", "image", "inspect", self.image]).returncode != 0:
            print(f"  [pull] {self.image} ...", flush=True)
            _run(["docker", "pull", self.image], timeout=300)

    def provision(self, name: str, mem=None, pool=None, config=None) -> Obs:
        """Поднять контейнер сущности с knob'ами агента; вернуть Obs. Здоровая остаётся слушать."""
        t = self.truth[name]
        c = self._c(name)
        _run(["docker", "rm", "-f", c])
        env = {
            "FOOTPRINT": str(t.get("footprint", 0)),
            "GOOD_CONFIG": t.get("good_config", ""),
            "CONFIG": config if config is not None else "",
            "POOL": str(pool if pool is not None else 0),
            "POOL_LO": str(t.get("pool_lo", 0)),
            "POOL_HI": str(t.get("pool_hi", 0)),
            "DEPS": ",".join(self._c(d) for d in t.get("deps", [])),
            "DEP_PORT": str(_PORT), "PORT": str(_PORT), "TTL": "30",
        }
        args = ["docker", "run", "-d", "--name", c, "--network", self.net]
        if mem is not None:
            args += ["--memory", f"{mem}m", "--memory-swap", f"{mem}m"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        args += [self.image, "python", "-c", _CMD]
        if _run(args).returncode != 0:
            return Obs("error", -1, False)
        # poll: exited (нездоров) vs стабильно running (здоров, слушает)
        for _ in range(9):
            time.sleep(0.7)
            ins = _run(["docker", "inspect", c, "--format",
                        "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}"])
            parts = ins.stdout.split()
            if len(parts) < 3:
                continue
            status, code, oom = parts[0], int(parts[1]), parts[2] == "true"
            if status == "exited":
                return Obs(classify(code, oom), code, oom)
        return Obs("running", 0, False)

    def teardown(self) -> None:
        for name in self.truth:
            _run(["docker", "rm", "-f", self._c(name)])
        _run(["docker", "network", "rm", self.net])


# ---------------------------------------------------------------------------
# A-0 selftest: доказать 4 ground-truth на реальном docker (вкл. сетевую зависимость).
# ---------------------------------------------------------------------------

_TRUTH = {
    "e_mem":  {"footprint": 350},                          # OOM ниже ~512
    "e_cfg":  {"good_config": "good"},                     # exit3 если не good
    "e_pool": {"pool_lo": 100, "pool_hi": 200},            # окно [100,200]
    "e_dep":  {"deps": ["e_mem"]},                         # должен дозвониться до e_mem
}


def _selftest() -> None:
    print("=== A-0 sandbox: 4 ground-truth на реальном docker ===\n")
    errors = []
    sb = Sandbox(_TRUTH)
    sb.prefetch()
    sb.reset()
    try:
        checks = [
            # mem (ordered_monotone)
            ("e_mem @64m",      lambda: sb.provision("e_mem", mem=64),            "oom_killed"),
            ("e_mem @512m",     lambda: sb.provision("e_mem", mem=512),           "running"),
            # config (categorical)
            ("e_cfg bad",       lambda: sb.provision("e_cfg", mem=256, config="bad"),  "bad_config"),
            ("e_cfg good",      lambda: sb.provision("e_cfg", mem=256, config="good"), "running"),
            # pool (non-monotone)
            ("e_pool=50",       lambda: sb.provision("e_pool", mem=256, pool=50),  "pool_under"),
            ("e_pool=300",      lambda: sb.provision("e_pool", mem=256, pool=300), "pool_over"),
            ("e_pool=150",      lambda: sb.provision("e_pool", mem=256, pool=150), "running"),
        ]
        for label, act, expect in checks:
            obs = act()
            ok = obs.phase == expect
            print(f"  [{'OK' if ok else 'FAIL'}] {label:14s} → {obs}  (ждали {expect})")
            if not ok:
                errors.append(f"{label}: ждали {expect}, got {obs.phase}")

        # dependency (порядок-в-реальности): без e_mem → dep_unmet; с поднятым e_mem → running
        print("  -- dependency (реальная docker-сеть) --")
        sb.provision  # noqa (readability)
        _run(["docker", "rm", "-f", sb._c("e_mem")])           # убедимся, что e_mem НЕ поднят
        obs_no = sb.provision("e_dep", mem=256)
        print(f"  [{'OK' if obs_no.phase=='dep_unmet' else 'FAIL'}] e_dep без e_mem → {obs_no}  (ждали dep_unmet)")
        if obs_no.phase != "dep_unmet":
            errors.append(f"e_dep без зависимости: ждали dep_unmet, got {obs_no.phase}")

        up = sb.provision("e_mem", mem=512)                    # поднимаем зависимость (слушает)
        if up.phase != "running":
            errors.append(f"e_mem не поднялся для dep-теста: {up.phase}")
        obs_yes = sb.provision("e_dep", mem=256)               # теперь дозвонится
        print(f"  [{'OK' if obs_yes.phase=='running' else 'FAIL'}] e_dep с поднятым e_mem → {obs_yes}  (ждали running)")
        if obs_yes.phase != "running":
            errors.append(f"e_dep с зависимостью: ждали running, got {obs_yes.phase}")
    finally:
        sb.teardown()

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("A-0 sandbox OK — mem/config/pool/dependency дают разные ground-truth на реальном docker; "
          "межконтейнерная зависимость через сеть работает, порядок провижинга влияет на исход.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v4.sandbox --selftest")
