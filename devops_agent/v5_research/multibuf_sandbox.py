"""
multibuf_sandbox.py — среда с N рычагов ОДИНАКОВОЙ структуры (ordered_monotone), но
РАЗНЫМИ симптомами (low_buf0, low_buf1, ...). Для E1: амортизирует ли агент по СТРУКТУРЕ
(kind) или только по строке симптома. Каждый buf: значение < порога → exit 10+i → low_<buf>.
Пороги (truth) скрыты от агента. Реальный docker.
"""

import time

from devops_agent.v4.sandbox import _IMAGE, Obs, _run

_CMD = (
    "import os,sys,time\n"
    "bufs=[b for b in os.environ.get('BUFS','').split(',') if b]\n"
    "for i,b in enumerate(bufs):\n"
    " if int(os.environ.get('V_'+b,'0'))<int(os.environ.get('T_'+b,'0')):sys.exit(10+i)\n"
    "time.sleep(30)\n"   # > окна опроса: здоровый СТОИТ running, не выходит с кодом 0
)


class MultiBufSandbox:
    def __init__(self, bufs: dict, network: str = "v5e1", image: str = _IMAGE):
        self.bufs = bufs              # {name: threshold} — truth, скрыт от агента
        self.net = network
        self.image = image

    def _c(self, name):
        return f"v5b-{name}"

    def prefetch(self):
        if _run(["docker", "image", "inspect", self.image]).returncode != 0:
            _run(["docker", "pull", self.image], timeout=300)

    def reset(self):
        _run(["docker", "rm", "-f", self._c("app")])
        _run(["docker", "network", "rm", self.net])
        _run(["docker", "network", "create", self.net])

    def provision(self, name, knobs=None, **kw) -> Obs:
        knobs = {**(knobs or {}), **kw}
        c = self._c(name)
        _run(["docker", "rm", "-f", c])
        names = list(self.bufs)
        env = {"BUFS": ",".join(names)}
        for b, th in self.bufs.items():
            env["T_" + b] = str(th)
        for k, v in knobs.items():
            if k in self.bufs:
                env["V_" + k] = str(v)
        args = ["docker", "run", "-d", "--name", c, "--network", self.net]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        args += [self.image, "python", "-c", _CMD]
        if _run(args).returncode != 0:
            return Obs("error", -1, False)
        for _ in range(9):
            time.sleep(0.7)
            ins = _run(["docker", "inspect", c, "--format", "{{.State.Status}} {{.State.ExitCode}}"])
            p = ins.stdout.split()
            if len(p) < 2:
                continue
            if p[0] == "exited":
                code = int(p[1])
                if 10 <= code < 10 + len(names):
                    return Obs(f"low_{names[code - 10]}", code, False)
                return Obs("error", code, False)
        return Obs("running", 0, False)

    def teardown(self):
        _run(["docker", "rm", "-f", self._c("app")])
        _run(["docker", "network", "rm", self.net])
