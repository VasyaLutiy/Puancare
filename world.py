"""Мини-мир для Эксперимента 0: симулированная мини-ОС.

Всё видимо агенту, КРОМЕ:
  - связей config -> process (какой файл является конфигом какого процесса)
  - самих правил динамики (их агент и должен выучить)

Ground-truth правила динамики (детерминированные):
  read(f)    -> ok всегда
  write(f)   -> err:permission если owner=root; иначе ok;
                если f — конфиг работающего процесса P, P умирает (corrupt)
  delete(f)  -> err:permission если owner=root; иначе файл удалён;
                если f — конфиг работающего P, P умирает
  chmod(f)   -> err:permission если owner=root; иначе exec_bit инвертируется
  exec(f)    -> err:permission если exec_bit=0;
                err:format если ftype != binary; иначе ok
  create(dir, ftype, exec_bit) -> ok, новый файл с owner=user  (инструмент do-вмешательств)
  kill(p)    -> ok если running, иначе err:not_running
  start(p)   -> err:already если running; err:no_config если конфиг удалён;
                иначе ok, процесс running

Ловушки для суеверий (корреляции в генераторе, не в динамике):
  - root-файлы лежат почти всегда в /etc и /bin, user-файлы в /home
    => пассивный майнер выучит "в /etc нельзя писать" вместо "root нельзя писать"
  - конфиги почти всегда в /etc
  Разбить суеверие можно только вмешательством: create в /etc даёт user-файл.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, asdict

OWNERS = ("root", "user")
FTYPES = ("text", "binary", "config")
DIRS = ("/etc", "/bin", "/home")
ACTIONS = ("read", "write", "delete", "chmod", "exec", "copy",
           "create", "kill", "start")


@dataclass
class File:
    name: str
    dir: str
    owner: str
    ftype: str
    exec_bit: int
    sealed: int = 0   # вторая скрытая ось (независимая от owner):
                      # опечатанный файл нельзя читать/копировать

    def visible(self):
        return asdict(self)


@dataclass
class Proc:
    name: str
    status: str          # running | dead
    config: str | None   # имя файла-конфига; СКРЫТО от агента
    config_ok: bool = True

    def visible(self):
        return {"name": self.name, "status": self.status}


class World:
    """Детерминированная среда. reset() пересоздаёт мир из seed."""

    def __init__(self, seed=0, n_files=8, quarantine=(), hidden=()):
        # quarantine: набор запрещённых при генерации комбинаций-предикатов,
        # например {("owner", "user"), ("dir", "/etc")} — держим для экзамена.
        # hidden: атрибуты файлов, невидимые агенту (динамика ими пользуется).
        self.seed = seed
        self.n_files = n_files
        self.quarantine = [frozenset(q) for q in quarantine]
        self.hidden = set(hidden)
        self.reset()

    # ---------- генерация ----------

    def _combo_banned(self, f: File):
        preds = {("owner", f.owner), ("dir", f.dir),
                 ("ftype", f.ftype), ("exec_bit", f.exec_bit)}
        return any(q <= preds for q in self.quarantine)

    def _gen_file(self, rng, name):
        for _ in range(100):
            owner = rng.choice(OWNERS)
            # корреляция-ловушка: root живёт в /etc|/bin, user — в /home (90%)
            if rng.random() < 0.9:
                dir_ = rng.choice(("/etc", "/bin")) if owner == "root" else "/home"
            else:
                dir_ = rng.choice(DIRS)
            ftype = rng.choice(FTYPES)
            # корреляция-ловушка: конфиги тянет в /etc (90%)
            if ftype == "config" and rng.random() < 0.9:
                dir_ = "/etc"
            exec_bit = int(ftype == "binary" and rng.random() < 0.7)
            f = File(name, dir_, owner, ftype, exec_bit,
                     sealed=int(rng.random() < 0.4))
            if not self._combo_banned(f):
                return f
        raise RuntimeError("не могу сгенерировать файл вне карантина")

    def reset(self):
        rng = random.Random(self.seed)
        self.files = {}
        for i in range(self.n_files):
            f = self._gen_file(rng, f"f{i}")
            self.files[f.name] = f
        # гарантируем хотя бы 2 конфига
        configs = [f for f in self.files.values() if f.ftype == "config"]
        while len(configs) < 2:
            victim = rng.choice([f for f in self.files.values() if f.ftype != "config"])
            victim.ftype, victim.exec_bit = "config", 0
            if victim.dir != "/etc" and rng.random() < 0.9:
                old_dir = victim.dir
                victim.dir = "/etc"
                if self._combo_banned(victim):   # карантин сильнее корреляции
                    victim.dir = old_dir
            configs = [f for f in self.files.values() if f.ftype == "config"]
        # хотя бы один конфиг должен быть user-овским, иначе смысл
        # "порча конфига убивает процесс" ненаблюдаем (root-файлы неизменяемы)
        if not any(f.owner == "user" for f in configs):
            lucky = rng.choice(configs)
            lucky.owner = "user"
            if self._combo_banned(lucky):
                lucky.dir = "/home"
        # обе оси печати должны присутствовать, иначе переменная невыводима
        if len({f.sealed for f in self.files.values()}) == 1:
            rng.choice(sorted(self.files.values(),
                              key=lambda f: f.name)).sealed ^= 1
        self.procs = {}
        for i, cfg in enumerate(configs[:3]):
            p = Proc(f"p{i}", "running", cfg.name)
            self.procs[p.name] = p
        self._create_counter = 0
        return self.observe()

    # ---------- наблюдение ----------

    def observe(self):
        return {
            "files": {n: {k: v for k, v in f.visible().items()
                          if k not in self.hidden}
                      for n, f in sorted(self.files.items())},
            "procs": {n: p.visible() for n, p in sorted(self.procs.items())},
        }

    # ---------- динамика ----------

    def step(self, action, target=None, **kw):
        """Возвращает transition-запись: (obs_before, action, result, effects, obs_after)."""
        before = self.observe()
        result, effects = self._apply(action, target, kw)
        after = self.observe()
        return {
            "obs": before, "action": action, "target": target, "args": kw,
            "result": result, "effects": effects, "obs_after": after,
        }

    def _crash_dependents(self, fname, effects):
        for p in self.procs.values():
            if p.config == fname and p.status == "running":
                p.status = "dead"
                effects.append(("proc_died", p.name))

    def _apply(self, action, target, kw):
        effects = []
        if action in ("read", "write", "delete", "chmod", "exec", "copy"):
            f = self.files.get(target)
            if f is None:
                return "err:no_such_file", effects

            if action in ("read", "copy"):
                if f.sealed:
                    return "err:sealed", effects
                return "ok", effects

            if action == "write":
                if f.owner == "root":
                    return "err:permission", effects
                self._crash_dependents(f.name, effects)
                for p in self.procs.values():
                    if p.config == f.name:
                        p.config_ok = False
                return "ok", effects

            if action == "delete":
                if f.owner == "root":
                    return "err:permission", effects
                self._crash_dependents(f.name, effects)
                del self.files[f.name]
                effects.append(("file_deleted", f.name))
                return "ok", effects

            if action == "chmod":
                if f.owner == "root":
                    return "err:permission", effects
                f.exec_bit ^= 1
                effects.append(("exec_bit", f.name, f.exec_bit))
                return "ok", effects

            if action == "exec":
                if not f.exec_bit:
                    return "err:permission", effects
                if f.ftype != "binary":
                    return "err:format", effects
                return "ok", effects

        if action == "create":
            dir_ = kw.get("dir", "/home")
            ftype = kw.get("ftype", "text")
            exec_bit = int(kw.get("exec_bit", 0))
            if dir_ not in DIRS or ftype not in FTYPES:
                return "err:bad_args", effects
            name = f"new{self._create_counter}"
            self._create_counter += 1
            self.files[name] = File(name, dir_, "user", ftype, exec_bit)
            effects.append(("file_created", name))
            return "ok", effects

        if action in ("kill", "start"):
            p = self.procs.get(target)
            if p is None:
                return "err:no_such_proc", effects
            if action == "kill":
                if p.status != "running":
                    return "err:not_running", effects
                p.status = "dead"
                effects.append(("proc_died", p.name))
                return "ok", effects
            if action == "start":
                if p.status == "running":
                    return "err:already", effects
                if p.config not in self.files or not p.config_ok:
                    return "err:no_config", effects
                p.status = "running"
                effects.append(("proc_started", p.name))
                return "ok", effects

        return "err:bad_action", effects

    # ---------- перечисление доступных действий ----------

    def action_space(self):
        acts = []
        for fname in self.files:
            for a in ("read", "write", "delete", "chmod", "exec", "copy"):
                acts.append((a, fname, {}))
        for pname in self.procs:
            acts.append(("kill", pname, {}))
            acts.append(("start", pname, {}))
        for dir_, ftype, xb in itertools.product(DIRS, FTYPES, (0, 1)):
            acts.append(("create", None, {"dir": dir_, "ftype": ftype, "exec_bit": xb}))
        return acts
