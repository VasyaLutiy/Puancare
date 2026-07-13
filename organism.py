"""Мир-агностичный организм: логика exp5 + реляционная абдукция (exp7)
+ факторизованное изобретение понятий (exp8, фаза 1.5).

Урок фазы 1 (камешки): в кластеризации сидела необъявленная аксиома
"скрытая переменная одна" — при двух независимых осях (чётность,
простота) общий симптом по одной оси склеивал объекты, различные по
другой, и рождались классы-химеры. Лечение: симптомы сначала
группируются по взаимной корреляции на со-наблюдённых объектах
(функциональная связь в обе стороны), затем по одной латентной
переменной на группу (class0, class1, ...).

Мирозависимое инжектится: object_actions, ctx_fn, truth_fn, entities_fn.
Ядро памяти (memory.Memory) импортируется без изменений.
"""

import random
from collections import defaultdict

from memory import EP_COST, RULE_BASE, RULE_COND, Memory

SLEEP_EVERY = 100
INVENT_AT = 5
MIN_CO = 3           # минимум со-наблюдений для вывода о корреляции симптомов
FUNC_THRESHOLD = 0.65  # A6: 65% доминирование достаточно — мир динамичен
FRESHNESS = 5        # A6: устаревшее наблюдение в динамическом мире = снова граница
CLASS_PENALTY = 10.0 # A6: штраф за каждую лишнюю скрытую переменную — экономия онтологии
CONSOL_WINDOW = FRESHNESS  # E1: окно когерентности при консолидации; None = без окна (E2)


def cluster_signatures(sigs):
    clusters = []
    for obj, s in sorted(sigs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for msig, objs in clusters:
            shared = set(s) & set(msig)
            if shared and all(s[a] == msig[a] for a in shared):
                msig.update(s)
                objs.append(obj)
                break
        else:
            clusters.append((dict(s), [obj]))
    return clusters


def _functional(pairs):
    f = {}
    for x, y in pairs:
        if f.setdefault(x, y) != y:
            return False
    return True


def _functional_prob(pairs, threshold=FUNC_THRESHOLD):
    """A6: x → y доминирует в порог% случаев (терпим к динамическому шуму)."""
    by_x = {}
    for x, y in pairs:
        by_x.setdefault(x, {})
        by_x[x][y] = by_x[x].get(y, 0) + 1
    for counts in by_x.values():
        total = sum(counts.values())
        if max(counts.values()) / total < threshold:
            return False
    return True


class Organism:
    def __init__(self, curious, object_actions, ctx_fn, truth_fn=None,
                 entities_fn=None, seed=0):
        self.curious = curious
        self.object_actions = set(object_actions)
        self.ctx_fn = ctx_fn
        self.truth_fn = truth_fn or (lambda w, t: None)
        self.entities_fn = entities_fn or (lambda obs: {})
        self.rng = random.Random(seed)
        self.mem = Memory()
        self.records = []
        self.sig = defaultdict(dict)
        self.sig_hist = defaultdict(lambda: defaultdict(list))  # E1: вся линия (результат, шаг)
        self.groups = []          # [{"attr","actions","clusters"}] — теории
        self.problem_actions = set()
        self.class_relevant = {}  # action -> какие class-атрибуты в правилах
        self.tried = set()
        self.R = set()            # связи: (obj, сущность, её живое состояние)
        self.steps = 0
        self._dyn = False         # A6: обнаружили что мир меняется под ногами

    # ---- понятия ----

    def obj_classes(self, obj):
        """Значения всех изобретённых переменных для объекта (если выводимы)."""
        out = {}
        for g in self.groups:
            s = {a: r for a, (r, _t) in self.sig.get(obj, {}).items()
                 if a in g["actions"]}
            for k, (msig, _) in enumerate(g["clusters"]):
                shared = set(s) & set(msig)
                if shared and all(s[a] == msig[a] for a in shared):
                    out[g["attr"]] = f"c{k}"
                    break
        return out

    def obj_classes_at(self, obj, t):
        """E1: классы объекта по наблюдениям, синхронным моменту t.

        Эпизод — свидетельство о паре (объект, t): метку класса ему дают
        только наблюдения из окна |шаг - t| <= CONSOL_WINDOW, ближайшие к t.
        В статическом мире (и при CONSOL_WINDOW=None) тождественно
        obj_classes — время-локальность обязана быть там невидимой."""
        if not self._dyn or CONSOL_WINDOW is None or t is None:
            return self.obj_classes(obj)
        out = {}
        hist = self.sig_hist.get(obj)
        if not hist:
            return out
        for g in self.groups:
            s = {}
            for a in g["actions"]:
                obs = hist.get(a)
                if not obs:
                    continue
                r, dt = min(((r, abs(st - t)) for r, st in obs),
                            key=lambda x: x[1])
                if dt <= CONSOL_WINDOW:
                    s[a] = r
            for k, (msig, _) in enumerate(g["clusters"]):
                shared = set(s) & set(msig)
                if shared and all(s[a] == msig[a] for a in shared):
                    out[g["attr"]] = f"c{k}"
                    break
        return out

    def infer(self, action, result):
        """(атрибут, значение) скрытой переменной по исходу зонда:
        сначала чтение правил в обратную сторону, потом сигнатуры."""
        cands = {(att, v) for r in self.mem.rules
                 if r["action"] == action and r["outcome"][0] == result
                 for att, v in r["conds"] if att.startswith("class")}
        if len(cands) == 1:
            return cands.pop()
        # метод исключения: значения переменной, чьи правила противоречат
        # наблюдению, отбрасываются; осталось одно — оно и есть ответ
        for g in self.groups:
            rel = [r for r in self.mem.rules if r["action"] == action
                   and any(att == g["attr"] for att, _ in r["conds"])]
            if not rel:
                continue
            compat = [v for v in (f"c{k}" for k in range(len(g["clusters"])))
                      if not any(r["outcome"][0] != result for r in rel
                                 if (g["attr"], v) in r["conds"])]
            if len(compat) == 1:
                return (g["attr"], compat[0])
        for g in self.groups:
            if action in g["actions"]:
                for k, (ms, _) in enumerate(g["clusters"]):
                    if ms.get(action) == result:
                        return (g["attr"], f"c{k}")
        return None

    def contextualize(self, ctx0, obj, ents, at=None):
        ctx = ctx0
        if obj:
            # E1: для исторических эпизодов (at задан) — время-локальные классы
            oc = (self.obj_classes_at(obj, at) if at is not None
                  else self.obj_classes(obj))
            if oc:
                ctx = ctx | set(oc.items())
        if self.R and obj is not None:
            live = any(o == obj and ents.get(e) == st
                       for (o, e, st) in self.R)
            ctx = ctx | {("linked_live", int(live))}
        return ctx

    # ---- драйв ----

    def is_frontier(self, ctx, obj, action):
        # симптом-сирота: проблемное действие вне всех групп. Каким
        # прибором оно является — решают только со-наблюдения: зондируй
        # его на объектах, уже измеренных другими симптомами
        if (obj is not None and action in self.problem_actions
                and all(action not in g["actions"] for g in self.groups)
                and action not in self.sig.get(obj, {})
                and self.sig.get(obj)):
            return True
        # теория ссылается на переменную, не измеренную у объекта,
        # но измеримую (есть незондированный симптом её группы)
        if obj is not None:
            rel = self.class_relevant.get(action, set())
            if rel:
                oc = self.obj_classes(obj)
                for g in self.groups:
                    if (g["attr"] in rel and g["attr"] not in oc
                            and any(a not in self.sig.get(obj, {})
                                    for a in g["actions"])):
                        return True
        # A6: в динамическом мире устаревшее наблюдение нужно освежить
        if (self._dyn and obj is not None
                and action in self.sig.get(obj, {})
                and self.steps - self.sig[obj][action][1] > FRESHNESS):
            return True
        if (ctx, action) in self.tried:
            return False
        return self.mem.predict(ctx, action) is None

    # ---- петля ----

    def act(self, w, ep):
        obs = w.observe()
        ents = {(ep, n): s for n, s in self.entities_fn(obs).items()}
        space = w.action_space()
        frontier = []
        for a, t, kw in space:
            obj = (ep, t) if a in self.object_actions else None
            ctx = self.contextualize(self.ctx_fn(obs, a, t, kw), obj, ents)
            if self.is_frontier(ctx, obj, a):
                frontier.append((a, t, kw))
        pool = frontier if (self.curious and frontier) else space
        a, t, kw = self.rng.choice(pool)

        obj = (ep, t) if a in self.object_actions else None
        ctx0 = self.ctx_fn(obs, a, t, kw)
        ctx_now = self.contextualize(ctx0, obj, ents)
        truth = self.truth_fn(w, t) if obj else None
        tr = w.step(a, t, **kw)
        outcome = (tr["result"], tuple(sorted({e[0] for e in tr["effects"]})))
        self.records.append({"obj": obj, "ctx": ctx0, "action": a,
                             "outcome": outcome, "truth": truth,
                             "ents": ents, "effects": tr["effects"],
                             "step": self.steps})  # E1: эпизод знает своё время
        self.tried.add((ctx_now, a))
        if obj:
            prev = self.sig[obj].get(a)
            if prev is not None and prev[0] != outcome[0]:
                self._dyn = True   # A6: то же действие — другой результат → мир меняется
            self.sig[obj][a] = (outcome[0], self.steps)  # A6: (результат, шаг)
            self.sig_hist[obj][a].append((outcome[0], self.steps))
        self.steps += 1
        if self.steps % SLEEP_EVERY == 0:
            self.sleep()
        return len(frontier)

    # ---- сон ----

    def _relabel(self):
        mem = Memory()
        for r in self.records:
            mem.episodes[(self.contextualize(r["ctx"], r["obj"], r["ents"],
                                             at=r.get("step")),
                          r["action"], r["outcome"])] += 1
        mem.consolidate()
        return mem

    def _dl(self, mem):
        bad = 0
        for (ctx, a, o) in mem.episodes:
            r = mem.predict(ctx, a)
            if r is None or r["outcome"] != o:
                bad += 1
        n_class_vars = len({att for r in mem.rules
                            for att, _ in r["conds"] if att.startswith("class")})
        return (sum(RULE_BASE + RULE_COND * len(r["conds"]) for r in mem.rules)
                + bad * EP_COST
                + n_class_vars * CLASS_PENALTY)  # A6: меньше переменных — лучше

    def _residue(self):
        return [r for r in self.records
                if (lambda p: p is None or p["outcome"] != r["outcome"])(
                    self.mem.predict(
                        self.contextualize(r["ctx"], r["obj"], r["ents"],
                                           at=r.get("step")),
                        r["action"]))]

    def _factor_groups(self):
        """Факторизация: симптомы -> группы взаимной корреляции ->
        одна латентная переменная на группу."""
        acts = sorted(self.problem_actions)
        parent = {a: a for a in acts}

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                pairs = [(ra, rb)
                         for s in self.sig.values()
                         if a in s and b in s
                         for (ra, _ta), (rb, _tb) in [(s[a], s[b])]]
                if (len(pairs) >= MIN_CO and _functional_prob(pairs)
                        and _functional_prob([(y, x) for x, y in pairs])):
                    parent[find(b)] = find(a)

        by_root = defaultdict(set)
        for a in acts:
            by_root[find(a)].add(a)
        groups = []
        for root in sorted(by_root):
            gacts = by_root[root]
            sigs = {o: {a: r for a, (r, _t) in s.items() if a in gacts}
                    for o, s in self.sig.items()}
            clusters = cluster_signatures(
                {o: s for o, s in sigs.items() if s})
            if len(clusters) >= 2:   # переменная с одним значением — не знание
                groups.append({"attr": f"class{len(groups)}",
                               "actions": gacts, "clusters": clusters})
        return groups

    def sleep(self):
        # A6: сканируем записи — если один объект при одном действии дал два
        # разных результата, мир точно динамический
        if not self._dyn:
            seen = {}
            for r in self.records:
                if not r["obj"]:
                    continue
                key = (r["obj"], r["action"])
                res = r["outcome"][0]
                if key in seen and seen[key] != res:
                    self._dyn = True
                    break
                seen[key] = res
        self.mem = self._relabel()
        residue = self._residue()
        uniq = {(r["ctx"], r["action"], r["outcome"]) for r in residue}
        if len(uniq) >= INVENT_AT:
            # изобретение понятий (B3, k=1, факторизованное) — под стражем B1
            old = (self.problem_actions, self.groups, self.mem)
            old_dl = self._dl(self.mem)
            # симптомы накапливаются монотонно: при однопартиционной
            # кластеризации это вредило (фрагментация), при факторизации —
            # обязательно (иначе группы амнезируют при поздних изобретениях)
            self.problem_actions = self.problem_actions | {
                r["action"] for r in residue if r["obj"]}
            self.groups = self._factor_groups()
            cand = self._relabel()
            if self._dl(cand) < old_dl:
                self.mem = cand
            else:
                self.problem_actions, self.groups, self.mem = old
        elif self.problem_actions:
            # A6: даже при малом остатке — перебалансируем топологию групп,
            # накопленных ранее (свежих пар могло стать больше)
            old = (self.groups, self.mem)
            old_dl = self._dl(self.mem)
            self.groups = self._factor_groups()
            cand = self._relabel()
            if self._dl(cand) < old_dl:
                self.mem = cand
            else:
                self.groups, self.mem = old
        # реляционная абдукция (B3, k=2) — под тем же стражем
        cand_R = set()
        for r in self.records:
            if r["obj"] is None:
                continue
            for eff in r["effects"]:
                if len(eff) >= 2:
                    ent = (r["obj"][0], eff[1])
                    if ent in r["ents"]:
                        cand_R.add((r["obj"], ent, r["ents"][ent]))
        if cand_R - self.R:
            old_R, old_mem = self.R, self.mem
            old_dl = self._dl(self.mem)
            self.R = self.R | cand_R
            cand = self._relabel()
            if self._dl(cand) < old_dl:
                self.mem = cand
            else:
                self.R, self.mem = old_R, old_mem
        self.class_relevant = defaultdict(set)
        for r in self.mem.rules:
            for att, _ in r["conds"]:
                if att.startswith("class"):
                    self.class_relevant[r["action"]].add(att)
