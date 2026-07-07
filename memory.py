"""Ядро Эксперимента 0: эпизодическая память + консолидация правил по MDL.

Эпизод: (контекст, действие, исход)
  контекст — frozenset видимых предикатов цели, например
             {(owner,root),(dir,/etc),(ftype,config),(x,0)}
  исход    — (result, отсортированные виды эффектов), например
             ("ok", ("file_deleted","proc_died"))

Правило: ЕСЛИ conds ⊆ контекст И действие=a ТО исход=o
  Приём правила — по MDL: правило принимается, если описание
  "правило + исключения" короче, чем перечисление эпизодов поштучно.

Стоимости кодирования (в условных единицах):
  EP_COST   — хранить один сырой уникальный эпизод
  RULE_BASE + RULE_COND * |conds| — хранить правило
  OVERRIDE  — исключение: эпизод, противоречащий правилу, стоит дороже,
              потому что должен переопределить его предсказание
Выигрыш правила: correct * EP_COST - rule_cost - exceptions * OVERRIDE.
Майнер — жадное покрытие: берём правило с максимальным выигрышем,
убираем объяснённые эпизоды, повторяем, пока выигрыш положителен.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

EP_COST = 4.0
RULE_BASE = 3.0
RULE_COND = 1.0
OVERRIDE = 12.0   # исключение = эпизод + указатель "переопределяю правило";
                  # дёшевые исключения позволяют мажоритарным правилам
                  # побеждать чистые (см. провал первой версии exp2)
MAX_CONDS = 2


def extract_episode(t):
    """transition из world.step() -> (ctx, action, outcome)."""
    action, target = t["action"], t["target"]
    ctx = set()
    if action in ("read", "write", "delete", "chmod", "exec"):
        f = t["obs"]["files"].get(target)
        if f is None:
            return None
        # словарь предикатов = видимые атрибуты; скрытое сюда не попадает
        ctx = {("x" if k == "exec_bit" else k, v)
               for k, v in f.items() if k != "name"}
    elif action in ("kill", "start"):
        p = t["obs"]["procs"].get(target)
        if p is None:
            return None
        ctx = {("status", p["status"])}
    elif action == "create":
        ctx = {("dir", t["args"].get("dir")),
               ("ftype", t["args"].get("ftype")),
               ("x", int(t["args"].get("exec_bit", 0)))}
    outcome = (t["result"], tuple(sorted({e[0] for e in t["effects"]})))
    return (frozenset(ctx), action, outcome)


class Memory:
    def __init__(self):
        self.episodes = Counter()   # (ctx, action, outcome) -> сколько раз видели
        self.rules = []             # (action, conds, outcome, support, exceptions)

    # ---------- запись ----------

    def store(self, transition):
        ep = extract_episode(transition)
        if ep is not None:
            self.episodes[ep] += 1

    # ---------- консолидация ("сон") ----------

    def consolidate(self):
        """Майнит правила из уникальных эпизодов. Мир детерминирован,
        поэтому повторы эпизода не добавляют информации — MDL по уникальным."""
        self.rules = []
        by_action = defaultdict(list)
        for (ctx, action, outcome) in self.episodes:
            by_action[action].append((ctx, outcome))

        for action, eps in sorted(by_action.items()):
            remaining = list(eps)
            while remaining:
                best = self._best_rule(remaining)
                if best is None:
                    break
                conds, outcome, correct, exceptions, gain = best
                self.rules.append({
                    "action": action, "conds": conds, "outcome": outcome,
                    "support": len(correct), "exceptions": len(exceptions),
                    "gain": gain,
                })
                remaining = [e for e in remaining if e not in correct]
        # то, что осталось необъяснённым, живёт в эпизодической памяти как есть
        return self.rules

    def _best_rule(self, eps):
        preds = sorted({p for ctx, _ in eps for p in ctx})
        candidates = [frozenset()]
        for k in range(1, MAX_CONDS + 1):
            candidates += [frozenset(c) for c in combinations(preds, k)]

        best, best_gain = None, 0.0
        for conds in candidates:
            covered = [e for e in eps if conds <= e[0]]
            if not covered:
                continue
            for outcome, n in Counter(o for _, o in covered).most_common(1):
                correct = [e for e in covered if e[1] == outcome]
                exceptions = [e for e in covered if e[1] != outcome]
                gain = (len(correct) * EP_COST
                        - (RULE_BASE + RULE_COND * len(conds))
                        - len(exceptions) * OVERRIDE)
                if gain > best_gain:
                    best_gain = gain
                    best = (conds, outcome, correct, exceptions, gain)
        return best

    # ---------- извлечение ----------

    def predict(self, ctx, action):
        """Матчим контекст к правилам (специфичные — раньше), иначе None."""
        matching = [r for r in self.rules
                    if r["action"] == action and r["conds"] <= ctx]
        if not matching:
            return None
        return max(matching, key=lambda r: (len(r["conds"]), r["support"]))

    # ---------- ответ со смирением ----------

    def _relevant_attrs(self):
        """Атрибуты, которые теория считает значимыми (есть хоть в одном
        правиле). Шум, выброшенный прессом из правил, не должен пугать
        ответчик при поиске свидетелей."""
        return {att for r in self.rules for att, _ in r["conds"]}

    @staticmethod
    def _conflict(ctx_a, ctx_b, relevant):
        d = dict(ctx_a)
        return any(a in d and d[a] != v
                   for a, v in ctx_b if a in relevant)

    def supported(self, ctx, action, rule):
        """Есть ли у правила свидетель (эпизод, породивший его),
        совместимый с запросом по всем значимым атрибутам. Если нет —
        правило в этой точке экстраполирует, а не знает."""
        relevant = self._relevant_attrs()
        for (e_ctx, a, o) in self.episodes:
            if (a == action and o == rule["outcome"]
                    and rule["conds"] <= e_ctx
                    and not self._conflict(e_ctx, ctx, relevant)):
                return True
        return False

    def answer(self, ctx, action):
        """Эпистемически честный ответ (долг раунда ores):
        1) точное воспоминание побеждает правило;
        2) правило отвечает только внутри области своих свидетелей;
        3) иначе None — "не знаю" вместо уверенной экстраполяции.
        Внутренняя машинерия (остаток, MDL, изобретение) продолжает
        пользоваться голым predict — иначе зубрёжка убьёт давление."""
        exact = {o for (c, a, o) in self.episodes
                 if a == action and c == ctx}
        if len(exact) == 1:
            return {"action": action, "conds": ctx, "outcome": exact.pop(),
                    "support": 1, "exceptions": 0, "episodic": True}
        r = self.predict(ctx, action)
        if r is None or not self.supported(ctx, action, r):
            return None
        return r

    # ---------- инспекция глазами ----------

    def dump(self):
        lines = []
        uniq, total = len(self.episodes), sum(self.episodes.values())
        raw_dl = uniq * EP_COST
        explained = set()
        for r in self.rules:
            for (ctx, action, outcome) in self.episodes:
                if action == r["action"] and r["conds"] <= ctx and outcome == r["outcome"]:
                    explained.add((ctx, action, outcome))
        rule_dl = sum(RULE_BASE + RULE_COND * len(r["conds"]) for r in self.rules)
        residual = uniq - len(explained)
        lines.append(f"эпизодов: {total} (уникальных {uniq}), правил: {len(self.rules)}, "
                     f"необъяснённых эпизодов: {residual}")
        lines.append(f"MDL: сырая память {raw_dl:.0f} ед. -> "
                     f"правила {rule_dl:.0f} + остаток {residual * EP_COST:.0f} "
                     f"= {rule_dl + residual * EP_COST:.0f} ед.")
        lines.append("")
        for r in sorted(self.rules, key=lambda r: -r["support"]):
            conds = " & ".join(f"{a}={v}" for a, v in sorted(r["conds"])) or "всегда"
            res, effs = r["outcome"]
            eff = f" + эффекты {list(effs)}" if effs else ""
            lines.append(f"  {r['action']}: ЕСЛИ {conds} ТО {res}{eff}   "
                         f"[объясняет {r['support']}, исключений {r['exceptions']}]")
        return "\n".join(lines)
