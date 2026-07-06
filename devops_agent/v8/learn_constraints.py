"""
learn_constraints.py — агент ДОБЫВАЕТ связки mutex/atmost пробой мира-судьи.

Это не наполнение KB готовыми правилами — это механизм ОБУЧЕНИЯ + стенд проверки.
Мир-судья хранит СКРЫТЫЕ законы (истину); агент их не видит, только ставит конфигурации
и получает ok/not-ok. Грунт = тот же do-исчисление, что и для базового закона mem≥θ:
тумблер/подъём уровня → наблюдение → выживший факт.

Связки (constraints.py):
  MUTEX(i, j)        — оба активны (≥1) нельзя.
  ATMOST(group, B)   — Σ уровней группы ≤ B.

Стенд (критерии из [[v8-rebuild-direction]]):
  A — соответствие истине: precision/recall восстановленных связок vs скрытых.
  B — обобщение НЕ зубрёжка: предсказать ok/not-ok на НОВЫХ случайных конфигурациях
      по ВЫУЧЕННЫМ связкам, 0 новых проб; точность vs судья.
  C — амортизация: B и есть амортизация — после обучения новые инстансы стоят 0 проб.
  D — рефьют: на подсаженных НЕзависимых парах/ресурсах ложных связок быть не должно.
  E — сходимость: число проб ограничено (полиномиально по числу ресурсов), не взрыв.

Допущение демо (честно): atmost-группы НЕпересекающиеся; mutex — над уровнем 1,
atmost-бюджет B≥2 — так связки различимы пробой. Пересекающиеся бюджеты — будущая работа.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from devops_agent.v8.constraints import AtMost, ConstraintLayer, Mutex, validate_constraints


# --------------------------- МИР-СУДЬЯ (истина скрыта) ---------------------------
@dataclass
class World:
    n: int                      # число ресурсов r0..r_{n-1}
    cap: int                    # потолок уровня каждого ресурса
    mutex_pairs: list           # list[tuple(i,j)]  — скрытая истина
    atmost_groups: list         # list[tuple(frozenset members, B)] — скрытая истина

    def judge(self, cfg: dict) -> bool:
        """cfg: {i: level}. ok ⟺ все скрытые законы выполнены. Это ЕДИНСТВЕННЫЙ канал правды."""
        for i, j in self.mutex_pairs:
            if cfg.get(i, 0) > 0 and cfg.get(j, 0) > 0:
                return False
        for members, B in self.atmost_groups:
            if sum(cfg.get(k, 0) for k in members) > B:
                return False
        return True


def make_world(seed: int, n: int = 9, cap: int = 3) -> World:
    """Подсаживаем истину: несколько mutex-пар, несколько atmost-групп, остальное — НЕзависимо.
    Независимые ресурсы — приманка для критерия D (там связок быть не должно)."""
    rnd = random.Random(seed)
    ids = list(range(n))
    rnd.shuffle(ids)
    pool = ids[:]
    mutex_pairs, atmost_groups, used = [], [], set()

    for _ in range(2):                                  # 2 mutex-пары
        free = [x for x in pool if x not in used]
        if len(free) < 2:
            break
        a, b = free[0], free[1]
        mutex_pairs.append((a, b)); used |= {a, b}

    for size in (2, 3):                                 # atmost-группы размером 2 и 3
        free = [x for x in pool if x not in used]
        if len(free) < size:
            break
        members = frozenset(free[:size])
        # bound связывающий: 2 ≤ B < Σcap, чтобы закон наблюдался пробой
        B = rnd.randint(2, max(2, size * cap - 1))
        atmost_groups.append((members, B)); used |= set(members)

    return World(n=n, cap=cap, mutex_pairs=mutex_pairs, atmost_groups=atmost_groups)


# --------------------------- ГРУНТ-ОБУЧАТЕЛЬ (пробами) ---------------------------
class ProbeLearner:
    def __init__(self, world: World):
        self.w = world
        self.probes = 0

    def _judge(self, cfg: dict) -> bool:
        self.probes += 1
        return self.w.judge(cfg)

    def _solo_max(self, i: int) -> int:
        """Макс. уровень ресурса i в одиночку, что ещё ok (раскрывает собственный потолок/одиночный бюджет)."""
        last = 0
        for v in range(0, self.w.cap + 1):
            if self._judge({i: v}):
                last = v
            else:
                break
        return last

    def learn(self) -> ConstraintLayer:
        n = self.w.n
        solo = {i: self._solo_max(i) for i in range(n)}
        active = [i for i in range(n) if solo[i] >= 1]      # ресурсы, способные быть активными

        # 1) MUTEX: оба активны на уровне 1 → not-ok, поодиночке ok.
        mutex = []
        mset = set()
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                i, j = active[a_idx], active[b_idx]
                if self._judge({i: 1, j: 1}) is False:
                    mutex.append(Mutex(members=(i, j)))
                    mset.add(i); mset.add(j)

        # 2) ATMOST: НЕ-mutex ресурсы накапливаем по одному на их СОЛО-максимумах
        #    (каждый поодиночке ok). Как только подмножество вместе not-ok — это группа бюджета;
        #    минимизируем её до ядра (убираем лишних, кто не нужен для нарушения) и ищем bound.
        #    Группа любого размера ≥2 видна (не только пары). Допущение: группы НЕпересекающиеся.
        quant = [i for i in active if i not in mset]
        groups = self._discover_groups(quant, solo)
        atmost = [AtMost(members=tuple(sorted(g)), bound=self._find_bound(g)) for g in groups]
        return ConstraintLayer(mutexes=mutex, atmosts=atmost)

    def _discover_groups(self, quant: list, solo: dict) -> list:
        remaining, groups, acc, i = list(quant), [], [], 0
        while i < len(remaining):
            acc.append(remaining[i])
            if self._judge({k: solo[k] for k in acc}) is False:   # подмножество вместе нарушает
                core = self._minimize(acc, solo)                  # сертификат-подмножество ⊆ группы
                group = self._grow(core, remaining, solo)         # дорастить до ПОЛНОЙ группы бюджета
                groups.append(group)
                remaining = [r for r in remaining if r not in group]
                acc, i = [], 0
            else:
                i += 1
        return groups

    def _minimize(self, acc: list, solo: dict) -> list:
        """Минимальное нарушающее ядро: выкидываем члена, если без него всё ещё not-ok."""
        core = list(acc)
        for x in list(acc):
            trial = [k for k in core if k != x]
            if len(trial) >= 1 and self._judge({k: solo[k] for k in trial}) is False:
                core = trial
        return core

    def _grow(self, core: list, pool: list, solo: dict) -> set:
        """Дорастить ядро до полного членства: насыщаем группу до её bound и добавляем r,
        если бамп r ломает (значит r тянет тот же бюджет). Ловит и тугие, и свободные группы."""
        group = set(core)
        B = self._find_bound(group)
        changed = True
        while changed:
            changed = False
            for r in pool:
                if r in group:
                    continue
                sat = self._spread(sorted(group), B)     # сумма по группе = B (на грани)
                sat[r] = solo[r]
                if self._judge(sat) is False:            # r делит тот же бюджет
                    group.add(r); B = self._find_bound(group); changed = True
        return group

    def _find_bound(self, members: set) -> int:
        """Бинарный поиск макс. суммы S по группе, что ещё ok (репрезентативная раскладка S по членам)."""
        ms = sorted(members)
        hi = sum(min(self.w.cap, self.w.cap) for _ in ms)   # верх: |members|*cap
        lo, best = 0, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._judge(self._spread(ms, mid)):
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _spread(self, ms: list, total: int) -> dict:
        """Разложить сумму total по членам ≤ cap (жадно). Для одиночного atmost ok зависит лишь от суммы."""
        cfg, left = {}, total
        for k in ms:
            take = min(self.w.cap, left)
            cfg[k] = take; left -= take
        return cfg


# --------------------------- СТЕНД A / B / D / E ---------------------------
def truth_layer(w: World) -> ConstraintLayer:
    return ConstraintLayer(
        mutexes=[Mutex(members=tuple(sorted(p))) for p in w.mutex_pairs],
        atmosts=[AtMost(members=tuple(sorted(m)), bound=B) for m, B in w.atmost_groups],
    )


def predict_ok(layer: ConstraintLayer, cfg: dict) -> bool:
    for m in layer.mutexes:
        a, b = m.members
        if cfg.get(a, 0) > 0 and cfg.get(b, 0) > 0:
            return False
    for am in layer.atmosts:
        if sum(cfg.get(k, 0) for k in am.members) > am.bound:
            return False
    return True


def evaluate(seed: int, n: int = 9, cap: int = 3, novel: int = 500) -> dict:
    w = make_world(seed, n=n, cap=cap)
    learner = ProbeLearner(w)
    learned = learner.learn()
    truth = truth_layer(w)

    tk, lk = truth.keys(), learned.keys()
    tp = len(tk & lk); fp = len(lk - tk); fn = len(tk - lk)

    # B/C — предсказание на НОВЫХ конфигурациях по выученному, 0 новых проб
    rnd = random.Random(seed * 7919 + 1)
    correct = 0
    for _ in range(novel):
        cfg = {i: rnd.randint(0, cap) for i in range(n)}
        if predict_ok(learned, cfg) == w.judge(cfg):
            correct += 1

    contract_viol = validate_constraints(
        learned, resource_ids=set(range(n)), status_ids=set(), setting_ids=set())

    return {
        "seed": seed,
        "truth": {"mutex": len(truth.mutexes), "atmost": len(truth.atmosts)},
        "learned": {"mutex": len(learned.mutexes), "atmost": len(learned.atmosts)},
        "A_tp": tp, "A_fp": fp, "A_fn": fn,
        "B_acc": correct / novel,
        "E_probes": learner.probes,
        "contract_ok": not contract_viol,
        "truth_keys": sorted(map(str, tk)),
        "learned_keys": sorted(map(str, lk)),
    }


def main() -> None:
    print("=== грунт-обучение связок mutex/atmost пробой мира-судьи; стенд A/B/D/E ===\n")
    seeds = list(range(8))
    agg_tp = agg_fp = agg_fn = 0
    accs, probes = [], []
    for s in seeds:
        r = evaluate(s)
        agg_tp += r["A_tp"]; agg_fp += r["A_fp"]; agg_fn += r["A_fn"]
        accs.append(r["B_acc"]); probes.append(r["E_probes"])
        flag = "" if (r["A_fp"] == 0 and r["A_fn"] == 0 and r["B_acc"] == 1.0) else "  <-- расхождение"
        print(f"seed {s}: истина {r['truth']} → выучено {r['learned']} | "
              f"A: tp={r['A_tp']} fp={r['A_fp']} fn={r['A_fn']} | "
              f"B(новые)={r['B_acc']:.3f} | E(проб)={r['E_probes']} | контракт={'ok' if r['contract_ok'] else 'НАРУШЕН'}{flag}")
        if flag:
            print(f"        истина : {r['truth_keys']}")
            print(f"        выучено: {r['learned_keys']}")

    prec = agg_tp / max(1, agg_tp + agg_fp)
    rec = agg_tp / max(1, agg_tp + agg_fn)
    print(f"\nA (соответствие истине): precision={prec:.3f} recall={rec:.3f}  (tp={agg_tp} fp={agg_fp} fn={agg_fn})")
    print(f"B/C (обобщение на новых, 0 проб): средняя точность={sum(accs)/len(accs):.3f}  min={min(accs):.3f}")
    print(f"D (рефьют): ложных связок на независимых ресурсах = {agg_fp}")
    print(f"E (сходимость): проб на мир min={min(probes)} max={max(probes)} avg={sum(probes)/len(probes):.0f} "
          f"(n=9, cap=3; перебор пар ~O(n²))")
    verdict = "ЖИВ" if (agg_fp == 0 and agg_fn == 0 and min(accs) == 1.0) else "ТРЕБУЕТ РАЗБОРА"
    print(f"\nвердикт механизма: {verdict}")


if __name__ == "__main__":
    main()
