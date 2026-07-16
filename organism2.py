"""organism2 — ядро, переписанное из MODEL.md (заморозка 13 июля 2026).

Четыре коробки модели:
  1. Мир — скрытая цепочка: у объекта по каждой тайной оси состояние,
     меняющееся во времени с выучиваемой скоростью (worldkit: hazard).
  2. Память — уверенность (belief) на объект и ось; увидел — вычеркни
     несовместимое, подождал — размажь по кубику мира.
  3. Обучение — счётчики: таблица ответов (эмиссии), счётчик кубика
     (грид-апостериор с прайором Джеффриса — равномерным по линейке
     различимости Фишера), правила по видимому контексту.
  4. Судья — сжатие минус Штраф Оккама. Все цены — в битах, выведенных
     из словарей данных; непрерывный параметр стоит полбита на порцию
     данных (½·log2 N — счёт делений линейки различимости).

Правило слепоты (MODEL.md, Д3): ядро не знает ни одного мира; каждое
число выводится из прогона. Единственная цитата из протокола:
MAX_CONDS = 2 — обещание грамматики миров (WORLD_PROTOCOL, R1).

Интерфейс совместим с runworld (см. runworld2.py).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from itertools import combinations

MAX_CONDS = 2      # R1 протокола: исход = конъюнкция <= 2 предикатов

LN2 = math.log(2.0)


def log2(x):
    return math.log(x) / LN2


# ======================================================================
# Судья-минёр правил: та же жадная консолидация, что в memory.py, но все
# цены — биты из словарей данных. EP_COST/RULE_BASE/OVERRIDE умерли.
# ======================================================================

class Ruleset:
    """Эпизод: (ctx: frozenset[(атрибут, значение)], действие, исход).

    Цены кодирования (биты, всё выведено из данных):
      сырой эпизод    цена максимально специфичного правила: (|ctx|+1)
                      флагов + |ctx| * log2(#предикатов) + биты исхода —
                      эпизодическая память не бесплатна, у неё есть длина
      правило         (|conds|+1) флагов «ещё условие?»
                      + |conds| * log2(#предикатов) + биты исхода
      исключение      log2(#эпизодов) указатель + биты исхода
    Правило принимается, если сжимает: выигрыш > 0. Никаких констант.
    """

    def __init__(self):
        self.episodes = Counter()
        self.rules = []
        self._outb = {}
        self._predb = 1.0

    # ---------- словари ----------

    def _vocab(self):
        outs = defaultdict(set)
        preds = set()
        for (ctx, a, o) in self.episodes:
            outs[a].add(o)
            preds |= set(ctx)
        self._outb = {a: log2(max(2, len(v))) for a, v in outs.items()}
        self._predb = log2(max(2, len(preds)))

    def out_bits(self, a):
        return self._outb.get(a, 1.0)

    def _rule_cost(self, a, conds):
        return (len(conds) + 1) + len(conds) * self._predb + self.out_bits(a)

    def _raw_cost(self, ctx, a):
        """Эпизод, хранимый как есть, = максимально специфичное правило."""
        return self._rule_cost(a, ctx)

    def _group_resid(self, ctx, a, oms, explained):
        """ЕДИНСТВЕННАЯ семантика остатка (арка F3): цена необъяснённых
        строк одной группы (ctx, действие). oms: [(исход, m)]. Каждая
        строка платит полный сюрприз m·out_bits; кап «запомнить как
        специфичное правило» валиден лишь для ОДНОЙ строки группы —
        правила о прочих исходах того же ctx взаимоисключающи — И только
        когда НИ ОДНА строка группы не предсказана верно (explained=False):
        специфичное правило-запоминалка перекрыло бы верное предсказание
        строки-сестры того же ctx, т.е. для смеси под работающим правилом
        капа нет — меньшинство платит полный сюрприз на каждом повторе
        (иначе незнание снова продаётся по цене знания, теперь через путь
        правил — чёрный ход дефекта 07935fe). При точной ничьей экономия
        капа делится поровну между претендентами: выбор любимчика по имени
        исхода — лексикография слов мира (F2). Потребители: судья
        (total_bits) и майнер (_best_rule) — по-строчный min-кап у майнера
        занижал цену беспорядка и давал ложные отказы правилам, которые
        судья бы окупил."""
        full = {o: m * self.out_bits(a) for o, m in oms}
        if explained:
            return full
        raw = self._raw_cost(ctx, a)
        smax = max(f - min(raw, f) for f in full.values())
        if smax <= 0.0:
            return full
        ties = [o for o, f in full.items() if f - min(raw, f) == smax]
        share = smax / len(ties)
        return {o: f - share if o in ties else f for o, f in full.items()}

    # ---------- консолидация ----------

    def consolidate(self):
        self._vocab()
        # Канонические ранги тай-брейков — порядок ПЕРВОГО ПОЯВЛЕНИЯ в
        # прожитой истории (у изоморфных жизней одинаков), не лексикография
        # слов мира (F2/N14: при точной ничьей выигрышей выбор кандидата
        # по алфавиту делал набор правил функцией переименования). Предел
        # немоты: предикаты-одновременники (впервые видны в ОДНОМ эпизоде)
        # ранжируются именем — стреляет лишь на точной ничьей кандидатов
        # с разными предикатами.
        self._o_rank, self._p_rank = {}, {}
        for (ctx, a, o) in self.episodes:
            self._o_rank.setdefault((a, o), len(self._o_rank))
            for p in sorted(ctx):
                self._p_rank.setdefault(p, len(self._p_rank))
        self.rules = []
        by_action = defaultdict(lambda: defaultdict(list))
        for (ctx, a, o), m in self.episodes.items():
            by_action[a][ctx].append((o, m))
        for a, groups in sorted(by_action.items()):
            while True:
                best = self._best_rule(a, groups)
                if best is None:
                    break
                conds, outcome, correct, exceptions, gain = best
                self.rules.append({
                    "action": a, "conds": conds, "outcome": outcome,
                    "support": len(correct), "exceptions": len(exceptions),
                    "gain": gain,
                })
            # опора/исключения — семантикой РЕШАЮЩЕГО СПИСКА: правило
            # отвечает лишь за группы, где оно победитель predict; счётчики
            # добычи приписали бы ∅-правилу строки, которые позже объяснит
            # перекрытие (и раздули бы uncertainty)
            arules = [r for r in self.rules if r["action"] == a]
            for r in arules:
                r["support"] = r["exceptions"] = 0
            for ctx, oms in groups.items():
                p = self.predict(ctx, a)
                if p is None:
                    continue
                for (o, _m) in oms:
                    if o == p["outcome"]:
                        p["support"] += 1
                    else:
                        p["exceptions"] += 1
        return self.rules

    def _best_rule(self, a, groups):
        # Выигрыш кандидата = ЧЕСТНАЯ дельта судьи (арка F3, одна
        # семантика остатка). Условия кандидата — подмножество ctx, поэтому
        # группа (ctx, a) покрывается целиком; но кандидат ВЫСТРЕЛИВАЕТ
        # только там, где он строго специфичнее текущего победителя predict
        # (при равной специфичности читается раньше добытое — новичок в
        # тени и не объясняет НИЧЕГО; теневые правила себе выигрыш не
        # пишут). В выстрелившей группе необъяснённое ДО — строки мимо
        # текущего победителя, ПОСЛЕ — строки мимо кандидата: воскрешение
        # уже объяснённых строк оплачивается, остаток выживших
        # пересчитывается той же _group_resid (кап достаётся заново).
        #   gain = Σ по выстрелившим группам [остаток ДО − остаток ПОСЛЕ]
        #          − цена правила.
        win = {}
        for ctx in groups:
            m_ = [(len(r["conds"]), -i, r["outcome"])
                  for i, r in enumerate(self.rules)
                  if r["action"] == a and r["conds"] <= ctx]
            win[ctx] = max(m_) if m_ else None
        wrong, before = {}, {}
        for ctx, oms in groups.items():
            w = win[ctx]
            bad = [(o, m) for o, m in oms if w is None or o != w[2]]
            wrong[ctx] = bad
            expl = w is not None and len(bad) < len(oms)
            before[ctx] = (sum(self._group_resid(ctx, a, bad, expl).values())
                           if bad else 0.0)
        # кандидаты — из групп, где есть необъяснённое (иначе выигрыша нет)
        cands = set()
        for ctx, bad in wrong.items():
            if not bad:
                continue
            preds = sorted(ctx, key=self._p_rank.get)
            cands.add(())
            cands.update((p,) for p in preds)
            if MAX_CONDS >= 2:
                cands.update((p, q) for i_, p in enumerate(preds)
                             for q in preds[i_ + 1:])
        best, best_gain = None, 0.0
        for c in sorted(cands, key=lambda c: (len(c),
                                              [self._p_rank[p] for p in c])):
            cs = frozenset(c)
            fired = [ctx for ctx in groups
                     if cs <= ctx
                     and (win[ctx] is None or len(cs) > win[ctx][0])]
            if not fired:
                continue
            outs = {o for ctx in fired for (o, _m) in wrong[ctx]}
            for outcome in sorted(outs, key=lambda o: self._o_rank[(a, o)]):
                gain = -self._rule_cost(a, cs)
                for ctx in fired:
                    rest = [(o, m) for o, m in groups[ctx] if o != outcome]
                    expl = len(rest) < len(groups[ctx])
                    after = (sum(self._group_resid(ctx, a, rest,
                                                   expl).values())
                             if rest else 0.0)
                    gain += before[ctx] - after
                if gain > best_gain:
                    best_gain = gain
                    best = (cs, outcome, fired)
        if best is None:
            return None
        conds, outcome, fired = best
        covered = [(ctx, o, m) for ctx in fired for (o, m) in groups[ctx]]
        correct = [e for e in covered if e[1] == outcome]
        exceptions = [e for e in covered if e[1] != outcome]
        return conds, outcome, correct, exceptions, best_gain

    def total_bits(self):
        """Полная длина описания пережитой истории этим набором правил.

        Остаток — через _group_resid (одна семантика с майнером, арка F3):
        кап «запомнить как специфичное правило» валиден лишь для ОДНОЙ
        строки группы (ctx, действие) — правила о прочих исходах того же
        ctx взаимоисключающи, их повторы платят полный сюрприз m·out_bits.
        Прежний по-строчный min-кап продавал незнание по цене знания
        (смесь исходов под одним ctx — а это и есть скрытое состояние —
        почти бесплатна), из-за чего ось не окупалась никогда: отказ hard1
        при +270 битах у форс-фита и немонотонность приёмки по бюджету
        (тумблер 1500+/2000−, насыщение капа). Арка судьи, 15.07.2026;
        диагностика — judge_probe.py."""
        bits = sum(self._rule_cost(r["action"], r["conds"])
                   for r in self.rules)
        groups = defaultdict(list)
        explained = set()
        for (ctx, a, o), m in self.episodes.items():
            p = self.predict(ctx, a)
            if p is None or p["outcome"] != o:
                groups[(ctx, a)].append((o, m))
            else:
                explained.add((ctx, a))
        for (ctx, a), oms in groups.items():
            bits += sum(self._group_resid(ctx, a, oms,
                                          (ctx, a) in explained).values())
        return bits

    # ---------- извлечение (семантика memory.py, без изменений) ----------

    def predict(self, ctx, action):
        """Порядок чтения = порядок добычи: майнер строил упорядоченное
        покрытие, ответчик обязан читать тот же список тем же порядком
        (специфичные раньше общих, при равной специфичности — раньше
        добытое). Иначе модель судьи и ответы расходятся."""
        matching = [(len(r["conds"]), -i, r)
                    for i, r in enumerate(self.rules)
                    if r["action"] == action and r["conds"] <= ctx]
        if not matching:
            return None
        return max(matching)[2]

    def _relevant_attrs(self):
        return {att for r in self.rules for att, _ in r["conds"]}

    @staticmethod
    def _conflict(ctx_a, ctx_b, relevant):
        d = dict(ctx_a)
        return any(a in d and d[a] != v
                   for a, v in ctx_b if a in relevant)

    def supported(self, ctx, action, rule):
        relevant = self._relevant_attrs()
        for (e_ctx, a, o) in self.episodes:
            if (a == action and o == rule["outcome"]
                    and rule["conds"] <= e_ctx
                    and not self._conflict(e_ctx, ctx, relevant)):
                return True
        return False

    def answer(self, ctx, action):
        exact = {o for (c, a, o) in self.episodes
                 if a == action and c == ctx}
        if len(exact) == 1:
            return {"action": action, "conds": ctx, "outcome": exact.pop(),
                    "support": 1, "exceptions": 0, "episodic": True}
        r = self.predict(ctx, action)
        if r is None or not self.supported(ctx, action, r):
            return None
        return r

    def uncertainty(self, ctx, action):
        """Насколько исход (ctx, action) неизвестен: нет правил или
        подходящие правила спорят между собой — полное незнание;
        иначе — доля исключений победителя."""
        matching = [r for r in self.rules
                    if r["action"] == action and r["conds"] <= ctx]
        if not matching:
            return 1.0
        if len({r["outcome"] for r in matching}) > 1:
            return 1.0
        r = max(matching, key=lambda r: (len(r["conds"]), r["support"]))
        tot = r["support"] + r["exceptions"]
        return r["exceptions"] / tot if tot else 1.0


# ======================================================================
# Тайная ось: k состояний, таблица ответов, кубик на состояние.
# ======================================================================

def _matmul(A, B):
    k = len(A)
    return [[sum(A[i][m] * B[m][j] for m in range(k)) for j in range(k)]
            for i in range(k)]


def _matpow(A, n):
    k = len(A)
    R = [[float(i == j) for j in range(k)] for i in range(k)]
    while n:
        if n & 1:
            R = _matmul(R, A)
        A = _matmul(A, A)
        n >>= 1
    return R


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


class Axis:
    def __init__(self, actions, k, res_a):
        self.actions = tuple(sorted(actions))
        self.k = k
        # порядок результатов — как дан (первое появление в трассе, F2);
        # sorted() здесь возвращал лексикографию слов мира в подпись
        self.res_a = {a: list(rs) for a, rs in res_a.items()}
        self.emis = {}                       # (s, a) -> {result: вес}
        self.haz = [0.0] * k                 # кубик: шанс уйти из s за тик
        self.tgt = [[0.0] * k for _ in range(k)]   # куда уходит
        self.pi = [1.0 / k] * k              # начальное распределение
        self.n_obs = 0
        self.n_pairs = 0
        self.score = -math.inf               # лок. счёт судьи (биты)
        self.pen_dyn = 2.0                   # биты динамики (кубики, цели, k)
        self._tpow = {}                      # кэш T^dt (сброс при обновлении)

    # ---------- вероятности ----------

    def emis_p(self, s, a, r):
        c = self.emis.get((s, a), {})
        tot = sum(c.values())
        ra = max(2, len(self.res_a.get(a, ())) + (r not in self.res_a.get(a, ())))
        return (c.get(r, 0.0) + 0.5) / (tot + 0.5 * ra)

    def T(self):
        m = [[0.0] * self.k for _ in range(self.k)]
        for s in range(self.k):
            m[s][s] = 1.0 - self.haz[s]
            for j in range(self.k):
                if j != s:
                    m[s][j] = self.haz[s] * self.tgt[s][j]
        return m

    def tpow(self, dt):
        m = self._tpow.get(dt)
        if m is None:
            m = self._tpow[dt] = _matpow(self.T(), dt)
        return m

    def evolve(self, bel, dt):
        """Подождал — размажь уверенность по кубику мира."""
        if dt <= 0:
            return list(bel)
        Tdt = self.tpow(dt)
        return [sum(bel[s] * Tdt[s][j] for s in range(self.k))
                for j in range(self.k)]

    def see(self, bel, a, r):
        """Увидел — вычеркни состояния, которые так не ответили бы."""
        return _norm([bel[s] * self.emis_p(s, a, r) for s in range(self.k)])

    # ---------- фильтрация/сглаживание ----------

    def filter(self, timeline, t_query):
        """Belief в момент t_query по наблюдениям до него включительно."""
        bel = list(self.pi)
        t_last = None
        for (t, a, r) in timeline:
            if t > t_query:
                break
            if t_last is not None:
                bel = self.evolve(bel, t - t_last)
            bel = self.see(bel, a, r)
            t_last = t
        if t_last is not None and t_query > t_last:
            bel = self.evolve(bel, t_query - t_last)
        return bel

    def fb(self, timeline):
        """Вперёд-назад (Баум—Велч): сглаженные апостериоры состояний
        γ, парные апостериоры переходов ξ по каждому промежутку и
        лог-правдоподобие линии."""
        n = len(timeline)
        if n == 0:
            return [], [], 0.0
        k = self.k
        Ts = [None] * n
        emit = [None] * n
        for i, (t, a, r) in enumerate(timeline):
            emit[i] = [self.emis_p(s, a, r) for s in range(k)]
            if i:
                Ts[i] = self.tpow(timeline[i][0] - timeline[i - 1][0])
        alpha, scales = [], []
        for i in range(n):
            if i == 0:
                v = [emit[0][s] * self.pi[s] for s in range(k)]
            else:
                prev = alpha[-1]
                v = [sum(prev[s] * Ts[i][s][j] for s in range(k))
                     * emit[i][j] for j in range(k)]
            sc = sum(v) or 1e-300
            alpha.append([x / sc for x in v])
            scales.append(sc)
        beta = [[1.0] * k for _ in range(n)]
        for i in range(n - 2, -1, -1):
            sc = scales[i + 1]
            beta[i] = [sum(Ts[i + 1][s][j] * emit[i + 1][j]
                           * beta[i + 1][j] for j in range(k)) / sc
                       for s in range(k)]
        gammas = [_norm([alpha[i][s] * beta[i][s] for s in range(k)])
                  for i in range(n)]
        xis = []
        for i in range(1, n):
            xi = [[alpha[i - 1][s] * Ts[i][s][j] * emit[i][j]
                   * beta[i][j] / scales[i]
                   for j in range(k)] for s in range(k)]
            z = sum(sum(row) for row in xi) or 1e-300
            xis.append([[x / z for x in row] for row in xi])
        return gammas, xis, sum(math.log(sc) for sc in scales)

    def smooth(self, timeline):
        gammas, _xis, ll = self.fb(timeline)
        return gammas, ll


def _cluster_signatures(sigs):
    """Жадная склейка совместимых подписей {действие: результат} —
    стартовая точка для EM (сшивка состояний между действиями идёт от
    объектов, измеренных несколькими симптомами). Решений не принимает."""
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
    clusters.sort(key=lambda c: -len(c[1]))
    return clusters


def _kt_regret(counts):
    """Штраф Оккама клетки-распределения: длина кода смеси Джеффриса
    (Кричевский–Трофимов) минус ML-сжатие. Точная конечная формула
    вместо асимптотики ½·log n: чёткой клетке у края дуги различимости
    честно дешевле — та самая геометрическая поправка юности."""
    n = sum(counts)
    m = len(counts)
    if n <= 0 or m < 2:
        return 0.0
    kt = -(math.lgamma(m * 0.5) - m * math.lgamma(0.5)
           + sum(math.lgamma(c + 0.5) for c in counts)
           - math.lgamma(n + m * 0.5)) / LN2
    ml = -sum(c * math.log(c / n) for c in counts if c > 0) / LN2
    return max(0.0, kt - ml)


HAZ_GRID = [(i + 0.5) / 202.0 for i in range(101)]      # (0, 0.5)


def _stay2(a, b, dt):
    """Точная двухсостоянийная цепь: P(остаться в 0 за dt) при уходах
    a (0->1) и b (1->0) — с учётом обратных путей."""
    if a + b <= 0.0:
        return 1.0
    lam = (1.0 - a - b) ** dt
    return b / (a + b) + (a / (a + b)) * lam


_HAZ_PRIOR = [1.0 / math.sqrt(h * (1.0 - h)) for h in HAZ_GRID]
_HAZ_PRIOR = [w / sum(_HAZ_PRIOR) for w in _HAZ_PRIOR]


def _hazard_coordinate(ax, s, pairs_ev):
    """Кубик состояния s: грид-апостериор с прайором Джеффриса
    (равномерным по дуге линейки различимости), правдоподобие — точное
    T^dt при текущих прочих параметрах: длинная пара почти ничего не
    говорит о скорости (стационар), и формула это знает. Возвращает
    (среднее апостериора, штраф Оккама кубика = длина кода точечной
    оценки минус длина кода смеси) — юность без данных отвечает
    серединой дуги и платит ноль."""
    k = ax.k
    # агрегировать улики по dt: сто одинаковых промежутков — одна строка
    by_dt = {}
    for dt, xi in pairs_ev:
        row = by_dt.get(dt)
        if row is None:
            row = by_dt[dt] = [0.0] * k
        for j in range(k):
            row[j] += xi[s][j]
    # возврат в s: для k=2 — точно (кубик соседа); для k>2 — цепь
    # сворачивается в «s против остальных» (средний приток обратно)
    if k == 2:
        b_ret = ax.haz[1 - s]
    else:
        b_ret = sum(ax.haz[j] * ax.tgt[j][s]
                    for j in range(k) if j != s) / (k - 1)
    b_ret = min(max(b_ret, 1e-9), 0.5)
    lls = []
    for h in HAZ_GRID:
        ll = 0.0
        for dt, row in by_dt.items():
            stay = min(max(_stay2(h, b_ret, dt), 1e-12), 1.0 - 1e-12)
            w_leave = sum(row) - row[s]
            ll += row[s] * math.log(stay) + w_leave * math.log(1.0 - stay)
        lls.append(ll)
    m = max(lls)
    post = [math.exp(ll - m) * pw for ll, pw in zip(lls, _HAZ_PRIOR)]
    z = sum(post)
    h_hat = sum(h * w for h, w in zip(HAZ_GRID, post)) / z
    # длина кода смеси Джеффриса против кода точечной оценки
    mix_ll = m + math.log(z)
    i_hat = min(range(len(HAZ_GRID)), key=lambda i: abs(HAZ_GRID[i] - h_hat))
    regret = max(0.0, (lls[i_hat] - mix_ll) / LN2)
    return h_hat, regret


def _hazard_mle(ax, s, pairs_ev):
    """M-шаг кубика s: мода того же правдоподобия (_stay2), найденная
    золотым сечением за ~18 вычислений вместо сетки в 101 точку. Нижняя
    граница = разрешение грида HAZ_GRID[0], чтобы почти-нулевой кубик
    садился туда же, куда садилось среднее апостериора грида (прайор-
    эффект сохранён без сетки). Регрет-штраф здесь НЕ считается —
    он берётся гридом один раз на сошедшихся параметрах."""
    k = ax.k
    by_dt = {}
    for dt, xi in pairs_ev:
        r = by_dt.get(dt)
        if r is None:
            r = by_dt[dt] = [0.0] * k
        for j in range(k):
            r[j] += xi[s][j]
    if k == 2:
        b_ret = ax.haz[1 - s]
    else:
        b_ret = min(max(sum(ax.haz[j] * ax.tgt[j][s]
                            for j in range(k) if j != s) / (k - 1),
                        1e-9), 0.5)
    rows = list(by_dt.items())

    def ll(h):
        out = 0.0
        for dt, row in rows:
            stay = min(max(_stay2(h, b_ret, dt), 1e-12), 1.0 - 1e-12)
            out += row[s] * math.log(stay) + (sum(row) - row[s]) * math.log(1 - stay)
        return out

    lo, hi = HAZ_GRID[0], 0.5
    g = (math.sqrt(5.0) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = ll(c), ll(d)
    for _ in range(18):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = ll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = ll(d)
    return (lo + hi) / 2


def fit_axis(actions, timelines, k, restarts=2, iters=5):
    """EM: таблица ответов + кубики. timelines: {obj: [(t, a, r), ...]}.

    res_a — упорядоченное множество: порядок ПЕРВОГО ПОЯВЛЕНИЯ результата
    в прожитой трассе, не алфавит (арка немоты F2: rng-старт эмиссий по
    sorted(res_a) делал форму функцией лексикографии слов мира — изоморф
    с обращённым алфавитом давал другие биты; порядок появления у
    изоморфных жизней одинаков по построению)."""
    res_a = defaultdict(dict)
    for tl in timelines.values():
        for (_t, a, r) in tl:
            res_a[a].setdefault(r, None)
    n_obs = sum(len(tl) for tl in timelines.values())
    n_pairs = sum(max(0, len(tl) - 1) for tl in timelines.values())
    # со-подписи объектов (последний результат по каждому действию) —
    # якорь сшивки состояний между действиями для стартовой точки EM
    sigs = {}
    for obj, tl in timelines.items():
        s = {}
        for (_t, a, r) in tl:
            s[a] = r
        if s:
            sigs[obj] = s
    clusters = _cluster_signatures(sigs)
    best = None
    for rs in range(restarts):
        rng = random.Random(rs)
        ax = Axis(actions, k, res_a)
        ax.n_obs, ax.n_pairs = n_obs, n_pairs
        for s in range(k):
            for a in res_a:
                if rs == 0 and s < len(clusters):
                    # рестарт 0: старт от со-подписей — эмиссии состояния
                    # s из реальных счётчиков s-го по величине кластера
                    # (сильный старт — вес старта равен весу улик)
                    cnt = defaultdict(float)
                    for obj in clusters[s][1]:
                        for (_t, a2, r) in timelines.get(obj, ()):
                            if a2 == a:
                                cnt[r] += 1.0
                    ax.emis[(s, a)] = {r: cnt.get(r, 0.0) + 0.5
                                       for r in res_a[a]}
                else:
                    ax.emis[(s, a)] = {r: 0.5 + rng.random()
                                       for r in res_a[a]}
            ax.haz[s] = 0.05 if k > 1 else 0.0
            ax.tgt[s] = [0.0 if j == s else 1.0 / max(1, k - 1)
                         for j in range(k)]
        ll = -math.inf
        row_w = [1.0] * k
        for _ in range(iters):
            new_emis = {(s, a): defaultdict(float)
                        for s in range(k) for a in res_a}
            pairs_ev = []
            tgt_w = [[0.0] * k for _ in range(k)]
            pi_w = [0.0] * k
            row_w = [0.0] * k
            ll = 0.0
            for tl in timelines.values():
                gammas, xis, l = ax.fb(tl)
                ll += l
                for s in range(k):
                    pi_w[s] += gammas[0][s]
                for i, (t, a, r) in enumerate(tl):
                    for s in range(k):
                        new_emis[(s, a)][r] += gammas[i][s]
                for i in range(1, len(tl)):
                    dt = tl[i][0] - tl[i - 1][0]
                    xi = xis[i - 1]
                    pairs_ev.append((dt, xi))
                    for s in range(k):
                        row_w[s] += sum(xi[s])
                        for j in range(k):
                            if j != s:
                                tgt_w[s][j] += xi[s][j]
            for key, cnt in new_emis.items():
                ax.emis[key] = dict(cnt)
            for s in range(k):
                # M-шаг кубика: мода правдоподобия золотым сечением.
                # Грид-регрет (Штраф Оккама) считаем ОДИН раз в конце —
                # на каждой итерации нужна только оценка h.
                ax.haz[s] = _hazard_mle(ax, s, pairs_ev) if k > 1 else 0.0
                tot = sum(tgt_w[s][j] for j in range(k) if j != s)
                ax.tgt[s] = [0.0 if j == s else
                             (tgt_w[s][j] + 0.5) / (tot + 0.5 * (k - 1))
                             for j in range(k)] if k > 1 else [0.0]
            ax.pi = _norm([w + 0.5 for w in pi_w])
            ax._tpow.clear()   # параметры сменились — степени устарели
        # Штраф Оккама кубика: грид-регрет ОДИН раз на сошедшихся
        # параметрах (а не на каждой EM-итерации)
        haz_regret = [(_hazard_coordinate(ax, s, pairs_ev)[1] if k > 1 else 0.0)
                      for s in range(k)]
        # Штраф Оккама оси: точный код смеси Джеффриса на клетку —
        # линейка различимости каждого параметра по ЕГО порции данных.
        emis_bits = sum(
            _kt_regret([ax.emis.get((s, a), {}).get(r, 0.0)
                        for r in res_a[a]])
            for s in range(k) for a in res_a)
        dyn_bits = 2.0                       # выбор k
        if k > 1:
            for s in range(k):
                dyn_bits += haz_regret[s]                        # кубик
                if k > 2:
                    dyn_bits += _kt_regret(
                        [tgt_w[s][j] for j in range(k) if j != s])
            dyn_bits += _kt_regret(pi_w)                          # пи
        ax.pen_dyn = dyn_bits
        ax.score = ll / LN2 - emis_bits - dyn_bits
        if best is None or ax.score > best.score:
            best = ax
    return best


def _partitions(items):
    items = list(items)
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    out = []
    for p in _partitions(rest):
        for i in range(len(p)):
            out.append(p[:i] + [p[i] + [first]] + p[i + 1:])
        out.append(p + [[first]])
    return out


# ======================================================================
# Организм
# ======================================================================

SLEEP_EVERY = 100    # ритм сна — расписание петли, не знание о мире


class Organism:
    def __init__(self, curious, object_actions, ctx_fn, truth_fn=None,
                 entities_fn=None, seed=0):
        self.curious = curious
        # ранг объявления действий (порядок анкеты — структурный, переживает
        # переименование): канонический порядок для _structure вместо
        # лексикографии слов мира (F2/N14)
        self._act_rank = {a: i for i, a in enumerate(object_actions)}
        self.object_actions = set(object_actions)
        self.ctx_fn = ctx_fn
        self.truth_fn = truth_fn or (lambda w, t: None)
        self.entities_fn = entities_fn or (lambda obs: {})
        self.rng = random.Random(seed)
        self.mem = Ruleset()
        self.records = []
        self.hist = defaultdict(lambda: defaultdict(list))  # obj -> a -> [(r, t)]
        self.axes = []            # принятые судьёй тайные оси
        self.problem_actions = set()
        self.R = set()
        self.steps = 0
        self._bel = {}            # (i_оси, obj) -> (belief, t) — кэш фильтра

    # ---------- belief ----------

    def _timeline(self, i, obj):
        ax = self.axes[i]
        tl = [(t, a, r) for a in ax.actions
              for (r, t) in self.hist.get(obj, {}).get(a, ())]
        tl.sort()
        return tl

    def belief(self, i, obj, t_query):
        """Уверенность по оси i для объекта в момент t_query."""
        ax = self.axes[i]
        cached = self._bel.get((i, obj))
        if cached is None:
            tl = [(t, a, r) for (t, a, r) in self._timeline(i, obj)
                  if t <= t_query]
            if not tl:
                return list(ax.pi)
            bel = ax.filter(tl, tl[-1][0])
            self._bel[(i, obj)] = (bel, tl[-1][0])
            cached = self._bel[(i, obj)]
        bel, t0 = cached
        return ax.evolve(bel, t_query - t0) if t_query > t0 else list(bel)

    def _bel_update(self, obj, a, r):
        """Вызывается ДО записи наблюдения в hist: поднимает уверенность
        из прошлого (кэш или фильтр по истории) и применяет «увидел»."""
        for i, ax in enumerate(self.axes):
            if a not in ax.actions:
                continue
            bel = self.belief(i, obj, self.steps)
            self._bel[(i, obj)] = (ax.see(bel, a, r), self.steps)

    def _labels_now(self, obj):
        """Метки состояний по текущему belief — только уверенные (>½:
        ответ вероятнее правильный, чем нет)."""
        out = set()
        for i, ax in enumerate(self.axes):
            bel = self.belief(i, obj, self.steps)
            m = max(bel)
            if m > 0.5:
                out.add((f"class{i}", f"c{bel.index(m)}"))
        return out

    def contextualize(self, ctx0, obj, ents):
        ctx = ctx0
        if obj is not None:
            ctx = ctx | self._labels_now(obj)
        if self.R and obj is not None:
            live = any(o == obj and ents.get(e) == st
                       for (o, e, st) in self.R)
            ctx = ctx | {("linked_live", int(live))}
        return ctx

    # ---------- любопытство: верни уверенность ----------

    def _gain(self, ctx, obj, a):
        """Ожидаемый прирост уверенности: вернуть уверенность о состоянии
        объекта + заполнить строку таблицы ответов (строка заполняется
        только при известном состоянии — знание атрибутируется)."""
        g = 0.0
        for i, ax in enumerate(self.axes):
            if a in ax.actions and obj is not None:
                bel = self.belief(i, obj, self.steps)
                mb = max(bel)
                s = bel.index(mb)
                row = sum(ax.emis.get((s, a), {}).values())
                g = max(g, (1.0 - mb) + mb / (1.0 + row))
        if obj is not None and a in self.object_actions:
            # со-измерение: пока есть необъяснённые действия, знание
            # атрибутируется только парами наблюдений на одном объекте —
            # незаполненная клетка (объект, действие) интересна максимально
            seen = set(self.hist.get(obj, ()))
            if a not in seen and seen and (a in self.problem_actions
                                           or self.problem_actions & seen):
                g = 1.0
        return max(g, self.mem.uncertainty(ctx, a))

    # ---------- петля ----------

    def act(self, w, ep):
        obs = w.observe()
        ents = {(ep, n): s for n, s in self.entities_fn(obs).items()}
        space = w.action_space()
        if self.curious:
            weights = []
            for a, t, kw in space:
                obj = (ep, t) if a in self.object_actions else None
                ctx = self.contextualize(self.ctx_fn(obs, a, t, kw), obj, ents)
                weights.append(self._gain(ctx, obj, a))
            if sum(weights) > 0:
                a, t, kw = self.rng.choices(space, weights=weights)[0]
            else:
                a, t, kw = self.rng.choice(space)
        else:
            a, t, kw = self.rng.choice(space)

        obj = (ep, t) if a in self.object_actions else None
        ctx0 = self.ctx_fn(obs, a, t, kw)
        truth = self.truth_fn(w, t) if obj else None
        tr = w.step(a, t, **kw)
        outcome = (tr["result"], tuple(sorted({e[0] for e in tr["effects"]})))
        self.records.append({"obj": obj, "ctx": ctx0, "action": a,
                             "outcome": outcome, "truth": truth,
                             "ents": ents, "effects": tr["effects"],
                             "step": self.steps})
        if obj is not None:
            self._bel_update(obj, a, outcome[0])
            self.hist[obj][a].append((outcome[0], self.steps))
        self.steps += 1
        if self.steps % SLEEP_EVERY == 0:
            self.sleep()

    # ---------- сон ----------

    def _axis_timelines(self, actions):
        tls = {}
        for obj, by_a in self.hist.items():
            tl = [(t, a, r) for a in actions
                  for (r, t) in by_a.get(a, ())]
            if tl:
                tl.sort()
                tls[obj] = tl
        return tls

    def _fit_block(self, actions):
        tls = self._axis_timelines(actions)
        if not tls:
            return None
        k_max = min(4, max(len(rs) for rs in
                           self._block_results(actions).values()) + 1)
        best = None
        for k in range(1, max(2, k_max) + 1):
            ax = fit_axis(actions, tls, k)
            if best is None or ax.score > best.score:
                best = ax
        return best

    def _block_results(self, actions):
        rs = defaultdict(set)
        for by_a in self.hist.values():
            for a in actions:
                for (r, _t) in by_a.get(a, ()):
                    rs[a].add(r)
        return rs

    def _fit_block_min2(self, actions):
        """Лучшая ось блока при k >= 2 (гипотеза «состояние есть»)."""
        tls = self._axis_timelines(actions)
        if not tls:
            return None
        k_max = min(4, max(len(rs) for rs in
                           self._block_results(actions).values()) + 1)
        best = None
        for k in range(2, max(2, k_max) + 1):
            ax = fit_axis(actions, tls, k)
            if best is None or ax.score > best.score:
                best = ax
        return best

    def _structure(self, actions):
        """Предложение структур: локальный счёт цепочек выдвигает
        кандидатов, но приговор выносит глобальный судья (sleep) — он
        видит и повторы противоречий, и видимые атрибуты через правила.
        Кандидаты: лучшее разбиение по локальному счёту; оно же с
        принудительными осями k>=2; одна общая ось на все симптомы."""
        acts = sorted(actions, key=lambda a: (self._act_rank.get(
            a, len(self._act_rank)), a))    # ранг анкеты, не алфавит (F2)
        memo = {}

        def fit(block, min2=False):
            key = (tuple(block), min2)
            if key not in memo:
                memo[key] = (self._fit_block_min2(block) if min2
                             else self._fit_block(block))
            return memo[key]
        def part_score(part):
            return sum((lambda ax: ax.score if ax else 0.0)(fit(b))
                       for b in part)

        if len(acts) <= 4:
            parts = _partitions(acts)
        else:
            cur = [[a] for a in acts]
            merged = True
            while merged and len(cur) > 1:
                merged = False
                base = part_score(cur)
                for i in range(len(cur)):
                    for j in range(i + 1, len(cur)):
                        cand = ([b for k_, b in enumerate(cur)
                                 if k_ not in (i, j)]
                                + [cur[i] + cur[j]])
                        if part_score(cand) > base:
                            cur, merged = cand, True
                            break
                    if merged:
                        break
            parts = [cur]
        best_part, best_axes, best_score = None, [], -math.inf
        for part in parts:
            axes, score = [], 0.0
            for block in part:
                ax = fit(block)
                if ax is None:
                    continue
                score += ax.score
                if ax.k >= 2:
                    axes.append(ax)
            if score > best_score:
                best_part, best_axes, best_score = part, axes, score
        candidates = [best_axes]
        if best_part is not None:
            forced = [fit(b, min2=True) for b in best_part]
            forced = [ax for ax in forced if ax is not None]
            candidates.append(forced)
        joint = fit(acts, min2=True)
        if joint is not None:
            candidates.append([joint])
        # дедупликация по структуре
        seen, out = set(), []
        for cand in candidates:
            key = tuple(sorted((ax.actions, ax.k) for ax in cand))
            if key not in seen:
                seen.add(key)
                out.append(cand)
        return out

    def _smoothed_labels(self, axes):
        """(i, obj) -> [(t, γ)] — сглаженные состояния для переразметки."""
        lab = {}
        for i, ax in enumerate(axes):
            for obj in self.hist:
                tl = [(t, a, r) for a in ax.actions
                      for (r, t) in self.hist[obj].get(a, ())]
                if not tl:
                    continue
                tl.sort()
                gammas, _ll = ax.smooth(tl)
                lab[(i, obj)] = (ax, [(t, g) for (t, _a, _r), g
                                      in zip(tl, gammas)])
        return lab

    @staticmethod
    def _label_at(entry, t):
        ax, line = entry
        (t0, g0) = min(line, key=lambda p: abs(p[0] - t))
        bel = ax.evolve(g0, abs(t - t0)) if t != t0 else g0
        m = max(bel)
        return (m, bel.index(m))

    def _relabel(self, axes):
        lab = self._smoothed_labels(axes)
        mem = Ruleset()
        for r in self.records:
            ctx = r["ctx"]
            obj = r["obj"]
            if obj is not None:
                for i in range(len(axes)):
                    entry = lab.get((i, obj))
                    if entry:
                        m, s = self._label_at(entry, r["step"])
                        if m > 0.5:
                            ctx = ctx | {(f"class{i}", f"c{s}")}
                if self.R:
                    live = any(o == obj and r["ents"].get(e) == st
                               for (o, e, st) in self.R)
                    ctx = ctx | {("linked_live", int(live))}
            mem.episodes[(ctx, r["action"], r["outcome"])] += 1
        mem.consolidate()
        return mem

    def _axes_bits(self, axes):
        """Штраф Оккама принятых осей в глобальном счёте: динамика
        (кубики, цели ухода, пи) и выбор k. Таблицу ответов не считаем —
        её уже оплачивают правила class->исход, дважды не берём."""
        return sum(ax.pen_dyn for ax in axes)

    def _total_bits(self, axes, mem):
        return mem.total_bits() + self._axes_bits(axes)

    def sleep(self):
        # 1. переразметка и остаток при текущей структуре
        self.mem = self._relabel(self.axes)
        cur_bits = self._total_bits(self.axes, self.mem)
        residue = [r for r in self.records
                   if (lambda p: p is None or p["outcome"] != r["outcome"])(
                       self.mem.predict(
                           self.contextualize(r["ctx"], r["obj"], r["ents"]),
                           r["action"]))]
        self.problem_actions |= {r["action"] for r in residue
                                 if r["obj"] is not None}
        # 2. кандидаты-структуры: предложения от локального счёта,
        # приговор — глобальные биты
        if self.problem_actions:
            cands = self._structure(self.problem_actions)
            for cand_axes in cands:
                cand_mem = self._relabel(cand_axes)
                cand_bits = self._total_bits(cand_axes, cand_mem)
                if cand_bits < cur_bits:
                    self.axes, self.mem = cand_axes, cand_mem
                    cur_bits = cand_bits
        # 3. реляционная абдукция — под тем же судьёй
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
            old_R = self.R
            self.R = self.R | cand_R
            cand_mem = self._relabel(self.axes)
            if self._total_bits(self.axes, cand_mem) < cur_bits:
                self.mem = cand_mem
            else:
                self.R = old_R
        self._bel.clear()   # оси могли смениться — фильтры пересчитаются

    # ---------- экзамен ----------

    def infer(self, action, result):
        """Belief по одному зонду на свежем объекте: незнание (равномерный
        прайор) × таблица ответов, шаг таяния до вопроса; MAP, если ответ
        вероятнее правильный, чем нет."""
        for i, ax in enumerate(self.axes):
            if action not in ax.actions:
                continue
            bel = ax.see(list(ax.pi), action, result)
            bel = ax.evolve(bel, 1)
            m = max(bel)
            if m > 0.5:
                return (f"class{i}", f"c{bel.index(m)}")
        return None

    # ---------- витрина (диагностика runworld; знанием не является) ----------

    @property
    def groups(self):
        out = []
        for i, ax in enumerate(self.axes):
            clusters = []
            for s in range(ax.k):
                msig = {}
                for a in ax.actions:
                    c = ax.emis.get((s, a))
                    if c:
                        msig[a] = max(c, key=c.get)
                objs = []
                for obj in self.hist:
                    tl = self._timeline(i, obj)
                    if tl:
                        bel = ax.filter(tl, tl[-1][0])
                        if bel.index(max(bel)) == s:
                            objs.append(obj)
                clusters.append((msig, objs))
            out.append({"attr": f"class{i}", "actions": set(ax.actions),
                        "clusters": clusters})
        return out

    @property
    def rates(self):
        out = {}
        for ax in self.axes:
            for s in range(ax.k):
                for a in ax.actions:
                    c = ax.emis.get((s, a))
                    if c:
                        out[(a, max(c, key=c.get))] = ax.haz[s]
        return out

    @property
    def horizon(self):
        hs = [(h, ax.n_pairs) for ax in self.axes for h in ax.haz]
        if not hs:
            return None
        h, n = max(hs)
        if h < 1.0 / max(2, n):    # неотличимо от нуля на этих данных
            return None
        return max(1, int(round(math.log(0.5) / math.log(1.0 - h))))
