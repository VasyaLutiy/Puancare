"""koopman — структурная подпись оси через спектр оператора Купмана.

Форма скрытой динамики оси = спектр её матрицы переходов T. Для конечной
цепи оператор Купмана на наблюдаемых (функциях от состояния) — это само T;
его собственные значения:
  λ₁ = 1          — стационар (сохранение вероятности), есть всегда
  λ₂, λ₃, ...     — моды релаксации; |λ| ∈ [0,1), τ = -1/ln|λ| — время
                    затухания памяти о прошлом; Im(λ)≠0 — скрытая осцилляция
Спектр перестановочно-инвариантен ПО ПОСТРОЕНИЮ (не зависит от нумерации
состояний) — то, что sig.py добывал руками через sorted(), тут даром. И k
входит не целым числом, а размерностью спектра: у k=2-формы одна мода
релаксации, у k=3 — две; недостающие моды = «быстрые/отсутствуют» (0).

persist/directed/horizon из sig — мутные проекции этого спектра:
  persist = 1 − 2ab/(a+b),  а купмановское  λ₂ = 1 − (a+b).
Разные функции: persist мешает темп затухания со стационарной асимметрией.

Наблюдаемая сторона (как зонды читают состояние) = та же резкость
I(состояние;исход)/H, что в sig: эмиссии — это карта наблюдаемых, спектр
динамики её не заменяет, поэтому берём как есть.

Считается по org.axes (что организм ВЫУЧИЛ), без единого слова из yaml.
"""

import sys
import math
from math import fsum

import sig                          # best_axis, signature (для сравнения)


# --- спектр без BLAS -------------------------------------------------------
# np.linalg.eigvals удалён: LAPACK-бэкенды дают разные последние биты
# (доказано: OpenBLAS-Linux vs Accelerate-мак расходятся на 1-5 ulp на
# одной и той же матрице), т.е. подпись формы зависела от сборки numpy.
# Для k<=4 спектр считается аналитически: у стохастической T всегда λ₁=1;
# остальные — корни дефлированного характеристического многочлена
# (Фаддеев—Леверье + бисекция для кубики + точная квадратура). Только
# +,-,*,/,sqrt — единственные операции, которые IEEE 754 обязывает
# округлять точно ⇒ бит-в-бит на любой машине ПО ПОСТРОЕНИЮ.

def _cabs(z):
    """|z| через sqrt(re²+im²): abs(complex) ходит в hypot() из libm,
    который не обязан совпадать между платформами; sqrt — обязан."""
    return math.sqrt(z.real * z.real + z.imag * z.imag)


def _charpoly(T):
    """det(λI−T) = λ^k + c₁λ^(k−1) + ... + c_k (Фаддеев—Леверье)."""
    k = len(T)
    c = [1.0]
    M = [[float(i == j) for j in range(k)] for i in range(k)]      # M₁ = I
    for m in range(1, k + 1):
        if m > 1:                                  # M_m = T·M_{m-1} + c·I
            TM = [[fsum(T[i][x] * M[x][j] for x in range(k))
                   for j in range(k)] for i in range(k)]
            M = [[TM[i][j] + (c[-1] if i == j else 0.0)
                  for j in range(k)] for i in range(k)]
        TMm = [[fsum(T[i][x] * M[x][j] for x in range(k))
                for j in range(k)] for i in range(k)]
        c.append(-fsum(TMm[i][i] for i in range(k)) / m)
    return c


def _quad_roots(p, q):
    """Корни λ² + pλ + q = 0 (вещ. p, q)."""
    D = p * p - 4.0 * q
    if D >= 0.0:
        sd = math.sqrt(D)
        return [complex((-p + sd) / 2.0, 0.0), complex((-p - sd) / 2.0, 0.0)]
    sd = math.sqrt(-D)
    return [complex(-p / 2.0, sd / 2.0), complex(-p / 2.0, -sd / 2.0)]


def _cubic_roots(a, b, c):
    """Корни λ³ + aλ² + bλ + c = 0. Вещественный корень — бисекцией
    (у цепи |λ|<=1, поэтому p(-2)<0<p(2) гарантированно; только
    арифметика — никаких cbrt/acos из libm), остальные — дефляцией
    в квадратуру."""
    def p(x):
        return ((x + a) * x + b) * x + c
    lo, hi = -2.0, 2.0
    neg = p(lo) < 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if (p(mid) < 0.0) == neg:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    p2 = a + r                       # λ³+aλ²+bλ+c = (λ−r)(λ²+p2·λ+q2)
    q2 = b + r * p2
    return [complex(r, 0.0)] + _quad_roots(p2, q2)


def spectrum(ax):
    """Собственные значения T, доминантный (λ₁=1) первым. Аналитически,
    без numpy/BLAS — см. блок выше."""
    T = ax.T()
    k = len(T)
    if k == 1:
        return [complex(1.0, 0.0)]
    c = _charpoly(T)
    b = [1.0]                                     # деление на (λ − 1)
    for i in range(1, k):
        b.append(c[i] + b[-1])
    if k == 2:
        roots = [complex(-b[1], 0.0)]
    elif k == 3:
        roots = _quad_roots(b[1], b[2])
    else:
        roots = _cubic_roots(b[1], b[2], b[3])
    ev = [complex(1.0, 0.0)] + roots
    return sorted(ev, key=lambda z: -_cabs(z))


def signature(ax):
    k = ax.k
    ev = spectrum(ax)
    sub = ev[1:]                                  # выбросить λ₁≈1 (стационар)
    spec = sorted(float(round(_cabs(z), 3)) for z in sub)  # магнитуды мод рел.
    osc = sorted(float(round(abs(z.imag), 3)) for z in sub)  # осцилляция мод
    # время релаксации медленнейшей моды — горизонт памяти (для показа)
    slow = float(max((_cabs(z) for z in sub), default=0.0))
    tau = (-1.0 / math.log(slow)) if 0 < slow < 1 else 0.0
    # наблюдаемая сторона: резкость эмиссий (как sig), стационар для веса
    pi = sig.stationary(ax)
    Hs = -fsum(p * math.log(p) for p in pi if p > 0) or 1e-12
    sharp, ncards = [], []
    for a in ax.actions:
        results = ax.res_a[a]
        Pr = {r: fsum(pi[s] * ax.emis_p(s, a, r) for s in range(k))
              for r in results}
        I = 0.0
        for s in range(k):
            for r in results:
                p_rs = ax.emis_p(s, a, r)
                if p_rs > 0 and Pr[r] > 0:
                    I += pi[s] * p_rs * math.log(p_rs / Pr[r])
        sharp.append(round(I / Hs, 3))
        ncards.append(len(results))
    return {"k": k, "spec": spec, "osc": osc, "tau": round(tau, 1),
            "sharp": sorted(sharp), "ncards": sorted(ncards)}


COMPS = ("spec", "osc", "sharp", "ncards")     # компоненты подписи


def _vals(S, c):
    return [float(v) for v in S[c]]


def _padL1(x, y):
    n = max(len(x), len(y))
    x = x + [0.0] * (n - len(x))
    y = y + [0.0] * (n - len(y))
    return fsum(abs(a - b) for a, b in zip(sorted(x), sorted(y)))


def _comp_d(A, B, c):
    return _padL1(_vals(A, c), _vals(B, c))


def _ks_norm(S):
    """Ключи "ks" -> int. JSON-раундтрип превращает int-ключи в строки —
    без нормализации пересечение срезов живой подписи ({2,3,4}) и карточки
    ({'2','3','4'}) пусто, и мульти-k сравнение молча выключается (улов
    арки ворот: карточки полки сравнивались фолбэком)."""
    if "ks" in S and any(isinstance(k, str) for k in S["ks"]):
        S = dict(S)
        S["ks"] = {int(k): v for k, v in S["ks"].items()}
    return S


def comp_dist(A, B, c):
    """Дистанция ОДНОЙ компоненты двух подписей, в её собственных единицах.
    Мульти-k: среднее по общим k (сравнение только при совпадающем k)."""
    A, B = _ks_norm(A), _ks_norm(B)
    if "ks" in A and "ks" in B:
        shared = sorted(set(A["ks"]) & set(B["ks"]))
        if shared:
            return fsum(_comp_d(A["ks"][k], B["ks"][k], c)
                        for k in shared) / len(shared)
    return _comp_d(A, B, c)


def _pair_d(A, B, w):
    """Расстояние двух подписей. Мульти-k (поле "ks"): среднее по ОБЩИМ k
    сравнений при СОВПАДАЮЩЕМ k — паддинга между разными k нет по
    построению, флип выбора k теряет силу (арка линзы: волк f0006 и ложный
    отказ b04-m04 были взрывом паддинга при k-несовпадении). Одиночные
    подписи сравниваются как раньше."""
    return fsum(w[c] * comp_dist(A, B, c) for c in COMPS)


def calibrate(sigs):
    """Веса компонент ИЗ БИБЛИОТЕКИ, а не из констант: 1/средний попарный
    зазор компоненты (выравнивание масштабов). Два правила, оба из данных:
      * нулевой разброс → нулевой вес (как раньше);
      * ШУМОВОЙ ПОЛ: если подписи несут поле "noise" (внутримировая
        болтанка компоненты между половинами данных), компонента получает
        вес только когда межмировой разброс БОЛЬШЕ средней болтанки —
        иначе калибровка инвертирует шум почти-константы (osc: округление
        0.001 давало вес 1000 и травило радиус)."""
    sigs = [_ks_norm(s) for s in sigs]
    pairs = [(a, b) for i, a in enumerate(sigs) for b in sigs[i + 1:]]
    w = {}
    for c in COMPS:
        if pairs:
            m = fsum(comp_dist(a, b, c) for a, b in pairs) / len(pairs)
        else:
            m = 0.0
        noise = [s["noise"][c] for s in sigs
                 if isinstance(s.get("noise"), dict) and c in s["noise"]]
        # пол = МАКС болтанка по членам: если разброс библиотеки объясним
        # джиттером одного её мира — компоненте веры нет (среднее размывают
        # чистые миры, а весь разброс может создавать один шумный)
        floor = max(noise) if noise else 0.0
        w[c] = 1.0 / m if (m > 1e-9 and m > floor) else 0.0
    return w


def dist(A, B, w=None):
    """Расстояние форм. w=None → равные веса (без ручной настройки);
    calibrate(библиотека) даёт веса из данных."""
    if w is None:
        w = {c: 1.0 for c in COMPS}
    return _pair_d(A, B, w)


def _matrix(sigs, ws, metric):
    print(" " * 11 + "".join(f"{w:>10s}" for w in ws))
    for a in ws:
        row = "".join(f"{metric(sigs[a], sigs[b]):10.2f}" for b in ws)
        print(f"{a:10s} {row}")


def _neighbors(sigs, ws, metric):
    for a in ws:
        near = min((b for b in ws if b != a),
                   key=lambda b: metric(sigs[a], sigs[b]))
        print(f"  {a:10s} ← ближе всего → {near}")


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["batteries", "car", "bird"]
    ksigs, ssigs = {}, {}
    print("=== купмановские подписи (спектр выученных осей, без слов) ===")
    for w in worlds:
        ax = sig.best_axis(w)
        if ax is None:
            print(f"{w:10s}: ось не выучена"); continue
        ks = ksigs[w] = signature(ax)
        ssigs[w] = sig.signature(ax)            # та же ось — честное сравнение
        ev = spectrum(ax)
        evs = " ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in ev)
        print(f"{w:10s}: k={ks['k']}  спектр=[{evs}]  |λ_sub|={ks['spec']}  "
              f"осц={ks['osc']}  τ={ks['tau']}  резк={ks['sharp']}  "
              f"исх={ks['ncards']}")

    ws = [w for w in worlds if w in ksigs]
    if len(ws) >= 2:
        cw = calibrate(ksigs[w] for w in ws)
        kd = lambda a, b: dist(a, b, cw)
        print(f"\n(веса калиброваны по библиотеке: "
              f"{ {c: round(v, 2) for c, v in cw.items()} })")
        print("=== KOOPMAN: попарные расстояния (форма) ===")
        _matrix(ksigs, ws, kd)
        print("\n=== KOOPMAN: ближайший сосед ===")
        _neighbors(ksigs, ws, kd)
        print("\n=== SIG (рукодельный, для сравнения): ближайший сосед ===")
        _neighbors(ssigs, ws, sig.dist)
