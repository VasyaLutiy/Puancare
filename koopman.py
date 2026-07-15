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

import numpy as np

import sig                          # best_axis, signature (для сравнения)


def spectrum(ax):
    """Собственные значения T, доминантный (|λ|≈1) первым."""
    ev = np.linalg.eigvals(np.array(ax.T(), dtype=float))
    return sorted(ev, key=lambda z: -abs(z))


def signature(ax):
    k = ax.k
    ev = spectrum(ax)
    sub = ev[1:]                                  # выбросить λ₁≈1 (стационар)
    spec = sorted(float(round(abs(z), 3)) for z in sub)  # магнитуды мод релакс.
    osc = sorted(float(round(abs(z.imag), 3)) for z in sub)  # осцилляция мод
    # время релаксации медленнейшей моды — горизонт памяти (для показа)
    slow = float(max((abs(z) for z in sub), default=0.0))
    tau = (-1.0 / math.log(slow)) if 0 < slow < 1 else 0.0
    # наблюдаемая сторона: резкость эмиссий (как sig), стационар для веса
    pi = sig.stationary(ax)
    Hs = -sum(p * math.log(p) for p in pi if p > 0) or 1e-12
    sharp, ncards = [], []
    for a in ax.actions:
        results = ax.res_a[a]
        Pr = {r: sum(pi[s] * ax.emis_p(s, a, r) for s in range(k))
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
    return sum(abs(a - b) for a, b in zip(sorted(x), sorted(y)))


def calibrate(sigs):
    """Веса компонент ИЗ БИБЛИОТЕКИ, а не из констант: 1/средний попарный
    зазор компоненты по членам (выравнивание масштабов — у каждой компоненты
    средний вклад 1). Компонента, не различающая членов библиотеки (нулевой
    разброс), веса не получает. Никаких ручных 3.0/2.0/0.5."""
    sigs = list(sigs)
    pairs = [(a, b) for i, a in enumerate(sigs) for b in sigs[i + 1:]]
    w = {}
    for c in COMPS:
        m = (sum(_padL1(_vals(a, c), _vals(b, c)) for a, b in pairs)
             / len(pairs)) if pairs else 0.0
        w[c] = 1.0 / m if m > 1e-9 else 0.0
    return w


def dist(A, B, w=None):
    """Расстояние форм. w=None → равные веса (без ручной настройки);
    calibrate(библиотека) даёт веса из данных. Заметь: НЕТ отдельного
    слагаемого по k — оно входит через длину спектра (padL1 добивает нулём
    недостающие быстрые моды)."""
    if w is None:
        w = {c: 1.0 for c in COMPS}
    return sum(w[c] * _padL1(_vals(A, c), _vals(B, c)) for c in COMPS)


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
