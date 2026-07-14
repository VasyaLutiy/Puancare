"""sig — структурная подпись выученной оси, БЕЗ единого слова из yaml.

Подпись = форма скрытой цепочки + роли действий, всё перестановочно- и
словарь-инвариантно:
  k             число состояний
  persist       средняя вероятность остаться за шаг (темп мира)
  directed      однонаправленность потока (0 обратимый ⇄, 1 односторонний →)
  sharp[]       резкость каждого действия как прибора (I(состояние;исход)/H),
                отсортировано → не зависит от имён действий
  ncards[]      число исходов у каждого действия, отсортировано

Считается по org.axes (что организм ВЫУЧИЛ), не по истине мира.
"""

import sys
import math

import runworld
from organism2 import Organism
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def stationary(ax, iters=300):
    p = list(ax.pi)
    T = ax.T()
    for _ in range(iters):
        p = [sum(p[s] * T[s][j] for s in range(ax.k)) for j in range(ax.k)]
        z = sum(p) or 1.0
        p = [x / z for x in p]
    return p


def signature(ax):
    k = ax.k
    T = ax.T()
    pi = stationary(ax)
    persist = sum(pi[s] * T[s][s] for s in range(k))          # темп
    # направленность: асимметрия скоростей ухода. Потоковая версия
    # (поток i->j vs j->i при стационаре) при k=2 тождественно 0 —
    # детальный баланс двух состояний выполняется всегда, мат-факт.
    # Читаем саму разброску hazard'ов: сток имеет состояние, из которого
    # почти не уходят (haz~0), обратимая цепь — сбалансированные haz.
    hz = ax.haz
    directed = ((max(hz) - min(hz)) / (max(hz) + min(hz))
                if max(hz) > 0 else 0.0)
    # роли действий: резкость = I(состояние;исход)/H(состояние)
    Hs = -sum(p * math.log(p) for p in pi if p > 0) or 1e-12
    sharp, ncards = [], []
    for a in ax.actions:
        results = ax.res_a[a]
        Pr = {r: sum(pi[s] * ax.emis_p(s, a, r) for s in range(k))
              for r in results}
        I = 0.0
        for s in range(k):
            for r in results:
                pr_rs = ax.emis_p(s, a, r)
                if pr_rs > 0 and Pr[r] > 0:
                    I += pi[s] * pr_rs * math.log(pr_rs / Pr[r])
        sharp.append(round(I / Hs, 3))
        ncards.append(len(results))
    return {"k": k, "persist": round(persist, 3),
            "directed": round(directed, 3),
            "sharp": sorted(sharp), "ncards": sorted(ncards)}


def best_axis(world, budget=2000, curious=False):
    spec = load_spec(f"worlds/{world}.yaml")
    org = runworld.live(make_glue(spec), curious, budget)
    return max(org.axes, key=lambda a: a.n_obs) if org.axes else None


def dist(A, B):
    def padL1(x, y):
        n = max(len(x), len(y))
        x = x + [0.0] * (n - len(x))
        y = y + [0.0] * (n - len(y))
        return sum(abs(a - b) for a, b in zip(sorted(x), sorted(y)))
    return (2.0 * abs(A["k"] - B["k"])
            + 3.0 * abs(A["persist"] - B["persist"])
            + 3.0 * abs(A["directed"] - B["directed"])
            + padL1(A["sharp"], B["sharp"])
            + 0.5 * padL1([float(c) for c in A["ncards"]],
                          [float(c) for c in B["ncards"]]))


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["batteries", "car", "bike", "bird"]
    sigs = {}
    print("=== структурные подписи (из выученных осей, без слов) ===")
    for w in worlds:
        ax = best_axis(w)
        if ax is None:
            print(f"{w:10s}: ось не выучена"); continue
        s = sigs[w] = signature(ax)
        print(f"{w:10s}: k={s['k']}  устойч={s['persist']:.3f}  "
              f"направл={s['directed']:.3f}  резкость={s['sharp']}  "
              f"исходы={s['ncards']}")
    print("\n=== попарные расстояния (форма) ===")
    ws = [w for w in worlds if w in sigs]
    print(" " * 11 + "".join(f"{w:>10s}" for w in ws))
    for a in ws:
        row = "".join(f"{dist(sigs[a], sigs[b]):10.2f}" for b in ws)
        print(f"{a:10s} {row}")
    print("\n=== ближайший сосед каждого ===")
    for a in ws:
        near = min((b for b in ws if b != a), key=lambda b: dist(sigs[a], sigs[b]))
        print(f"  {a:10s} ← ближе всего → {near}")
