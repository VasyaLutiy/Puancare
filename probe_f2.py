"""probe_f2 — течёт ли лексикографический порядок слов в ядро (F2).

Метод: дневник собирается ОДИН раз (случайный агент, сид 0), затем все
слова (объекты, действия, результаты) переименовываются биекцией и ядро
(select_k / fit_axis / кластеры-старт) запускается на ТОМ ЖЕ дневнике.
Немота требует: результаты совпадают бит-в-бит. Варианты:

  ident — тождество (санити: должен совпасть сам с собой)
  keep  — переименование, СОХРАНЯЮЩЕЕ лексикографический порядок слов
  flip  — переименование, ОБРАЩАЮЩЕЕ порядок (держи-аут из F2)

Уровни сравнения (от строгого к грубому):
  биты   — ax.score всех k, точное сравнение float
  подпись — koopman.signature (путь узнавания, округлена до 3 знаков)
  k      — выбор select_k
  метки  — разбиение объектов по состояниям (с точн. до перестановки)
  старт  — порядок кластеров co-подписей (restart 0 EM)

Отдельно: --hashcheck — ТЕ ЖЕ слова, разные PYTHONHASHSEED (порядок
обхода set(строк) в res_a[a]) — детерминизм между процессами.

  python3 probe_f2.py [мир=last/apples] [бюджет=600]
"""

import json
import os
import subprocess
import sys

import fastpath
import koopman
from organism2 import _cluster_signatures

KS = (2, 3, 4)


# ---------------------------------------------------------- переименование

def _bij(words, mode, tag):
    ws = sorted(words)
    n = len(ws)
    if mode == "ident":
        return {w: w for w in ws}
    if mode == "keep":
        return {w: f"{tag}{i:04d}" for i, w in enumerate(ws)}
    if mode == "flip":
        return {w: f"{tag}{n - 1 - i:04d}" for i, w in enumerate(ws)}
    raise ValueError(mode)


def rename_diary(tls, acts, mode):
    """Биекции строятся по трём словарям мира (объекты/действия/результаты);
    возвращает (tls', acts', inv_obj) — inv_obj для отображения меток назад."""
    objs, ress = set(), set()
    for (_ep, name), tl in tls.items():
        objs.add(name)
        for (_t, _a, r) in tl:
            ress.add(r)
    mo = _bij(objs, mode, "о")
    ma = _bij(acts, mode, "д")
    mr = _bij(ress, mode, "р")
    tls2 = {(ep, mo[name]): [(t, ma[a], mr[r]) for (t, a, r) in tl]
            for (ep, name), tl in tls.items()}
    acts2 = tuple(sorted(ma[a] for a in acts))   # как в fastpath
    inv_obj = {v: k for k, v in mo.items()}
    return tls2, acts2, inv_obj


# ---------------------------------------------------------- прогон ядра

def core(tls, acts):
    """Прогнать ядро на дневнике, вернуть сравнимые артефакты."""
    k, full, _fa, _fb = fastpath.select_k(acts, tls, KS)
    scores = {kk: (full[kk].score if full.get(kk) is not None else None)
              for kk in KS}
    sig = koopman.signature(full[k])
    # кластеры co-подписей — стартовая точка EM (restart 0)
    sigs, first_seen = {}, {}
    for obj, tl in tls.items():
        s = {}
        for (t, a, r) in tl:
            s[a] = r
            if obj not in first_seen or t < first_seen[obj]:
                first_seen[obj] = t
        if s:
            sigs[obj] = s
    clusters = [tuple(objs)
                for _m, objs in _cluster_signatures(sigs, first_seen)]
    # финальные метки объектов (argmax сглаженной gamma на конце жизни)
    ax = full[k]
    labels = {}
    for obj, tl in tls.items():
        gam, _ll = ax.smooth(tl)
        g = gam[-1]
        labels[obj] = max(range(ax.k), key=lambda s: g[s])
    return {"k": k, "scores": scores, "sig": sig,
            "clusters": clusters, "labels": labels}


def diary(world, budget):
    glue = fastpath.glue_of(world)
    org = fastpath._collect(glue, budget)
    acts = tuple(sorted(org.object_actions))
    return org._axis_timelines(acts), acts


# ---------------------------------------------------------- сравнение

def _partition(labels, inv_obj):
    """метки -> разбиение ИСХОДНЫХ имён по состояниям, без номеров состояний"""
    groups = {}
    for (ep, name), lab in labels.items():
        groups.setdefault(lab, set()).add((ep, inv_obj[name]))
    return sorted(frozenset(g) for g in groups.values())


def compare(base, other, inv_obj):
    rows = {}
    rows["k"] = base["k"] == other["k"]
    rows["биты score"] = all(
        (base["scores"][kk] is None) == (other["scores"][kk] is None)
        and (base["scores"][kk] is None
             or base["scores"][kk] == other["scores"][kk])
        for kk in KS)
    rows["подпись"] = base["sig"] == other["sig"]
    rows["метки (до перест.)"] = (
        _partition(base["labels"], {n: n for _e, n in base["labels"]})
        == _partition(other["labels"], inv_obj))
    oc = [tuple((ep, inv_obj[n]) for ep, n in cl) for cl in other["clusters"]]
    rows["старт-кластеры (порядок)"] = base["clusters"] == oc
    deltas = {kk: (None if base["scores"][kk] is None
                   or other["scores"][kk] is None
                   else abs(base["scores"][kk] - other["scores"][kk]))
              for kk in KS}
    return rows, deltas


# ---------------------------------------------------------- режимы запуска

def main(world, budget):
    print(f"=== F2-зонд: мир {world}, бюджет {budget}, сид 0 ===")
    tls, acts = diary(world, budget)
    n_obj = len(tls)
    n_obs = sum(len(tl) for tl in tls.values())
    print(f"дневник: {n_obj} объектов, {n_obs} наблюдений, "
          f"действия {list(acts)}")

    results = {}
    for mode in ("ident", "keep", "flip"):
        tls2, acts2, inv = rename_diary(tls, acts, mode)
        results[mode] = (core(tls2, acts2), inv)

    base = results["ident"][0]
    print(f"\nident: k={base['k']}, spec={base['sig']['spec']}, "
          f"scores={ {kk: (round(v, 6) if v is not None else None) for kk, v in base['scores'].items()} }")

    leak = False
    for mode in ("keep", "flip"):
        other, inv = results[mode]
        rows, deltas = compare(base, other, inv)
        print(f"\n--- {mode} ({'порядок сохранён' if mode == 'keep' else 'порядок ОБРАЩЁН'}) ---")
        for name, ok in rows.items():
            print(f"  {name:24s}: {'СОВПАЛО' if ok else 'РАЗОШЛОСЬ'}")
        ds = {kk: (f"{d:.3e}" if d else ("0" if d == 0.0 else "-"))
              for kk, d in deltas.items()}
        print(f"  |Δscore| по k: {ds}")
        if not all(rows.values()):
            leak = True
            if not rows["подпись"]:
                print(f"  подпись ident: {base['sig']}")
                print(f"  подпись {mode}:  {other['sig']}")

    print("\n=== ВЕРДИКТ ===")
    if leak:
        print("ТЕЧЁТ: переименование слов меняет результат ядра на том же "
              "дневнике.\nНемота «бит-в-бит» — свойство выборки изоморфов, "
              "не конструкции (F2 подтверждена экспериментом).")
    else:
        print("НЕ ТЕЧЁТ на этом мире/бюджете: все уровни совпали бит-в-бит.")
    return leak


def hashcheck_child(world, budget):
    tls, acts = diary(world, budget)
    r = core(tls, acts)
    out = {"k": r["k"],
           "scores": {kk: (v.hex() if v is not None else None)
                      for kk, v in r["scores"].items()},
           "sig": r["sig"]}
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


def hashcheck(world, budget):
    print(f"\n=== hash-зонд: ТЕ ЖЕ слова, PYTHONHASHSEED=0/1/2 "
          f"(порядок обхода set) ===")
    outs = []
    for hs in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        p = subprocess.run(
            [sys.executable, __file__, "--child", world, str(budget)],
            capture_output=True, text=True, env=env, check=True)
        outs.append(p.stdout.strip().splitlines()[-1])
    same = len(set(outs)) == 1
    print("биты score/подпись/k по трём hash-сидам: "
          + ("СОВПАЛИ" if same else "РАЗОШЛИСЬ"))
    if not same:
        for hs, o in zip(("0", "1", "2"), outs):
            print(f"  PYTHONHASHSEED={hs}: {o[:200]}")
    return not same


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:]]
    if argv and argv[0] == "--child":
        hashcheck_child(argv[1], int(argv[2]))
        sys.exit(0)
    world = argv[0] if argv else "last/apples"
    budget = int(argv[1]) if len(argv) > 1 else 600
    leak = main(world, budget)
    leak_hash = hashcheck(world, budget)
    sys.exit(1 if (leak or leak_hash) else 0)
