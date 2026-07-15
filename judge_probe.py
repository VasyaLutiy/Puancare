"""judge_probe — вскрытие битов судьи: куда девается локальный выигрыш оси.

hard1: форс-фит даёт +270 бит score при 2->3, спектр стабилен — а судья
sleep() ось не принимает. Локальный score = правдоподобие таймлайнов −
штрафы (fit_axis). Глобальный приговор = mem.total_bits() + pen_dyn, где
вся польза оси обязана материализоваться ЧЕРЕЗ ПРАВИЛА над жёсткими
метками: _relabel вешает class{i}=c_s только при уверенности сглаживания
m>0.5. Гипотеза: между осью и судьёй бутылочное горлышко
«мягкое знание -> жёсткая метка -> правило».

Инструмент ЧИТАЕТ счета ядра (_relabel, total_bits, pen_dyn), ничего не
меняя. Разложение по кандидатам без оси / k=2..4:
  биты правил | биты остатка | pen_dyn | ИТОГО (приговор судьи)
  покрытие метками (доля объектных эпизодов с class при m>0.5)
  локальный score оси — для сопоставления с вердиктом.

Запуск: python3 judge_probe.py <мир> <бюджет> [<мир> <бюджет> ...]
"""

import sys
from collections import defaultdict

from organism2 import fit_axis
from fastpath import _collect, glue_of


def honest_bits(mem, axes):
    """Пересчёт приговора с ГРУППОВЫМ остатком: min-кап _resid_cost валиден
    лишь для ОДНОЙ строки на (ctx, действие) — правила о прочих исходах
    того же ctx взаимоисключающи, их повторы платят полный сюрприз
    m*out_bits. (Диагностика гипотезы; ядро не тронуто.)"""
    bits = sum(mem._rule_cost(r["action"], r["conds"]) for r in mem.rules)
    groups = defaultdict(list)
    for (ctx, a, o), m in mem.episodes.items():
        p = mem.predict(ctx, a)
        if p is None or p["outcome"] != o:
            groups[(ctx, a)].append(m * mem.out_bits(a))
    for (ctx, a), full in groups.items():
        raw = mem._raw_cost(ctx, a)
        save = max((f - min(raw, f)) for f in full)   # кап — одной строке
        bits += sum(full) - save
    return bits + sum(ax.pen_dyn for ax in axes)


def autopsy(name, glue, budget):
    org = _collect(glue, budget)
    acts = tuple(sorted(org.object_actions))
    tls = org._axis_timelines(acts)
    obj_recs = [r for r in org.records if r["obj"] is not None]
    print(f"\n=== {name} @ бюджет {budget} "
          f"(эпизодов {len(org.records)}, объектных {len(obj_recs)}) ===")
    print(f"{'кандидат':>9} | {'правила':>8} | {'остаток':>8} | {'pen':>6} | "
          f"{'ИТОГО':>9} | {'ЧЕСТНЫЙ':>9} | {'метки%':>6} | {'правил':>6} | "
          f"{'score оси':>9}")
    base_total, base_honest = None, None
    for k in (None, 2, 3, 4):
        axes = [] if k is None else [fit_axis(acts, tls, k)]
        if k is not None and axes[0] is None:
            print(f"{f'k={k}':>9} | фит не собрался")
            continue
        mem = org._relabel(axes)
        rule_bits = sum(mem._rule_cost(r["action"], r["conds"])
                        for r in mem.rules)
        total_mem = mem.total_bits()
        resid_bits = total_mem - rule_bits
        pen = sum(ax.pen_dyn for ax in axes)
        total = total_mem + pen
        honest = honest_bits(mem, axes)
        # покрытие метками — тем же условием, что _relabel
        if axes:
            lab = org._smoothed_labels(axes)
            n_cov = 0
            for r in obj_recs:
                entry = lab.get((0, r["obj"]))
                if entry:
                    m, _s = org._label_at(entry, r["step"])
                    if m > 0.5:
                        n_cov += 1
            cov = 100.0 * n_cov / max(1, len(obj_recs))
            sc = f"{axes[0].score:9.1f}"
        else:
            cov, sc = 0.0, " базовый"
        if base_total is None:
            base_total, base_honest = total, honest
        verdict = ""
        if k is not None:
            v1 = "прин" if total < base_total else "ОТКАЗ"
            v2 = "ПРИН" if honest < base_honest else "отказ"
            verdict = f"судья:{v1} честный:{v2}"
        print(f"{('без оси' if k is None else f'k={k}'):>9} | "
              f"{rule_bits:8.1f} | {resid_bits:8.1f} | {pen:6.1f} | "
              f"{total:9.1f} | {honest:9.1f} | {cov:5.1f}% | "
              f"{len(mem.rules):6d} | {sc} {verdict}")


if __name__ == "__main__":
    args = sys.argv[1:]
    pairs = [(args[i], int(args[i + 1])) for i in range(0, len(args), 2)]
    for w, b in pairs:
        autopsy(w, glue_of(w), b)
