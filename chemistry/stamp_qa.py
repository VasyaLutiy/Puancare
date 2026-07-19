"""stamp_qa — штамповка Q/A из мира: порции × пары действий × моменты.

Хвост цепочки датасета: мир (yaml) → вопросы с МАШИННОЙ истиной.
Ответ вычисляет движок (worldkit), не разметчик и не LLM: вопрос —
«сделал A, получил X; что даст B на той же порции сейчас?», ответ —
результат честного step(B) в том же состоянии мира. Динамика двигается
самими шагами, поэтому поздние вопросы — про изменившийся мир.

    python3 chemistry/stamp_qa.py <мир.yaml> [--show N]
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from worldkit import load_spec, make_glue, GenericWorld  # noqa: E402


def stamp(path, moments=(0, 20), seed=1):
    spec = load_spec(path)
    acts = list(spec.obj_actions)
    pairs = [(a, b) for a in acts for b in acts if a != b]
    qa = []
    for t in moments:
        w = GenericWorld(spec, seed=seed)
        names = sorted(w.objects)
        for i in range(t):                       # прокрутка динамики
            w.step(acts[0], names[i % len(names)])
        for name in sorted(w.objects):
            vis = dict(w.observe()["объекты"][name])
            for a, b in pairs:
                if name not in w.objects:
                    break
                ra = w.step(a, name)["result"]
                rb = w.step(b, name)["result"]
                qa.append(dict(момент=t, порция=name, видимое=vis,
                               зонд=a, отклик=ra, вопрос=b, ответ=rb))
    return spec, acts, pairs, qa


def main():
    path = sys.argv[1]
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 12
    spec, acts, pairs, qa = stamp(path)
    print(f"мир: {spec.name}; порций {spec.n_objects}; действий {len(acts)}; "
          f"упорядоченных пар {len(pairs)}")
    print(f"наштамповано в демо (2 момента): {len(qa)}; "
          f"комбинаторика на мир: {spec.n_objects} порций × {len(pairs)} пар "
          f"× T моментов динамики = {spec.n_objects * len(pairs)}×T вопросов\n")
    step = max(1, len(qa) // show)
    for i, q in enumerate(qa[::step][:show], 1):
        vis = ", ".join(f"{k}={v}" for k, v in q["видимое"].items())
        print(f"Q{i} [шаг~{q['момент']}] Порция «{q['порция']}» ({vis}). "
              f"Проба «{q['зонд']}» дала «{q['отклик']}». "
              f"Что даст «{q['вопрос']}»?")
        print(f"    ОТВЕТ ДВИЖКА: «{q['ответ']}»")


if __name__ == "__main__":
    main()
