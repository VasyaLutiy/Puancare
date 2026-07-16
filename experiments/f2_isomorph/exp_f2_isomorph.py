# F2 held-out: изоморф с обращённым алфавитом.
# Два мира, тождественные по структуре (lamps vs lamps_iso), у изоморфа
# все слова переименованы так, что лексикографический порядок ОБРАЩЁН
# внутри каждого сортируемого множества (действия, результаты, атрибуты,
# значения); порядок строк/списков YAML сохранён, чтобы потребление rng
# миром и агентом было тождественным.
#
# Проверки:
#   1) трасса изоморфна (та же прожитая жизнь с точностью до переименования)
#   2) подпись quick_signature бит-в-бит
#   3) если не бит-в-бит — насколько далеко (koopman.dist, по компонентам)

import sys, json
sys.path.insert(0, "/Users/kyrylo/Documents/Pers/ccc")
import os
os.chdir("/Users/kyrylo/Documents/Pers/ccc")

from worldkit import load_spec, make_glue
import fastpath, koopman

SCRATCH = os.path.dirname(os.path.abspath(__file__))
ORIG = "worlds/last/lamps.yaml"
ISO = os.path.join(SCRATCH, "lamps_iso.yaml")

A2O = {
    "п_вкрутить_в_патрон": "вкрутить_в_патрон",
    "я_включить_надолго": "включить_надолго",
    "а_потрясти": "потрясти",
}
R2O = {
    "я_зажглась": "зажглась", "а_темно": "темно",
    "а_светит": "светит", "я_не_горит": "не_горит",
    "а_тихо": "тихо", "я_дребезжит": "дребезжит",
}


def run(path, budget):
    glue = make_glue(load_spec(path))
    org = fastpath._collect(glue, budget)      # тот же сид, что внутри quick_signature
    sig, fit = fastpath.quick_signature(glue, budget)
    return org, sig, fit


for budget in (300, 600, 1000):
    orgA, sigA, fitA = run(ORIG, budget)
    orgB, sigB, fitB = run(ISO, budget)

    seqA = [(r["action"], r["outcome"][0]) for r in orgA.records]
    seqB = [(A2O.get(r["action"], r["action"]),
             R2O.get(r["outcome"][0], r["outcome"][0])) for r in orgB.records]
    iso_ok = seqA == seqB

    bit_ok = sigA == sigB
    d = koopman.dist(sigA, sigB)

    print(f"\n=== budget={budget} ===")
    print(f"трасса изоморфна: {iso_ok}  (шагов: {len(seqA)}/{len(seqB)})")
    print(f"подпись бит-в-бит: {bit_ok}   koopman.dist={d:.6g}")
    print(f"k: {fitA.k} vs {fitB.k}")
    if not bit_ok:
        for c in koopman.COMPS:
            print(f"  {c}: d={koopman.comp_dist(sigA, sigB, c):.6g}")
        for kk in sorted(set(sigA.get("ks", {})) & set(sigB.get("ks", {}))):
            a, b = sigA["ks"][kk], sigB["ks"][kk]
            mark = "==" if a == b else "!="
            print(f"  срез k={kk}: {mark}")
            if a != b:
                print(f"    A: {json.dumps(a, ensure_ascii=False)}")
                print(f"    B: {json.dumps(b, ensure_ascii=False)}")
    print(f"верхний срез A: spec={sigA['spec']} osc={sigA['osc']} "
          f"tau={sigA['tau']} sharp={sigA['sharp']}")
    print(f"верхний срез B: spec={sigB['spec']} osc={sigB['osc']} "
          f"tau={sigB['tau']} sharp={sigB['sharp']}")
