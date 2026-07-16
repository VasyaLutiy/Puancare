# F2 held-out #2: comp/donor (k=3, по 3 результата на действие) vs его
# изоморф с обращённым алфавитом. Здесь в игре и трёхчленные суммы
# (неассоциативность float), и rng-старт эмиссий по sorted(res_a).

import sys, json
sys.path.insert(0, "/Users/kyrylo/Documents/Pers/ccc")
import os
os.chdir("/Users/kyrylo/Documents/Pers/ccc")

from worldkit import load_spec, make_glue
import fastpath, koopman

SCRATCH = os.path.dirname(os.path.abspath(__file__))
ORIG = "worlds/comp/donor.yaml"
ISO = os.path.join(SCRATCH, "donor_iso.yaml")

A2O = {"а_просветить": "просветить", "я_нагрузить": "нагрузить"}
R2O = {
    "я_пик": "пик", "п_полка": "полка", "а_провал": "провал",
    "п_мощно": "мощно", "а_слабо": "слабо", "я_глохнет": "глохнет",
}


def run(path, budget):
    glue = make_glue(load_spec(path))
    org = fastpath._collect(glue, budget)
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
        ka = sigA.get("ks", {})
        kb = sigB.get("ks", {})
        for kk in sorted(set(ka) & set(kb)):
            a, b = ka[kk], kb[kk]
            mark = "==" if a == b else "!="
            print(f"  срез k={kk}: {mark}")
            if a != b:
                print(f"    A: {json.dumps(a, ensure_ascii=False)}")
                print(f"    B: {json.dumps(b, ensure_ascii=False)}")
        na, nb = sigA.get("noise"), sigB.get("noise")
        if na != nb:
            print(f"  noise A: {na}")
            print(f"  noise B: {nb}")
    print(f"верхний срез A: k={sigA['k']} spec={sigA['spec']} osc={sigA['osc']} "
          f"tau={sigA['tau']} sharp={sigA['sharp']}")
    print(f"верхний срез B: k={sigB['k']} spec={sigB['spec']} osc={sigB['osc']} "
          f"tau={sigB['tau']} sharp={sigB['sharp']}")
