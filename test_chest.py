"""test_chest — доказательство, что сундок держит: прожить -> dump -> load ->
сверить оси, правила и балл экзамена. Совпало всё -> сундук честный."""

import sys

import runworld
import chest
from organism2 import Organism
from worldkit import load_spec, make_glue

runworld.Organism = Organism


def axis_fingerprint(ax):
    return (tuple(ax.actions), ax.k,
            {a: sorted(rs) for a, rs in ax.res_a.items()},
            {f"{s}|{a}": {r: round(w, 9) for r, w in c.items()}
             for (s, a), c in sorted(ax.emis.items())},
            [round(h, 9) for h in ax.haz],
            [round(p, 9) for p in ax.pi])


def rules_fingerprint(mem):
    return sorted((r["action"], tuple(sorted(r["conds"])), r["outcome"],
                   r["support"], r["exceptions"]) for r in mem.rules)


def run(world):
    spec = load_spec(f"worlds/{world}.yaml")
    glue = make_glue(spec)
    org1 = runworld.live(glue, curious=True, budget=2000)

    path = f"/private/tmp/claude-501/chest_{world}.json"
    chest.dump(org1, path)
    org2 = chest.load(path, glue)

    ok = True
    # 1. оси
    f1 = [axis_fingerprint(a) for a in org1.axes]
    f2 = [axis_fingerprint(a) for a in org2.axes]
    print(f"[{world}] осей: {len(org1.axes)} -> {len(org2.axes)}   "
          f"оси идентичны: {f1 == f2}")
    ok &= f1 == f2
    # 2. правила
    r1, r2 = rules_fingerprint(org1.mem), rules_fingerprint(org2.mem)
    print(f"          правил: {len(org1.mem.rules)} -> {len(org2.mem.rules)}   "
          f"правила идентичны: {r1 == r2}")
    ok &= r1 == r2
    # 3. экзамен
    for label, probe, ask, undo in glue["exams"]:
        s1 = runworld.exam(org1, glue, probe, ask, undo)
        s2 = runworld.exam(org2, glue, probe, ask, undo)
        print(f"          экзамен {label}: {s1[0]}/{s1[1]} -> {s2[0]}/{s2[1]}   "
              f"совпал: {s1 == s2}")
        ok &= s1 == s2
    print(f"          ИТОГ: {'СУНДУК ДЕРЖИТ' if ok else 'ТЕЧЁТ'}\n")
    return ok


if __name__ == "__main__":
    worlds = sys.argv[1:] or ["batteries"]
    all_ok = all(run(w) for w in worlds)
    sys.exit(0 if all_ok else 1)
