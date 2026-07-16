# Широкий держи-аут F2: изоморф с обращённым алфавитом для КАЖДОГО мира.
# Уровень 1 (узнавание): quick_signature бит-в-бит @300/600/1000.
# Уровень 2 (полный конвейер): runworld.live @1000 — спектры осей бит-в-бит,
# экзамен (если объявлен) с тем же счётом.
import sys, os, glob, json
sys.path.insert(0, "/Users/kyrylo/Documents/Pers/ccc")
os.chdir("/Users/kyrylo/Documents/Pers/ccc")

from worldkit import load_spec, make_glue
from experiments.f2_isomorph.isomorph import build_map, apply_map, check_reversal
import fastpath, koopman, runworld
from organism2 import Organism
runworld.Organism = Organism

SCRATCH = os.path.dirname(os.path.abspath(__file__))

WORLDS = sorted(glob.glob("worlds/*.yaml")) \
       + sorted(glob.glob("worlds/last/*.yaml")) \
       + sorted(glob.glob("worlds/comp*/*.yaml")) \
       + sorted(glob.glob("worlds/hard*/*.yaml")) \
       + sorted(glob.glob("worldforge/in/batch-*/m*/*.yaml"))
WORLDS = [w for w in WORLDS if "ШАБЛОН" not in w]

fails = 0
n_sig = n_full = 0
for wp in WORLDS:
    try:
        src = open(wp, encoding="utf-8").read()
        m = build_map([src])
        check_reversal(m)
        iso = os.path.join(SCRATCH, "sweep_iso.yaml")
        with open(iso, "w", encoding="utf-8") as fh:
            fh.write(apply_map(src, m))
        ga, gb = make_glue(load_spec(wp)), make_glue(load_spec(iso))
    except Exception as e:
        print(f"SKIP {wp}: {e}", flush=True)
        continue
    bad = []
    for tb in (300, 600, 1000):
        try:
            sa, _ = fastpath.quick_signature(ga, tb)
            sb, _ = fastpath.quick_signature(gb, tb)
            n_sig += 1
            if sa != sb:
                bad.append(f"sig@{tb}")
        except Exception as e:
            bad.append(f"sig@{tb}:КРАХ {e}")
    try:
        oa = runworld.live(ga, False, 1000)
        ob = runworld.live(gb, False, 1000)
        n_full += 1
        ka = sorted((ax.k, json.dumps(koopman.signature(ax), sort_keys=True))
                    for ax in oa.axes)
        kb = sorted((ax.k, json.dumps(koopman.signature(ax), sort_keys=True))
                    for ax in ob.axes)
        if ka != kb:
            bad.append(f"оси {[x[0] for x in ka]}vs{[x[0] for x in kb]}")
        if ga["exams"]:
            _, probe, ask, undo = ga["exams"][0]
            inv = {v: k for k, v in m.items()}
            sca = runworld.exam(oa, ga, probe, ask, undo)
            scb = runworld.exam(ob, gb, m.get(probe, probe), m.get(ask, ask), undo)
            if sca != scb:
                bad.append(f"экзамен {sca}vs{scb}")
    except Exception as e:
        bad.append(f"live:КРАХ {e}")
    if bad:
        fails += 1
        print(f"✗ {wp}: {'; '.join(bad)}", flush=True)
print(f"\nИТОГ: миров {len(WORLDS)}, провалов {fails}; "
      f"sig-сравнений {n_sig}, полных прогонов {n_full}")
