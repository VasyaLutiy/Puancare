# Тот же мир, тот же сид организма — зависит ли подпись от PYTHONHASHSEED
# (порядок итерации set строк)? Если да — «бит-в-бит» не воспроизводится
# даже без всяких изоморфов, между двумя запусками питона.

import sys, json, hashlib
sys.path.insert(0, "/Users/kyrylo/Documents/Pers/ccc")
import os
os.chdir("/Users/kyrylo/Documents/Pers/ccc")

from worldkit import load_spec, make_glue
import fastpath

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 600
world = sys.argv[2] if len(sys.argv) > 2 else "worlds/comp/donor.yaml"

glue = make_glue(load_spec(world))
sig, fit = fastpath.quick_signature(glue, budget)
blob = json.dumps(sig, ensure_ascii=False, sort_keys=True, default=repr)
print(f"HASHSEED={os.environ.get('PYTHONHASHSEED', '?'):>3} budget={budget} "
      f"md5={hashlib.md5(blob.encode()).hexdigest()} noise={sig['noise']}")
