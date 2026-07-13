"""runworld2 — тот же полигон, что runworld.py, но ядро — organism2
(переписано из MODEL.md). Использование идентично:

  python3 runworld2.py мир.yaml [--check]

Вся логика прогона/экзаменов — из runworld: сравнение ядер честно
только при побайтово общем полигоне.
"""

import sys

import runworld
from organism2 import Organism
from worldkit import load_spec, make_glue

runworld.Organism = Organism

if __name__ == "__main__":
    args = list(sys.argv[1:])
    if not args:
        sys.exit("использование: python3 runworld2.py мир.yaml [--check]")
    try:
        spec = load_spec(args[0])
    except (ValueError, OSError) as e:
        sys.exit(f"АНКЕТА НЕ ПРИНЯТА: {e}")
    glue = make_glue(spec)
    if "--check" in args:
        runworld.check(spec, glue)
    else:
        runworld.full_run(spec, glue)
