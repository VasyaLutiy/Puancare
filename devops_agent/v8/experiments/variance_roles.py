"""Дисперсия компилятора: K компиляций × N контрольных абзацев → стабильность ролей.

Урок 7 июл: batch-замер по одной пробе на абзац недостоверен — тот же абзац даёт
то пусто, то учебник. Меряем ДО гейта (сырой компилятор), чтобы не путать молчание
ролей с гейт-дропом интервенций.

Метрика на абзац: для каждого слота (uses/acts_on/requires/establishes) —
|пересечение id по всем K прогонам| / |объединение| (Жаккар). 1.0 = детерминизм.

  python -m devops_agent.v8.experiments.variance_roles [K=3] [N=10]
"""
import json
import os
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V8))
sys.path.insert(0, str(V8.parents[1]))

for line in open(V8.parents[1] / ".env"):
    line = line.strip()
    if "=" not in line or line.startswith("#"):
        continue
    k, v = line.split("=", 1)
    os.environ[{"AZURE_OPENAI_ENDPROINT": "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_KEY": "AZURE_API_KEY"}.get(k.strip(), k.strip())] = v.strip().strip('"\'')

from devops_agent.v8.contract import LAW_PROMPT, META_SCHEMA  # noqa: E402
from book_graph import paragraphs  # noqa: E402
from utils_azure import AzureJSON  # noqa: E402

SYS = ("You are the NLU COMPILER of an agent that thinks ONLY in six meta-types plus a thin logical "
       "layer. Translate the human text into that structure; named things are snake_case English ids "
       "only, no domain words in structure. ENTITY (a thing); RESOURCE (quantitative knob OF an "
       "entity, kind 'ordered'|'bounded'); SETTING (non-quantitative knob OF an entity, kind "
       "'categorical'|'boolean'); STATUS (state axis OF an entity: health/readiness/goal); "
       "INTERVENTION (an action: it ACTS_ON patient-entities, USES instrument-entities, REQUIRES "
       "state preconditions, ESTABLISHES its effects); DEPENDENCY (entity depends on entity).\n"
       + LAW_PROMPT +
       "\nUse ONLY what the text implies; do not invent. The text is from a BOOK — it may be "
       "narrative/advice with no concrete system at all; then emit empty lists. Respond JSON only.")

SLOTS = ("uses", "acts_on", "requires", "establishes")


def slot_ids(meta, slot):
    out = set()
    for iv in meta.get("interventions") or []:
        if isinstance(iv, dict):
            out |= {str(x) for x in (iv.get(slot) or [])}
    return out


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    ps = paragraphs(Path("/private/tmp/ccc/docspdf/output/IaC Thales.md"))
    idx = list(range(0, len(ps), max(1, len(ps) // N)))[:N]     # детерминированные контрольные
    az = AzureJSON()
    agg = {s: [] for s in SLOTS}
    for i in idx:
        runs = [az.ask(system=SYS, user=f"book paragraph: {ps[i]}", schema=META_SCHEMA)
                for _ in range(K)]
        row = []
        for s in SLOTS:
            sets = [slot_ids(m, s) for m in runs]
            union = set().union(*sets)
            inter = set.intersection(*sets) if sets else set()
            j = (len(inter) / len(union)) if union else None    # None = слот пуст во всех прогонах
            if j is not None:
                agg[s].append(j)
            row.append("—" if j is None else f"{j:.2f}")
        print(f"iac[{i:3d}]  " + "  ".join(f"{s}={v}" for s, v in zip(SLOTS, row)), flush=True)
    print("\nсредний Жаккар по слотам (только непустые):")
    for s in SLOTS:
        xs = agg[s]
        print(f"  {s:12s} {sum(xs)/len(xs):.2f} (n={len(xs)})" if xs else f"  {s:12s} — везде пусто")


if __name__ == "__main__":
    main()
