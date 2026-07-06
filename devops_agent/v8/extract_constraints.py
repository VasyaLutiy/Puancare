"""
extract_constraints.py — ТЕСТ на NeMo-данных: вытаскивает ли NLU-компилятор mutex/atmost
из человеческого текста tradeoff-строк (вариант 1, «человек → связка»).

Грунт-обучатель сюда не применим (нет мира-судьи). Здесь меряем ИЗВЛЕЧЕНИЕ:
расширяем схему компилятора связками, перекомпилируем tradeoff-строки и считаем
с ДВУМЯ ground-truth:
  • контрол-ось: relation='tradeoff' ⇒ ждём ≥1 связку (recall сигнала);
  • семантический судья: стало ли структурно вернее, чем baseline без связок.
Плюс честная карта residue: какие tradeoff-ы это НЕ mutex и НЕ atmost (знаковая сцепка → if-класс).
"""

import json
import sys
from pathlib import Path

from devops_agent.v8.constraints import AtMost, ConstraintLayer, Mutex, validate_constraints
from utils_azure import AzureJSON

DS = Path(__file__).resolve().parent / "datasets" / "tasks.jsonl"

# схема компилятора (6 мета-типов, как в metagen.META_SCHEMA) + логический слой
SCHEMA = {
    "entities": "list of {id} — the things",
    "resources": "list of {id, entity, kind} kind='ordered'|'bounded' — quantitative adjustable properties",
    "settings": "list of {id, entity, kind} kind='categorical'|'boolean' — non-quantitative settings",
    "statuses": "list of {id, entity} — health/goal states",
    "interventions": "list of {id, establishes, requires} — actions; establishes/requires are id lists",
    "dependencies": "list of {from, to} — entity 'from' depends on entity 'to'",
    "goal": "list of status ids that are the desired target (empty for a pure fact)",
}
SCHEMA["constraints"] = (
    "list of {kind, members, bound} — the LOGICAL LAYER over the 6 types. "
    "kind='mutex' (exactly TWO things cannot both be ON / a hard either-or; members=[id,id]) OR "
    "kind='atmost' (TWO OR MORE things share ONE limited budget; members=[id,id,...], bound=integer). "
    "RULES: (1) every member MUST be a DECLARED id; (2) for an either-or among options, model each "
    "option as a boolean SETTING (e.g. band_2_4ghz_on, band_5ghz_on) and put the constraint over those "
    "SETTING/STATUS ids — NEVER over an entity id; (3) all members of one constraint must be the SAME "
    "kind (all resources, OR all settings, OR all statuses); (4) atmost needs ≥2 members sharing a "
    "budget — a single thing with a numeric cap is just a resource with kind='bounded', NOT a constraint; "
    "(5) if there is no genuine shared-limit or either-or, emit constraints: []. "
    "Do NOT use this for one action with both a good and a bad effect — that is not mutex/atmost."
)
COMPILER_SYS = (
    "You are the NLU COMPILER of an agent that thinks ONLY in six meta-types PLUS a thin logical layer. "
    "Translate the human text into that structure; named things are ids only, no domain words in structure. "
    "ENTITY (a thing); RESOURCE (quantitative knob, kind 'ordered'|'bounded'); SETTING (non-quantitative "
    "knob, kind 'categorical'|'boolean'); STATUS (health/goal state); INTERVENTION (action establishing "
    "states, may require others); DEPENDENCY (entity depends on entity). "
    "LOGICAL LAYER: MUTEX = two things cannot both be ON (hard either-or); ATMOST = several things share "
    "ONE limited budget/capacity (raising one leaves less for the others). A tradeoff from a shared limit "
    "=> atmost; a strict either-or => mutex. Constraint members must be DECLARED ids of the SAME kind "
    "(model an either-or as boolean SETTINGS and constrain THOSE, never the entity). A lone numeric cap "
    "is a bounded resource, not a constraint. If no shared-limit/either-or, constraints=[]. "
    "Use ONLY what the text implies; do not invent. Respond JSON only."
)
JUDGE_SCHEMA = {
    "faithful": "true if the structure (incl. constraints) captures the text's things/limits/tradeoff; else false",
    "missing": "one short phrase: what was dropped or invented; empty if faithful",
}
JUDGE_SYS = (
    "You verify a compiler that may emit a logical layer (mutex/atmost) on top of 6 meta-types. Given the "
    "ORIGINAL text and the compiled structure, decide if it FAITHFULLY captures the meaning INCLUDING any "
    "shared-limit/either-or tradeoff. Strict: dropped or invented elements => false. JSON only."
)


def declared_ids(meta: dict):
    res = {r.get("id") for r in meta.get("resources", [])}
    sett = {s.get("id") for s in meta.get("settings", [])}
    st = {s.get("id") for s in meta.get("statuses", [])}
    return res, st, sett


def to_layer(meta: dict):
    """Разобрать emitted constraints в ConstraintLayer; вернуть (layer, n_malformed)."""
    layer, bad = ConstraintLayer(), 0
    for c in meta.get("constraints", []):
        try:
            members = tuple(c["members"])
            if c.get("kind") == "mutex":
                layer.mutexes.append(Mutex(members=members))
            elif c.get("kind") == "atmost":
                if len(set(members)) < 2:        # одночленный «atmost» = bounded-resource, не связка
                    continue
                layer.atmosts.append(AtMost(members=members, bound=int(c.get("bound", 0))))
            else:
                bad += 1
        except (KeyError, TypeError, ValueError):
            bad += 1
    return layer, bad


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 38
    rows = [json.loads(l) for l in DS.open()]
    tr = [r for r in rows if r["controls"]["relation"] == "tradeoff"][:cap]
    az = AzureJSON()
    print(f"=== извлечение mutex/atmost из {len(tr)} tradeoff-строк NeMo; before/after ===\n")

    def judge(human, structure):
        return bool(az.ask(system=JUDGE_SYS,
                           user=f"HUMAN: {human}\n\nSTRUCTURE: {json.dumps(structure, ensure_ascii=False)}",
                           schema=JUDGE_SCHEMA).get("faithful"))

    base_faithful = emitted = struct_ok = new_faithful = mutex_n = atmost_n = no_constraint = 0
    residue = []
    for i, r in enumerate(tr):
        meta = az.ask(system=COMPILER_SYS, user=f"human text: {r['human']}", schema=SCHEMA)
        layer, bad = to_layer(meta)
        ncon = len(layer.mutexes) + len(layer.atmosts)
        res_ids, st_ids, set_ids = declared_ids(meta)
        viol = validate_constraints(layer, res_ids, st_ids, set_ids)
        ok_struct = ncon >= 1 and not viol and bad == 0
        # ЧЕСТНОЕ before/after: ОДИН судья на baseline-мета (без связок) и на новую (со связками)
        base_f = judge(r["human"], r["meta"])
        faithful = judge(r["human"], meta)

        base_faithful += base_f
        emitted += (ncon >= 1)
        struct_ok += ok_struct
        new_faithful += faithful
        mutex_n += len(layer.mutexes); atmost_n += len(layer.atmosts)
        no_constraint += (ncon == 0)
        if ncon == 0:
            residue.append(r["human"][:90])

        kinds = [("mutex", m.members) for m in layer.mutexes] + \
                [("atmost", a.members, a.bound) for a in layer.atmosts]
        print(f"[{i}] {r['human'][:78]}")
        print(f"    связки: {kinds if kinds else '— нет —'}"
              f"{'  ВНЕ-КОНТРАКТА:'+str(sorted(set(viol))) if viol else ''}"
              f"{'  malformed='+str(bad) if bad else ''}")
        print(f"    faithful (один судья): baseline={base_f} → c-связками={faithful}\n")

    n = len(tr)
    print("=" * 60)
    print(f"эмитнул ≥1 связку:        {emitted}/{n}  (контрол-ось ждёт связку на КАЖДОЙ tradeoff → recall={emitted/n:.2f})")
    print(f"структурно валидны:       {struct_ok}/{n}")
    print(f"всего связок:             mutex={mutex_n}  atmost={atmost_n}")
    print(f"семантич. faithful:       baseline {base_faithful}/{n}  →  со связками {new_faithful}/{n}")
    print(f"\nresidue (tradeoff БЕЗ связки — кандидаты в if-класс / знаковая сцепка): {no_constraint}/{n}")
    for h in residue[:12]:
        print(f"  · {h}")


if __name__ == "__main__":
    main()
