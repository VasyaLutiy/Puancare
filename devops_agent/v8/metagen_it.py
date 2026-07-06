"""
metagen_it.py — датасет v2: ОДИН домен (бытовое IT «для чайников»), острые контрол-оси.

Отличие от metagen.py:
  • домен фиксирован (снижение дисперсии для отладки моста человек→связка); живёт ТОЛЬКО в
    поверхности — мета и агент доменно-слепы;
  • relation явный: {none, dependency, mutex, atmost} вместо размытого 'tradeoff' →
    контрол-ось = ОСТРАЯ ground-truth (relation=atmost ⇒ ждём atmost, и т.д.);
  • первичная истина — покомпонентный scorecard по осям; LLM-судья вторичен (на простых
    фразах перестаёт быть насыщенным);
  • if-класс (знаковая сцепка) по построению исключён — мерим mutex/atmost без confound.

Компилятор (SCHEMA, COMPILER_SYS) и парсер связок переиспользуются из extract_constraints.py.
"""

import json
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

from devops_agent.v8.constraints import validate_constraints
from devops_agent.v8.extract_constraints import (COMPILER_SYS, JUDGE_SCHEMA, JUDGE_SYS,
                                                 SCHEMA, declared_ids, to_layer)
from utils_azure import AzureJSON, _load_env

OUT = Path(__file__).resolve().parent / "datasets"

# --- контрол-оси: домен один, структура — острая ---
SUBTOPICS = ["personal_website", "email", "phone_storage", "home_wifi",
             "cloud_photos", "streaming_subscription", "password_manager"]
RELATIONS = ["none", "dependency", "mutex", "atmost"]
PROPERTY_KINDS = ["ordered", "categorical", "boolean"]
N_THINGS = [1, 2, 3]
YES_NO = ["yes", "no"]

SYS_TEXT = ("You write one or two short, plain sentences for a NON-TECHNICAL beginner. "
            "Everyday words, no jargon, no lists, no preamble.")
PROMPT = (
    "Write ONE or TWO short, plain sentences a non-technical beginner might say or hear about their "
    "'{{ subtopic }}'. Involve about {{ n_things }} concrete everyday thing(s). Express a "
    "'{{ relation }}' relationship: none = just one thing and a setting of it; dependency = one thing "
    "needs another to work; mutex = two options where you must pick ONE, you cannot have both; "
    "atmost = several things share ONE limited capacity/quota/budget. Center on a "
    "'{{ property_kind }}'-type property (ordered = a number/amount, categorical = a choice among "
    "options, boolean = on/off). {{ has_goal }} mention a desired working/healthy outcome. "
    "Everyday language only."
)


def gen_human(n: int):
    _load_env()
    dz = DataDesigner()
    b = dd.DataDesignerConfigBuilder()
    for name, vals in [("subtopic", SUBTOPICS), ("relation", RELATIONS),
                       ("property_kind", PROPERTY_KINDS), ("n_things", N_THINGS), ("has_goal", YES_NO)]:
        b.add_column(dd.SamplerColumnConfig(name=name, sampler_type=dd.SamplerType.CATEGORY,
                                            params=dd.CategorySamplerParams(values=vals)))
    b.add_column(dd.LLMTextColumnConfig(name="human_text", model_alias="nvidia-text",
                                        system_prompt=SYS_TEXT, prompt=PROMPT))
    return dz.preview(b, num_records=n).dataset.copy()


def scorecard(ctrl: dict, meta: dict, layer) -> dict:
    """Острая ground-truth из осей: ждём ровно ту структуру, что заявлена. Каждый пункт True/False."""
    n_mutex, n_atmost = len(layer.mutexes), len(layer.atmosts)
    n_dep = len(meta.get("dependencies", []))
    rel = ctrl["relation"]
    out = {}
    if rel == "none":
        out["relation"] = (n_mutex == 0 and n_atmost == 0 and n_dep == 0)
    elif rel == "dependency":
        out["relation"] = n_dep >= 1
    elif rel == "mutex":
        out["relation"] = n_mutex >= 1
    elif rel == "atmost":
        out["relation"] = n_atmost >= 1
    kinds = [x.get("kind") for x in meta.get("resources", [])] + \
            [x.get("kind") for x in meta.get("settings", [])]
    expect = {"ordered": {"ordered", "bounded"}, "categorical": {"categorical"},
              "boolean": {"boolean"}}[ctrl["property_kind"]]
    out["property_kind"] = bool(expect & set(kinds))
    if str(ctrl["has_goal"]) == "yes":
        out["goal"] = len(meta.get("goal", [])) >= 1
    return out


def build(n: int, name: str = "tasks_it"):
    az = AzureJSON()
    df = gen_human(n)
    cols = ["subtopic", "relation", "property_kind", "n_things", "has_goal"]
    rows = []
    for _, r in df.iterrows():
        ctrl = {c: r[c] for c in cols}
        human = r["human_text"]
        meta = az.ask(system=COMPILER_SYS, user=f"human text: {human}", schema=SCHEMA)
        layer, bad = to_layer(meta)
        res_ids, st_ids, set_ids = declared_ids(meta)
        viol = validate_constraints(layer, res_ids, st_ids, set_ids)
        sc = scorecard(ctrl, meta, layer)
        sem = az.ask(system=JUDGE_SYS,
                     user=f"HUMAN: {human}\n\nSTRUCTURE: {json.dumps(meta, ensure_ascii=False)}",
                     schema=JUDGE_SCHEMA)
        rows.append({"controls": ctrl, "human": human, "meta": meta,
                     "constraint_viol": sorted(set(viol)), "constraint_malformed": bad,
                     "scorecard": sc, "semantic": sem})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, path


def recompile(name: str = "tasks_it"):
    """Перекомпилировать ТЕ ЖЕ human-фразы (без NeMo) — чистый before/after на идентичном входе."""
    az = AzureJSON()
    src = [json.loads(l) for l in (OUT / f"{name}.jsonl").open()]
    rows = []
    for old in src:
        ctrl, human = old["controls"], old["human"]
        meta = az.ask(system=COMPILER_SYS, user=f"human text: {human}", schema=SCHEMA)
        layer, bad = to_layer(meta)
        res_ids, st_ids, set_ids = declared_ids(meta)
        viol = validate_constraints(layer, res_ids, st_ids, set_ids)
        sc = scorecard(ctrl, meta, layer)
        sem = az.ask(system=JUDGE_SYS,
                     user=f"HUMAN: {human}\n\nSTRUCTURE: {json.dumps(meta, ensure_ascii=False)}",
                     schema=JUDGE_SCHEMA)
        rows.append({"controls": ctrl, "human": human, "meta": meta,
                     "constraint_viol": sorted(set(viol)), "constraint_malformed": bad,
                     "scorecard": sc, "semantic": sem})
    path = OUT / f"{name}_recompiled.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, path


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "recompile":
        print("=== metagen_it RECOMPILE: те же human-фразы, новый контракт+промпт; scorecard ===\n")
        rows, path = recompile()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        print(f"=== metagen_it: бытовое IT, острые оси, человек(NeMo)→мета(Azure)→scorecard; N={n} ===\n")
        rows, path = build(n)
    sc_hit = sc_tot = sem_ok = viol_rows = 0
    by_rel = {}
    for i, row in enumerate(rows):
        c, sc = row["controls"], row["scorecard"]
        sem = bool(row["semantic"].get("faithful"))
        sc_hit += sum(1 for v in sc.values() if v); sc_tot += len(sc)
        sem_ok += sem
        viol_rows += bool(row["constraint_viol"] or row["constraint_malformed"])
        rel = c["relation"]
        d = by_rel.setdefault(rel, [0, 0])
        d[0] += int(sc.get("relation", False)); d[1] += 1
        print(f"[{i}] {c['subtopic']}/{c['relation']}/{c['property_kind']}/n={c['n_things']}/goal={c['has_goal']}")
        print(f"  HUMAN: {row['human']}")
        print(f"  META : {json.dumps(row['meta'], ensure_ascii=False)}")
        flags = []
        if row["constraint_viol"]:
            flags.append(f"вне-контракта={row['constraint_viol']}")
        if row["constraint_malformed"]:
            flags.append(f"malformed={row['constraint_malformed']}")
        print(f"  SCORECARD: {sc}{'  ' + '; '.join(flags) if flags else ''}")
        print(f"  судья(вторичн.): faithful={sem} missing={row['semantic'].get('missing','')!r}\n")
    print("=" * 60)
    print(f"scorecard (острая истина по осям): {sc_hit}/{sc_tot} пунктов")
    print("  relation-точность по типу связи:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_rel.items())})
    print(f"связки вне контракта/malformed:    в {viol_rows}/{len(rows)} строк")
    print(f"судья (вторичный):                 faithful {sem_ok}/{len(rows)}")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
