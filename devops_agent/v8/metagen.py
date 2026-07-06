"""
metagen.py — генератор датасета: КОНТРОЛИРУЕМЫЙ human-like → мета-предикат → двойная сверка.

Как в доке Data Designer: сперва КОНТРОЛ-КОЛОНКИ (сэмплеры) ОЧЕРЧИВАЮТ область генерации,
затем LLM-колонка их рендерит через {{ }}. Это распределение генерации (законно), не хардкод знания.

Источники и проверка:
  (a) NVIDIA human-like: контрол-оси (domain, n_things, relation, property_kind, has_goal) → LLM-текст.
  (b) наш компилятор (Azure): human → мета-предикат в 6 мета-типах ISA v4.
  СВЕРКА двойная:
    • СТРУКТУРНАЯ — контрол-оси = частичная ground-truth (заявлена dependency → ждём dependency и т.п.);
    • СЕМАНТИЧЕСКАЯ — LLM-судья: поймала ли структура смысл текста.
Контрол-оси доменно-широкие (не IT-lock) и НЕ текут в структуру мета-предиката — только в human-текст.
"""

import json
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

from utils_azure import AzureJSON, _load_env

OUT = Path(__file__).resolve().parent / "datasets"

# --- (a) КОНТРОЛ-ОСИ области генерации (распределение, не знание агента) ---
DOMAINS = ["logistics", "cooking", "gardening", "automotive", "healthcare",
           "manufacturing", "retail", "networking"]
N_THINGS = [1, 2, 3]
RELATIONS = ["none", "dependency", "tradeoff"]
PROPERTY_KINDS = ["ordered", "categorical", "boolean"]
YES_NO = ["yes", "no"]

# --- (b) контракт компилятора = 6 мета-типов ISA v4 ---
META_SCHEMA = {
    "entities": "list of {id} — the things",
    "resources": "list of {id, entity, kind} kind='ordered'|'bounded' — quantitative adjustable properties",
    "settings": "list of {id, entity, kind} kind='categorical'|'boolean' — non-quantitative settings",
    "statuses": "list of {id, entity} — health/goal states",
    "interventions": "list of {id, establishes, requires} — actions; establishes/requires are id lists",
    "dependencies": "list of {from, to} — entity 'from' depends on entity 'to'",
    "goal": "list of status ids that are the desired target (empty for a pure fact)",
}
COMPILER_SYS = (
    "You are the NLU COMPILER of an agent that thinks ONLY in six meta-types. Translate the human text "
    "into that fixed structure; do NOT add domain words to the structure — named things are ids only. "
    "ENTITY (a thing); RESOURCE (quantitative knob, kind 'ordered'|'bounded'); SETTING (non-quantitative "
    "knob, kind 'categorical'|'boolean'); STATUS (health/goal state); INTERVENTION (action that establishes "
    "states/resources/settings, may require others); DEPENDENCY (entity depends on entity). Use ONLY what "
    "the text explicitly implies — do not invent entities, dependencies or states. Respond JSON only."
)
JUDGE_SCHEMA = {
    "faithful": "true if the structure captures the text's things/properties/dependencies/goal; else false",
    "missing": "one short phrase: what was dropped or invented; empty if faithful",
}
JUDGE_SYS = (
    "You verify a compiler. Given ORIGINAL human text and its compiled 6-meta-type structure, decide if "
    "the structure FAITHFULLY captures the text. Strict: dropped or invented elements => false. JSON only."
)


def gen_human(n: int):
    _load_env()
    dz = DataDesigner()
    b = dd.DataDesignerConfigBuilder()
    for name, vals in [("domain", DOMAINS), ("n_things", N_THINGS), ("relation", RELATIONS),
                       ("property_kind", PROPERTY_KINDS), ("has_goal", YES_NO)]:
        b.add_column(dd.SamplerColumnConfig(name=name, sampler_type=dd.SamplerType.CATEGORY,
                                            params=dd.CategorySamplerParams(values=vals)))
    b.add_column(dd.LLMTextColumnConfig(
        name="human_text", model_alias="nvidia-text",
        system_prompt="You write one short natural sentence. No lists, no preamble.",
        prompt=("Write ONE short (1–2 sentence) natural request or fact a person in the '{{ domain }}' "
                "domain might state. Involve about {{ n_things }} concrete thing(s); express a "
                "'{{ relation }}' relationship (none = a single thing, dependency = one needs another, "
                "tradeoff = shared limited resource); center on a {{ property_kind }}-type adjustable "
                "property; {{ has_goal }} mention a desired healthy/target state. Plain language."),
    ))
    return dz.preview(b, num_records=n).dataset.copy()


def compile_to_meta(human: str, az: AzureJSON) -> dict:
    return az.ask(system=COMPILER_SYS, user=f"human text: {human}", schema=META_SCHEMA)


def judge(human: str, meta: dict, az: AzureJSON) -> dict:
    return az.ask(system=JUDGE_SYS,
                  user=f"HUMAN: {human}\n\nMETA: {json.dumps(meta, ensure_ascii=False)}",
                  schema=JUDGE_SCHEMA)


def structural_check(ctrl: dict, meta: dict) -> dict:
    """Контрол-оси как частичная ground-truth: совпала ли структура с заявленным."""
    out = {}
    if ctrl["relation"] == "dependency":
        out["dependency"] = len(meta.get("dependencies", [])) >= 1
    if str(ctrl["has_goal"]) == "yes":
        out["goal"] = len(meta.get("goal", [])) >= 1
    kinds = [x.get("kind") for x in meta.get("resources", [])] + \
            [x.get("kind") for x in meta.get("settings", [])]
    pk = ctrl["property_kind"]
    expect = {"ordered": {"ordered", "bounded"}, "categorical": {"categorical"}, "boolean": {"boolean"}}[pk]
    out["property_kind"] = bool(expect & set(kinds))
    return out


def build(n: int, name: str = "tasks"):
    az = AzureJSON()
    df = gen_human(n)
    ctrl_cols = ["domain", "n_things", "relation", "property_kind", "has_goal"]
    rows = []
    for _, r in df.iterrows():
        ctrl = {c: r[c] for c in ctrl_cols}
        human = r["human_text"]
        meta = compile_to_meta(human, az)
        rows.append({"controls": ctrl, "human": human, "meta": meta,
                     "structural": structural_check(ctrl, meta),
                     "semantic": judge(human, meta, az)})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, path


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"=== metagen: контролируемый human (NVIDIA) → мета (Azure) → структурная+семантич. сверка; N={n} ===\n")
    rows, path = build(n)
    sem_ok = struct_ok = struct_tot = 0
    for i, row in enumerate(rows):
        c = row["controls"]
        sc = row["structural"]
        sem = bool(row["semantic"].get("faithful"))
        sem_ok += sem
        struct_ok += sum(1 for v in sc.values() if v); struct_tot += len(sc)
        print(f"[{i}] {c}")
        print(f"  (a) HUMAN: {row['human']}")
        print(f"  (b) META : {json.dumps(row['meta'], ensure_ascii=False)}")
        print(f"  СТРУКТУРНО (vs контрол-оси): {sc}")
        print(f"  СЕМАНТИЧЕСКИ (судья): faithful={sem} missing={row['semantic'].get('missing','')!r}\n")
    print(f"структурных совпадений: {struct_ok}/{struct_tot} | семантич. верность: {sem_ok}/{len(rows)} | {path}")


if __name__ == "__main__":
    main()
