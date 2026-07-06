"""
datagen.py — синтетические АБСТРАКТНЫЕ миры (NVIDIA NeMo Data Designer). Ноль домена.

Мир = пространство БЕЗЫМЯННЫХ фич f0..f{m-1} (каждая своего вида: ordered|categorical|boolean)
+ таблица ТИПОВ. Тип = какие фичи РЕЛЕВАНТНЫ (гейтят статус) и каким условием (gate).
Статус вещи = ok ⟺ все релевантные фичи удовлетворяют свой gate. Никакой семантики.

Истину (какие фичи релевантны + их gate) СЭМПЛИТ Data Designer — не моя рука (нет bias GD1).
surface = ШУМНАЯ проекция НАБОРА релевантных фич (не их gate и не type_id): какие фичи «торчат».
⇒ распознать тип можно лишь до набора фич; сам gate (порог/значение) добывается ПРОБОЙ.
Истину сохраняем → стенд сможет проверить обучение (A–E).
"""

import json
import random
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

from utils_azure import _load_env

KINDS = ["ordered", "categorical", "boolean"]
ORD_LEVELS = [1, 2, 3, 4]                 # абстрактная шкала; gate = порог (value ≥ gate → ok)
CAT_VALUES = ["v0", "v1", "v2"]           # gate = «правильное» значение
DECOYS = [f"d{i}" for i in range(6)]      # шумовые токены поверхности, не несут типа
OUT = Path(__file__).resolve().parent / "worlds"


def _designer() -> DataDesigner:
    _load_env()
    return DataDesigner()


def _feature_space(m: int, rng: random.Random) -> dict:
    """Конфиг мира: m безымянных фич, каждой случайный вид."""
    return {f"f{i}": rng.choice(KINDS) for i in range(m)}


def gen_world_spec(dz: DataDesigner, kinds: dict, k_types: int):
    """Сэмплируем для каждого типа: релевантна ли фича и её gate (по виду фичи)."""
    b = dd.DataDesignerConfigBuilder()
    for fi, kind in kinds.items():
        b.add_column(dd.SamplerColumnConfig(
            name=f"{fi}_rel", sampler_type=dd.SamplerType.BERNOULLI,
            params=dd.BernoulliSamplerParams(p=0.45)))
        gate_vals = ORD_LEVELS if kind == "ordered" else CAT_VALUES if kind == "categorical" else [1]
        b.add_column(dd.SamplerColumnConfig(
            name=f"{fi}_gate", sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=gate_vals)))
    df = dz.preview(b, num_records=k_types).dataset.copy()
    df.insert(0, "type_id", [f"T{i}" for i in range(len(df))])
    return df


def laws_from(df, kinds: dict) -> dict:
    laws = {}
    for _, r in df.iterrows():
        feats = {}
        for fi, kind in kinds.items():
            if bool(r[f"{fi}_rel"]):
                g = r[f"{fi}_gate"]
                feats[fi] = {"kind": kind, "gate": (str(g) if kind == "categorical" else int(g))}
        laws[r["type_id"]] = feats
    return laws


def gen_instances(dz: DataDesigner, type_ids: list, n: int):
    b = dd.DataDesignerConfigBuilder()
    b.add_column(dd.SamplerColumnConfig(
        name="instance_id", sampler_type=dd.SamplerType.UUID,
        params=dd.UUIDSamplerParams(prefix="i-", short_form=True, uppercase=False)))
    b.add_column(dd.SamplerColumnConfig(
        name="type_id", sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=type_ids)))
    return dz.preview(b, num_records=n).dataset.copy()


def surface_of(law: dict, rng: random.Random) -> list:
    """Какие фичи релевантны (без gate) + шум. type_id НЕ утекает."""
    toks = [f"r:{fi}" for fi in law]
    if rng.random() < 0.25:
        toks.append(rng.choice(DECOYS))
    if len(toks) > 1 and rng.random() < 0.15:
        toks.pop(rng.randrange(len(toks)))
    return sorted(set(toks))


def build_world(world_id: str, m: int = 4, k_types: int = 6, n: int = 24, seed: int = 0):
    rng = random.Random(seed)
    kinds = _feature_space(m, rng)
    dz = _designer()
    spec = gen_world_spec(dz, kinds, k_types)
    laws = laws_from(spec, kinds)
    inst = gen_instances(dz, list(spec["type_id"]), n)
    inst["surface"] = [surface_of(laws[t], rng) for t in inst["type_id"]]

    wdir = OUT / world_id
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "world_spec.json").write_text(
        json.dumps({"feature_kinds": kinds, "types": laws}, ensure_ascii=False, indent=2))
    with (wdir / "instances.jsonl").open("w") as f:
        for _, r in inst.iterrows():
            f.write(json.dumps({"instance_id": r["instance_id"], "type_id": r["type_id"],
                                "surface": r["surface"]}, ensure_ascii=False) + "\n")
    return kinds, laws, inst, wdir


def main() -> None:
    print("=== datagen: АБСТРАКТНЫЙ мир (NeMo Data Designer), ноль домена ===\n")
    kinds, laws, inst, wdir = build_world("world_demo", m=4, k_types=6, n=24)

    print("ПРОСТРАНСТВО ФИЧ (вид каждой, безымянно):", kinds)
    print("\nСКРЫТЫЕ ЗАКОНЫ ТИПОВ (релевантные фичи + gate; ground-truth, агенту НЕ виден):")
    for t, feats in laws.items():
        body = ", ".join(
            f"{fi}{'≥' if d['kind'] == 'ordered' else '='}{d['gate']}" for fi, d in feats.items()
        ) or "(пусто — всегда ok)"
        print(f"  {t}: {body}")

    print("\nПОТОК ИНСТАНСОВ (агент видит id+surface; type_id — только для стенда):")
    print(inst[["instance_id", "surface", "type_id"]].head(10).to_string(index=False))
    print(f"\nсохранено: {wdir}")
    print("surface = какие фичи релевантны (шумно), БЕЗ gate ⇒ gate добывается пробой.")


if __name__ == "__main__":
    main()
