#!/usr/bin/env python3
"""
Step B1 — КОМПИЛЯЦИЯ атомов (LLM, ОДИН РАЗ).

Фиксированный словарь атомов-свойств + понятия. Спрашиваем gpt-5.4-mini, какие
атомы применимы к каждому понятию. Сохраняем atoms.json — статический фундамент.
После этого рантайм (is-a = вложение, сходство = пересечение) работает БЕЗ LLM.

Словарь атомов общий для всех понятий — иначе пересечения невозможны.
"""

import sys
import json
import os
from concurrent.futures import ThreadPoolExecutor

from utils_azure import AzureJSON

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json")

# --- фиксированный словарь атомов (named, дискретные оси) ---
ATOMS = [
    "has_backbone", "warm_blooded", "cold_blooded", "has_gills", "has_lungs",
    "lives_in_water", "lives_on_land", "can_fly", "has_wings", "has_feathers",
    "has_fur", "has_scales", "smooth_skin", "lays_eggs", "gives_live_birth",
    "produces_milk", "has_legs", "four_legs", "two_legs", "no_legs",
    "has_fins", "has_tail", "breathes_air", "has_beak", "has_teeth",
    "is_predator", "eats_plants", "has_exoskeleton", "six_legs", "eight_legs",
    "segmented_body", "streamlined_body", "large_size", "small_size",
    "has_claws", "can_swim", "can_walk", "is_social", "domesticated", "has_eyes",
]

# --- понятия разных уровней абстракции + трудные случаи ---
CONCEPTS = [
    "animal", "vertebrate", "invertebrate", "mammal", "bird", "fish",
    "reptile", "amphibian", "insect",
    "carp", "goldfish", "pike", "salmon", "shark", "tuna", "eel",
    "whale", "dolphin", "seal", "bat", "dog", "cat", "horse", "cow",
    "mouse", "elephant",
    "eagle", "penguin", "ostrich", "sparrow", "owl", "duck",
    "frog", "snake", "lizard", "turtle", "crocodile",
    "bee", "ant", "butterfly", "spider", "octopus", "crab",
]

SYSTEM = (
    "Ты биолог-онтолог (json). Для данного животного-понятия выбери ВСЕ свойства "
    "из фиксированного списка, которые ТИПИЧНО применимы к этому понятию "
    "(прототипически, для обычного представителя). Не добавляй свойств вне списка."
)


def compile_atoms():
    az = AzureJSON()
    atoms_list = ", ".join(ATOMS)
    schema = {"applies": "список применимых свойств строго из данного списка"}

    def work(concept):
        user = (f"Понятие: '{concept}'. Список свойств: [{atoms_list}]. "
                f"Верни какие применимы.")
        try:
            r = az.ask(SYSTEM, user, schema)
            feats = [a for a in r.get("applies", []) if a in set(ATOMS)]
            return concept, feats
        except Exception as e:
            return concept, None

    result = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for concept, feats in ex.map(work, CONCEPTS):
            if feats is not None:
                result[concept] = sorted(feats)
                print(f"  {concept:12s} ({len(feats):2d}): {', '.join(feats)}", file=sys.stderr)
            else:
                print(f"  {concept:12s} ОШИБКА", file=sys.stderr)
    return result


def main():
    print(f"[compile] {len(CONCEPTS)} понятий x {len(ATOMS)} атомов -> {OUT}", file=sys.stderr)
    atoms = compile_atoms()
    payload = {"atom_vocab": ATOMS, "concepts": atoms}
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n[compile] сохранено {len(atoms)} понятий в atoms.json")
    # сводка
    sizes = {c: len(v) for c, v in atoms.items()}
    print("  Размер атом-набора (общее = мало атомов, конкретное = много):")
    for c in CONCEPTS:
        if c in sizes:
            print(f"    {c:12s} {sizes[c]:2d}")


if __name__ == "__main__":
    main()
