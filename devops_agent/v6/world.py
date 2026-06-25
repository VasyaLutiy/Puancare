"""
world.py — генератор снимков: поток ФРАГМЕНТОВ-сущностей (выход NLU/восприятия).

Несколько скрытых шаблонов-категорий (одинаковая СТРУКТУРА, разные инстансы) + новый
шаблон появляется по ходу потока. Это даёт корпус для проверки распознавания/обобщения:
агент должен сформировать КОНЕЧНОЕ число категорий из растущего потока инстансов.
Фрагмент = {id, label, resources[kinds], settings[kinds], status, deps[ids]}.
"""

import random

# скрытые шаблоны: (label, resource-kinds, setting-kinds, status?) — СТРУКТУРА категории
_TEMPLATES = [
    ("database", ["ordered_monotone"], ["categorical"], True),       # диск + auth + ready
    ("cache",    ["ordered_monotone", "bounded"], [], True),         # mem + evict-window
    ("queue",    [], ["categorical"], True),                          # только настройка
    ("worker",   ["ordered_monotone"], [], True),                     # cpu
    ("gateway",  ["bounded"], ["categorical"], True),                 # НОВЫЙ — появится позже
]


def stream(n: int = 30, seed: int = 1, new_at: int = 18):
    """n фрагментов. Сначала 4 шаблона, с new_at добавляется 5-й (gateway). Deps — к ранним инстансам."""
    rng = random.Random(seed)
    placed = []
    for i in range(n):
        pool = _TEMPLATES[: (5 if i >= new_at else 4)]
        label, res, sett, has_status = rng.choice(pool)
        fid = f"{label}_{i}"
        deps = rng.sample(placed, min(len(placed), rng.choice([0, 0, 1, 2]))) if placed else []
        frag = {
            "id": fid,
            "label": label,
            "resources": list(res),
            "settings": list(sett),
            "status": "ready" if has_status else None,
            "deps": deps,
        }
        placed.append(fid)
        yield frag
