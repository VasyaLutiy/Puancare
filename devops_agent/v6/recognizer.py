"""
recognizer.py — ЯДРО агента: структурное распознавание + размещение в DAG.

Линия не-фронтенда: распознавание — ДЕТЕРМИНИРОВАННОЕ, в агенте, по СТРУКТУРНОЙ СИГНАТУРЕ
(не по строке-id, не по семантическому ярлыку LLM). Сигнатура = форма мета-типов
(виды ресурсов + виды настроек + наличие статуса). Одинаковая форма → одна категория →
«узнал, не учу заново». Это и есть обобщение, которого не хватало (E1).

Валидатор (без PDDL): контракт формы фрагмента + DAG-консистентность (ацикличность).
"""

from dataclasses import dataclass, field

RES_KINDS = {"ordered_monotone", "bounded"}
SET_KINDS = {"categorical", "boolean"}


def signature(frag: dict):
    """Структурная сигнатура = форма, БЕЗ id/ярлыка/значений. Ключ категории."""
    return (
        tuple(sorted(frag.get("resources", []))),
        tuple(sorted(frag.get("settings", []))),
        bool(frag.get("status")),
    )


def contract(frag: dict, kb) -> list:
    """Корректность формы фрагмента (без PDDL). Пусто = ок."""
    v = []
    if not isinstance(frag.get("id"), str) or not frag["id"]:
        v.append("нет id")
    if frag.get("id") in kb.instances:
        v.append(f"дубль id {frag.get('id')!r}")
    for r in frag.get("resources", []):
        if r not in RES_KINDS:
            v.append(f"неизвестный resource-kind {r!r}")
    for s in frag.get("settings", []):
        if s not in SET_KINDS:
            v.append(f"неизвестный setting-kind {s!r}")
    for d in frag.get("deps", []):
        if d == frag.get("id"):
            v.append("зависимость на себя")
        elif d not in kb.instances:
            v.append(f"зависимость на несуществующий инстанс {d!r}")
    return v


@dataclass
class KB:
    categories: dict = field(default_factory=dict)   # sig -> cat_id
    cat_shape: dict = field(default_factory=dict)     # cat_id -> shape(для аудита)
    instances: dict = field(default_factory=dict)     # inst_id -> {category, label}
    edges: list = field(default_factory=list)         # (src, dst) depends_on

    def _reaches(self, a, b):
        """достижим ли b из a по edges (для проверки ацикличности)."""
        seen, stack = set(), [a]
        while stack:
            x = stack.pop()
            for s, d in self.edges:
                if s == x:
                    if d == b:
                        return True
                    if d not in seen:
                        seen.add(d); stack.append(d)
        return False

    def to_json(self):
        import json
        return json.dumps({
            "categories": {v: list(k) if isinstance(k, tuple) else k for k, v in self.categories.items()},
            "instances": self.instances,
            "edges": self.edges,
        }, ensure_ascii=False, indent=2, default=str)


class Recognizer:
    def __init__(self, kb: KB | None = None):
        self.kb = kb or KB()

    def observe(self, frag: dict) -> dict:
        kb = self.kb
        viol = contract(frag, kb)
        if viol:
            return {"id": frag.get("id"), "rejected": True, "violations": viol}
        sig = signature(frag)
        is_new = sig not in kb.categories
        if is_new:
            cat_id = f"cat{len(kb.categories)}"
            kb.categories[sig] = cat_id
            kb.cat_shape[cat_id] = {"resources": sorted(frag.get("resources", [])),
                                     "settings": sorted(frag.get("settings", [])),
                                     "status": bool(frag.get("status"))}
        cat_id = kb.categories[sig]
        kb.instances[frag["id"]] = {"category": cat_id, "label": frag.get("label")}
        for d in frag.get("deps", []):
            if kb._reaches(d, frag["id"]):       # ребро создало бы цикл
                return {"id": frag["id"], "rejected": True, "violations": [f"цикл через {d}"]}
            kb.edges.append((frag["id"], d))
        return {"id": frag["id"], "rejected": False, "recognized": not is_new,
                "category": cat_id, "label": frag.get("label"), "via": "structural-signature"}
