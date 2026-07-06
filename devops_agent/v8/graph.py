"""
graph.py — v8 KnowledgeGraph: типизированная упаковка ОДНОЙ меты (задачи/факта).

Написан заново 6 июл (решение Кирилла: не тянуть v4). Отличия от v4-предка:
  • нет мёртвого MetaType.DEPENDENCY (зависимость — ребро, не узел);
  • HAS_GOAL → HAS_STATUS: статус — ось состояния (running/unhealthy), не «цель»;
    цель — отдельное поле graph.goal, валидируется контрактом;
  • новое отношение USES: intervention → entity («инструмент»). Книги дали 2115
    рёбер «действию нужна вещь» — это не грязь, а роль, и она ПРОБУЕМА:
    уронил инструмент → действие деградировало. REQUIRES остаётся чисто состояниям;
  • constraint-слой (mutex/atmost) едет ВМЕСТЕ с графом (self.layer), контракт
    валидирует их одним вызовом;
  • part-of (композиция, кейс Chef-компонентов) СОЗНАТЕЛЬНО в карантине, как if:
    ISA растим, только когда residue накопится и станет пробуемым.

Это упаковка одной меты, НЕ накопительная KB — та живёт в kb_store (merge,
голосование метатипов, слои testimony/grounded, провенанс).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from devops_agent.v8.constraints import ConstraintLayer


class MetaType(str, Enum):
    ENTITY = "entity"
    RESOURCE = "resource"
    SETTING = "setting"
    STATUS = "status"
    INTERVENTION = "intervention"


class Rel(str, Enum):
    HAS = "has"                  # entity → resource/setting
    HAS_STATUS = "has_status"    # entity → status (ось состояния, не «цель»)
    DEPENDS_ON = "depends_on"    # entity → entity (единственный потребитель — топосорт)
    ESTABLISHES = "establishes"  # intervention → resource/setting/status (эффект)
    REQUIRES = "requires"        # intervention → resource/setting/status (предусловие-состояние)
    USES = "uses"                # intervention → entity (инструмент; проба: уронить и смотреть)


@dataclass
class Node:
    id: str
    mtype: MetaType
    attrs: dict = field(default_factory=dict)   # kind, layer, ...


@dataclass(frozen=True)
class Edge:
    src: str
    rel: Rel
    dst: str


@dataclass
class KnowledgeGraph:
    nodes: dict = field(default_factory=dict)            # id -> Node
    edges: list = field(default_factory=list)            # list[Edge]
    goal: list = field(default_factory=list)             # список status-id
    layer: ConstraintLayer = field(default_factory=ConstraintLayer)

    def add_node(self, id: str, mtype, **attrs) -> Node:
        mt = mtype if isinstance(mtype, MetaType) else MetaType(mtype)
        if id in self.nodes:
            n = self.nodes[id]
            if n.mtype != mt:
                raise ValueError(f"узел {id!r} уже есть как {n.mtype.value}, не {mt.value}")
            n.attrs.update(attrs)
            return n
        n = Node(id, mt, dict(attrs))
        self.nodes[id] = n
        return n

    def add_edge(self, src: str, rel, dst: str) -> Edge:
        e = Edge(src, rel if isinstance(rel, Rel) else Rel(rel), dst)
        if e not in self.edges:
            self.edges.append(e)
        return e

    def drop_node(self, nid: str) -> None:
        self.nodes.pop(nid, None)
        self.edges = [e for e in self.edges if e.src != nid and e.dst != nid]

    def nodes_of(self, mtype) -> list:
        mt = mtype if isinstance(mtype, MetaType) else MetaType(mtype)
        return [n for n in self.nodes.values() if n.mtype == mt]

    def ids_of(self, mtype) -> set:
        return {n.id for n in self.nodes_of(mtype)}

    def out(self, src: str, rel=None) -> list:
        r = None if rel is None else (rel if isinstance(rel, Rel) else Rel(rel))
        return [e for e in self.edges if e.src == src and (r is None or e.rel == r)]

    def has_incoming(self, dst: str, rel) -> bool:
        r = rel if isinstance(rel, Rel) else Rel(rel)
        return any(e.dst == dst and e.rel == r for e in self.edges)

    # --- персистентность (cat-абельно = аудируемо) ---
    def to_json(self) -> str:
        return json.dumps(
            {"nodes": [asdict(n) for n in self.nodes.values()],
             "edges": [asdict(e) for e in self.edges],
             "goal": self.goal,
             "constraints": [{"kind": "mutex", "members": list(m.members)} for m in self.layer.mutexes]
                          + [{"kind": "atmost", "members": list(a.members), "bound": a.bound}
                             for a in self.layer.atmosts]},
            ensure_ascii=False, indent=1)
