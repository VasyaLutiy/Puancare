"""
graph.py — KnowledgeGraph: типизированный граф знаний агента (v4).

Знание живёт ЗДЕСЬ (а не в PDDL — PDDL это его исполнимое/проверяющее лицо).
Узлы типизированы 6 мета-типами (ISA агента), рёбра — отношениями между ними.
Лёгкое представление: dataclasses + dict-узлы + список рёбер + JSON (без graph-БД —
тестируем гипотезу, не инфраструктуру). cat-абельно = аудируемо.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class MetaType(str, Enum):
    ENTITY = "entity"
    RESOURCE = "resource"
    SETTING = "setting"
    STATUS = "status"
    INTERVENTION = "intervention"
    DEPENDENCY = "dependency"


class Rel(str, Enum):
    HAS = "has"                  # entity → resource/setting
    HAS_GOAL = "has_goal"        # entity → status (целевой)
    ESTABLISHES = "establishes"  # intervention → resource/setting/status
    REQUIRES = "requires"        # intervention → resource/setting/status
    DEPENDS_ON = "depends_on"    # entity → entity


@dataclass
class Node:
    id: str
    mtype: MetaType
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    rel: Rel
    dst: str


@dataclass
class KnowledgeGraph:
    nodes: dict = field(default_factory=dict)   # id -> Node
    edges: list = field(default_factory=list)   # list[Edge]

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

    def nodes_of(self, mtype) -> list:
        mt = mtype if isinstance(mtype, MetaType) else MetaType(mtype)
        return [n for n in self.nodes.values() if n.mtype == mt]

    def out(self, src: str, rel=None) -> list:
        r = None if rel is None else (rel if isinstance(rel, Rel) else Rel(rel))
        return [e for e in self.edges if e.src == src and (r is None or e.rel == r)]

    def has_incoming(self, dst: str, rel) -> bool:
        r = rel if isinstance(rel, Rel) else Rel(rel)
        return any(e.dst == dst and e.rel == r for e in self.edges)

    # --- персистентность (str-Enum сериализуется в своё значение автоматически) ---
    def to_json(self) -> str:
        return json.dumps(
            {"nodes": [asdict(n) for n in self.nodes.values()],
             "edges": [asdict(e) for e in self.edges]},
            ensure_ascii=False, indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "KnowledgeGraph":
        d = json.loads(s)
        g = cls()
        for n in d["nodes"]:
            g.nodes[n["id"]] = Node(n["id"], MetaType(n["mtype"]), dict(n.get("attrs", {})))
        for e in d["edges"]:
            g.edges.append(Edge(e["src"], Rel(e["rel"]), e["dst"]))
        return g

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        with open(path) as f:
            return cls.from_json(f.read())
