"""
packer.py — Knowledge Packer: фрагмент знания → validate → commit | reject.

Здесь граф РАСТЁТ из фрагментов (а не строится руками). Фрагмент source-agnostic:
список атомарных утверждений (op-словари) — такой JSON может выдавать LLM, интроспекция,
проба или тест. Packer стейджит фрагмент на копии графа, проверяет КОНТРАКТОМ (структурный
ground-truth), и коммитит только корректное; иначе reject, граф не меняется.

Поглощает v3 mint_dimension: «вшить измерение memory/ordered_monotone» = фрагмент
{entity has resource memory; intervention set_mem establishes memory; deploy requires memory}.

Атомы фрагмента (op):
  {"op":"entity",       "id":E, ...attrs}
  {"op":"resource"|"setting", "id":R, "of":E, ...attrs}
  {"op":"status",       "id":S, "of":E, "goal":true}      # целевой статус (HAS_GOAL)
  {"op":"intervention", "id":I, "establishes":[...], "requires":[...]}
  {"op":"dependency",   "of":E, "on":D}                    # E DEPENDS_ON D
"""

from dataclasses import dataclass, field

from devops_agent.v4.contract import validate
from devops_agent.v4.graph import KnowledgeGraph, MetaType, Rel

_SKIP = {"op", "id", "of", "on", "establishes", "requires", "goal"}


@dataclass
class PackResult:
    ok: bool
    violations: list = field(default_factory=list)
    note: str = ""


def _apply(g: KnowledgeGraph, fragment: list) -> None:
    """Применить атомы фрагмента к графу (идемпотентно — add_node/add_edge дедуплят)."""
    for a in fragment:
        op = a["op"]
        attrs = {k: v for k, v in a.items() if k not in _SKIP}
        if op == "entity":
            g.add_node(a["id"], MetaType.ENTITY, **attrs)
        elif op in ("resource", "setting"):
            g.add_node(a["id"], MetaType(op), **attrs)
            g.add_edge(a["of"], Rel.HAS, a["id"])
        elif op == "status":
            g.add_node(a["id"], MetaType.STATUS, **attrs)
            if a.get("goal"):
                g.add_edge(a["of"], Rel.HAS_GOAL, a["id"])
        elif op == "intervention":
            g.add_node(a["id"], MetaType.INTERVENTION, **attrs)
            for t in a.get("establishes", []):
                g.add_edge(a["id"], Rel.ESTABLISHES, t)
            for t in a.get("requires", []):
                g.add_edge(a["id"], Rel.REQUIRES, t)
        elif op == "dependency":
            g.add_edge(a["of"], Rel.DEPENDS_ON, a["on"])
        else:
            raise ValueError(f"неизвестный op фрагмента: {op!r}")


class Packer:
    """Хранит граф; pack() стейджит+валидирует+коммитит фрагмент или отклоняет (граф цел)."""

    def __init__(self, graph: KnowledgeGraph | None = None):
        self.g = graph or KnowledgeGraph()

    def pack(self, fragment: list) -> PackResult:
        staged = KnowledgeGraph.from_json(self.g.to_json())  # глубокая копия
        try:
            _apply(staged, fragment)
        except (KeyError, ValueError) as e:
            return PackResult(ok=False, violations=[f"малформ-фрагмент: {e}"])
        viol = validate(staged)
        if viol:
            return PackResult(ok=False, violations=viol)      # reject — self.g не тронут
        self.g = staged                                       # commit
        return PackResult(ok=True, note=f"nodes={len(staged.nodes)} edges={len(staged.edges)}")


# ---------------------------------------------------------------------------
# Self-test (item 2 + засев батареи item 3): рост из фрагментов, отказы, идемпотентность,
# композиция (с Dependency → FD секвенирует), поглощение mint_dimension.
# ---------------------------------------------------------------------------

_DB_FRAG = [
    {"op": "entity", "id": "orders_db", "cls": "database"},
    {"op": "resource", "id": "conn_pool", "of": "orders_db", "kind": "ordered_monotone"},
    {"op": "status", "id": "db_ready", "of": "orders_db", "goal": True},
    {"op": "intervention", "id": "set_conn_pool", "establishes": ["conn_pool"]},
    {"op": "intervention", "id": "provision_db", "establishes": ["db_ready"], "requires": ["conn_pool"]},
]

_API_FRAG = [
    {"op": "entity", "id": "api_gw", "cls": "gateway"},
    {"op": "resource", "id": "routes", "of": "api_gw"},
    {"op": "status", "id": "api_ready", "of": "api_gw", "goal": True},
    {"op": "intervention", "id": "set_routes", "establishes": ["routes"]},
    {"op": "intervention", "id": "provision_api", "establishes": ["api_ready"], "requires": ["routes"]},
    {"op": "dependency", "of": "api_gw", "on": "orders_db"},
]

# Поглощение v3 mint_dimension: измерение memory (ordered_monotone) на сервисе как фрагмент.
_MINT_MEMORY_FRAG = [
    {"op": "entity", "id": "svc", "cls": "service"},
    {"op": "resource", "id": "mem", "of": "svc", "kind": "ordered_monotone"},
    {"op": "status", "id": "running", "of": "svc", "goal": True},
    {"op": "intervention", "id": "set_mem", "establishes": ["mem"]},
    {"op": "intervention", "id": "deploy", "establishes": ["running"], "requires": ["mem"]},
]


def _selftest() -> None:
    import os
    import sys

    from devops_agent.v4.projection import kg_to_pddl, solve

    print("=== v4 packer: рост графа из фрагментов ===\n")
    errors = []
    _OUT = os.path.join(os.path.dirname(__file__), "_generated", "packer")

    # 1. валидный фрагмент коммитится
    p = Packer()
    r = p.pack(_DB_FRAG)
    print(f"pack DB: ok={r.ok} {r.note} {r.violations}")
    if not r.ok:
        errors.append(f"валидный DB-фрагмент отклонён: {r.violations}")

    # 2. идемпотентность: повторный pack того же фрагмента не растит граф
    n_nodes, n_edges = len(p.g.nodes), len(p.g.edges)
    p.pack(_DB_FRAG)
    if (len(p.g.nodes), len(p.g.edges)) != (n_nodes, n_edges):
        errors.append(f"идемпотентность нарушена: {(n_nodes, n_edges)} → {(len(p.g.nodes), len(p.g.edges))}")
    else:
        print(f"идемпотентность ✓ (nodes={n_nodes}, edges={n_edges})")

    # 3. малформ-фрагмент отклоняется, граф НЕ меняется
    before = p.g.to_json()
    rbad = p.pack([{"op": "intervention", "id": "noop"}])  # establishes пуст → контракт
    print(f"pack malformed: ok={rbad.ok} viol={rbad.violations}")
    if rbad.ok:
        errors.append("малформ-фрагмент закоммичен (ожидали reject)")
    if p.g.to_json() != before:
        errors.append("граф изменился после reject (должен быть цел)")

    # 4. композиция: добавляем api с Dependency на orders_db → FD секвенирует
    r2 = p.pack(_API_FRAG)
    print(f"pack API: ok={r2.ok} {r2.note}")
    if not r2.ok:
        errors.append(f"API-фрагмент (композиция) отклонён: {r2.violations}")
    else:
        dpath, ppath = kg_to_pddl(p.g, "api_gw", out_dir=_OUT)
        plan = solve(dpath, ppath)
        print(f"  выращенный граф → plan(api_gw): {plan}")
        j = [str(x) for x in (plan or [])]
        try:
            i_db = next(i for i, s in enumerate(j) if "provision_db" in s)
            i_api = next(i for i, s in enumerate(j) if "provision_api" in s)
            if i_db >= i_api:
                errors.append(f"композиция: порядок dep нарушен: {plan}")
            else:
                print(f"  ✓ Dependency держится в выращенном графе: provision_db[{i_db}]<provision_api[{i_api}]")
        except StopIteration:
            errors.append(f"композиция: нет provision_db/provision_api в плане: {plan}")

    # 5. поглощение v3 mint_dimension: фрагмент memory → домен с mem_ok+set_mem+deploy-requires-mem
    pm = Packer()
    rm = pm.pack(_MINT_MEMORY_FRAG)
    if not rm.ok:
        errors.append(f"mint_dimension-фрагмент отклонён: {rm.violations}")
    else:
        dpath, _ = kg_to_pddl(pm.g, "svc", out_dir=os.path.join(_OUT, "mint"))
        dom = open(dpath).read()
        for need in ("mem_ok", "set_mem", "deploy", "(mem_ok svc)"):
            if need not in dom:
                errors.append(f"поглощение mint_dimension: в домене нет {need!r}")
        print(f"\nпоглощение mint_dimension ✓ — memory-измерение выращено фрагментом:\n{dom}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("v4 packer OK — граф растёт из фрагментов; reject цел; идемпотентность; композиция с Dependency; "
          "mint_dimension поглощён.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m devops_agent.v4.packer --selftest")
