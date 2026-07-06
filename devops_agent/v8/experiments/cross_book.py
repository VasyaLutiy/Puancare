"""Межкнижный анализ KB: что вторая книга сделала со знанием первой."""
import sys
from collections import Counter

sys.path.insert(0, "/private/tmp/ccc/devops_agent/v8")
import kb_store

kb = kb_store.load_kb()
nodes = kb["nodes"]

def books_of(n):
    return {("dummies" if s["run"].startswith("dummies") else "iac") for s in n["sources"]}

both = [n for n in nodes.values() if len(books_of(n)) == 2]
only_d = [n for n in nodes.values() if books_of(n) == {"dummies"}]
only_i = [n for n in nodes.values() if books_of(n) == {"iac"}]

print(f"узлов всего: {len(nodes)}")
print(f"  только Dummies: {len(only_d)}")
print(f"  только IaC:     {len(only_i)}")
print(f"  В ОБЕИХ КНИГАХ: {len(both)}  ← межкнижное подкрепление")

print("\nтоп-20 межкнижных узлов (freq, метатип):")
for n in sorted(both, key=lambda n: -n["freq"])[:20]:
    print(f"  {n['id']:32s} {kb_store.meta_type(n):12s} freq={n['freq']}")

# состав метатипов по книгам (в чём разница жанра)
def comp(ns):
    c = Counter(kb_store.meta_type(n) for n in ns)
    t = sum(c.values())
    return {k: f"{v/t:.0%}" for k, v in c.most_common()}
print(f"\nсостав узлов Dummies-only: {comp(only_d)}")
print(f"состав узлов IaC-only:     {comp(only_i)}")

# споры метатипа МЕЖДУ книгами (одно слово — разные роли в разных книгах)
cross_disputes = [n for n in both if len(n["votes"]) > 1]
print(f"\nмежкнижные споры о метатипе: {len(cross_disputes)}")
for n in sorted(cross_disputes, key=lambda n: -n["freq"])[:10]:
    print(f"  {n['id']:32s} votes={n['votes']}")

# рёбра, подтверждённые обеими книгами
be = [e for e in kb["edges"].values()
      if {("dummies" if r.startswith("dummies") else "iac") for r in e["sources"]} == {"dummies", "iac"}]
print(f"\nрёбра, встреченные в ОБЕИХ книгах: {len(be)}")
for e in sorted(be, key=lambda e: -e["freq"])[:10]:
    print(f"  {e['kind']:12s} {e['from']} -> {e['to']}  freq={e['freq']}")
