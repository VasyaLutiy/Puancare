"""
hood.py — «ПОД КАПОТОМ»: прозрачность агента по построению (не логи постфактум).

Слой наблюдаемости, которым МЫ проверяем, что агент реально распознаёт, а не на слово:
  • трейс решения на каждый сигнал (узнал категорию | положил новую | отклонил);
  • кривая: число КАТЕГОРИЙ и доля распознанного по ходу потока;
  • итог + (метрика обучения) категории выходят на плато при росте инстансов.
"""


class Hood:
    def __init__(self):
        self.steps = []          # (i, n_categories, recognition_rate)
        self.recognized = 0
        self.placed = 0
        self.rejected = 0
        self.i = 0

    def record(self, decision: dict, n_categories: int) -> None:
        self.i += 1
        if decision.get("rejected"):
            self.rejected += 1
            tag = f"ОТКЛОНЁН {decision['violations']}"
        elif decision["recognized"]:
            self.recognized += 1
            tag = f"УЗНАЛ {decision['category']} ({decision.get('label')})"
        else:
            self.placed += 1
            tag = f"НОВАЯ КАТЕГОРИЯ {decision['category']} ({decision.get('label')})"
        done = self.recognized + self.placed
        rate = self.recognized / done if done else 0.0
        self.steps.append((self.i, n_categories, rate))
        print(f"  [{self.i:2d}] {decision.get('id'):14} → {tag}")

    def summary(self, kb) -> None:
        n_inst = len(kb.instances)
        n_cat = len(kb.categories)
        done = self.recognized + self.placed
        rate = self.recognized / done if done else 0.0
        print("\n  кривая (i: категорий | доля распознанного):")
        for i, ncat, r in self.steps:
            if i % 5 == 0 or i == self.steps[-1][0]:
                print(f"    i={i:2d}: категорий={ncat}  распознано={r:.0%}")
        print(f"\n  инстансов размещено: {n_inst} | категорий: {n_cat} | "
              f"распознано {self.recognized}/{done} ({rate:.0%}) | отклонено: {self.rejected}")
