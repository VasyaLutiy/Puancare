# Puancare

Энергетическое поле смыслов с явным направленным потенциалом абстракции.
Теория и якоря — в [`ResearchPuancare.md`](./ResearchPuancare.md).

## Запуск (VPS / локально)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python step0_tda_gate.py            # TDA-гейт на animal.n.01
python step0_tda_gate.py entity.n.01   # на любом другом subtree
```

## Порядок коммитов (ResearchPuancare.md §8)

- **0. TDA-гейт** — `step0_tda_gate.py` — β₁ noun-subtree. Гейт перед всем. ← *тут*
- 1. Потенциал `h` из noun-hypernymy (Poincaré-стиль).
- 2. Формула стоимости (ген ≈ 0, спец ≈ log|поддерево|).
- 3. Проба асимметрии на held-out парах vs бейзлайн.
- 4. Тупой декодер + проба декодируемости vs BERT.
